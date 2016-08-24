#include <iostream>
#include <nppi.h>
#include "libsgm.h"
#include "internal.h"

namespace sgm {
	static bool is_cuda_input(EXECUTE_INOUT type) { return (int)type & 0x1; }
	static bool is_cuda_output(EXECUTE_INOUT type) { return (int)type & 0x2; }

	struct CudaStereoSGMResources {
		void* d_src_left;	//original image
		void* d_src_right;
		void* d_left;		// census image
		void* d_right;
		void* d_matching_cost;
		void* d_scost;		// core part for calculating the cost in 8 directions
		void* d_left_disp;
		void* d_right_disp;
		void* d_median_left_disp;
		void* d_median_right_disp;

		cudaStream_t cuda_streams[8];

		Npp32u median_buffer_size;
		void* d_median_filter_buffer;

		// void* d_output_16bit_buffer;			//comment out by Haibo
		uint16_t* h_output_16bit_buffer;		// do we need to change it to 8-bit?

		CudaStereoSGMResources(int width_, int height_, int disparity_size_, EXECUTE_INOUT inout_type_) {

			if (is_cuda_input(inout_type_)) {
				this->d_src_left = NULL;
				this->d_src_right = NULL;
			}
			else {
				CudaSafeCall(cudaMalloc(&this->d_src_left, width_ * height_));
				CudaSafeCall(cudaMalloc(&this->d_src_right, width_ * height_));
			}
			
			CudaSafeCall(cudaMalloc(&this->d_left, sizeof(uint64_t) * width_ * height_));
			CudaSafeCall(cudaMalloc(&this->d_right, sizeof(uint64_t) * width_ * height_));
			CudaSafeCall(cudaMalloc(&this->d_matching_cost, sizeof(uint8_t) * width_ * height_ * disparity_size_));
			CudaSafeCall(cudaMalloc(&this->d_scost, sizeof(uint16_t) * width_ * height_ * disparity_size_));
			CudaSafeCall(cudaMalloc(&this->d_left_disp, sizeof(uint16_t) * width_ * height_));
			CudaSafeCall(cudaMalloc(&this->d_right_disp, sizeof(uint16_t) * width_ * height_));
			CudaSafeCall(cudaMalloc(&this->d_median_left_disp, sizeof(uint16_t) * width_ * height_));
			CudaSafeCall(cudaMalloc(&this->d_median_right_disp, sizeof(uint16_t) * width_ * height_));

			for (int i = 0; i < 8; i++) {
				CudaSafeCall(cudaStreamCreate(&this->cuda_streams[i]));
			}

			NppiSize roi = { width_, height_ };
			NppiSize mask = { 3, 3 }; // width, height
			NppStatus status;
			status = nppiFilterMedianGetBufferSize_16u_C1R(roi, mask, &this->median_buffer_size);
			if (status != NPP_SUCCESS) {
				throw std::runtime_error("nppi error");
			}
			CudaSafeCall(cudaMalloc(&this->d_median_filter_buffer, this->median_buffer_size));

			// create temporary buffer when dst type is host pointer
			if (!is_cuda_output(inout_type_)) {
				this->h_output_16bit_buffer = (uint16_t*)malloc(sizeof(uint16_t) * width_ * height_);
			}
			else {
				this->h_output_16bit_buffer = NULL;
			}
		}

		~CudaStereoSGMResources() {
			CudaSafeCall(cudaFree(this->d_src_left));
			CudaSafeCall(cudaFree(this->d_src_right));
			CudaSafeCall(cudaFree(this->d_left));
			CudaSafeCall(cudaFree(this->d_right));
			CudaSafeCall(cudaFree(this->d_matching_cost));
			CudaSafeCall(cudaFree(this->d_scost));
			CudaSafeCall(cudaFree(this->d_left_disp));
			CudaSafeCall(cudaFree(this->d_right_disp));
			CudaSafeCall(cudaFree(this->d_median_left_disp));
			CudaSafeCall(cudaFree(this->d_median_right_disp));

			for (int i = 0; i < 8; i++) {
				CudaSafeCall(cudaStreamDestroy(this->cuda_streams[i]));
			}
			CudaSafeCall(cudaFree(this->d_median_filter_buffer));

			free(h_output_16bit_buffer);
		}
	};

	StereoSGM::StereoSGM(int width, int height, int disparity_size, EXECUTE_INOUT inout_type) :
		width_(width),
		height_(height),
		disparity_size_(disparity_size),
		inout_type_(inout_type) //,
//		cu_res_(NULL)
	{
		cu_res_ = new CudaStereoSGMResources(width_, height_, disparity_size_, inout_type_);
	}

	StereoSGM::~StereoSGM() {
		delete cu_res_;
	}

	void StereoSGM::execute(const void* left_pixels, const void* right_pixels, void** dst) {

		const void *d_input_left, *d_input_right;
		
		if (is_cuda_input(inout_type_)) {
			d_input_left = left_pixels;
			d_input_right = right_pixels;
		}
		else {
			CudaSafeCall(cudaMemcpy(cu_res_->d_src_left, left_pixels, width_ * height_, cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpy(cu_res_->d_src_right, right_pixels, width_ * height_, cudaMemcpyHostToDevice));
			d_input_left = cu_res_->d_src_left;
			d_input_right = cu_res_->d_src_right;
		}

		CudaSafeCall(cudaMemsetAsync(cu_res_->d_left_disp, 0, sizeof(uint16_t) * width_ * height_, cu_res_->cuda_streams[2]));
		CudaSafeCall(cudaMemsetAsync(cu_res_->d_right_disp, 0, sizeof(uint16_t) * width_ * height_, cu_res_->cuda_streams[3]));
		CudaSafeCall(cudaMemsetAsync(cu_res_->d_scost, 0, sizeof(uint16_t) * width_ * height_ * disparity_size_, cu_res_->cuda_streams[4]));

		sgm::details::census(d_input_left, (uint64_t*)cu_res_->d_left, width_, height_, cu_res_->cuda_streams[0]);
		sgm::details::census(d_input_right, (uint64_t*)cu_res_->d_right, width_, height_, cu_res_->cuda_streams[1]);

		sgm::details::matching_cost((const uint64_t*)cu_res_->d_left, (const uint64_t*)cu_res_->d_right, (uint8_t*)cu_res_->d_matching_cost, width_, height_, disparity_size_);

		sgm::details::scan_scost((const uint8_t*)cu_res_->d_matching_cost, (uint16_t*)cu_res_->d_scost, width_, height_, disparity_size_, cu_res_->cuda_streams);
		// the function scan_scost consumes all the cuda_streams

		sgm::details::winner_takes_all((const uint16_t*)cu_res_->d_scost, (uint16_t*)cu_res_->d_left_disp, (uint16_t*)cu_res_->d_right_disp, width_, height_, disparity_size_);

		sgm::details::median_filter((uint16_t*)cu_res_->d_left_disp, (uint16_t*)cu_res_->d_median_left_disp, cu_res_->d_median_filter_buffer, width_, height_);
		sgm::details::median_filter((uint16_t*)cu_res_->d_right_disp, (uint16_t*)cu_res_->d_median_right_disp, cu_res_->d_median_filter_buffer, width_, height_);

		sgm::details::check_consistency((uint16_t*)cu_res_->d_median_left_disp, (uint16_t*)cu_res_->d_median_right_disp, d_input_left, width_, height_);

		// output disparity image
		void* disparity_image = cu_res_->d_median_left_disp;

		if (!is_cuda_output(inout_type_)) {
			CudaSafeCall(cudaMemcpy(cu_res_->h_output_16bit_buffer, disparity_image, sizeof(uint16_t) * width_ * height_, cudaMemcpyDeviceToHost));
			for (int i = 0; i < width_ * height_; i++) { ((uint8_t*)*dst)[i] = (uint8_t)cu_res_->h_output_16bit_buffer[i]; }
		}

		else {
			sgm::details::cast_16bit_8bit_array((const uint16_t*)disparity_image, (uint8_t*)*dst, width_ * height_);
		}

	}
}


//cudaEvent_t start, stop;
//float time;
//cudaEventCreate(&start);
//cudaEventCreate(&stop);
//cudaEventRecord( start, 0 );
//

//cudaEventRecord( stop, 0 );
//cudaEventSynchronize( stop );
//cudaEventElapsedTime( &time, start, stop );
//cudaEventDestroy( start );
//cudaEventDestroy( stop );
//std::cout << time << std::endl;
