/*
	internal.h
*/

#ifndef _INTERNAL_H
#define _INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CudaSafeCall(error) sgm::details::cuda_safe_call(error, __FILE__, __LINE__)

#define CudaKernelCheck() CudaSafeCall(cudaGetLastError())

namespace sgm {
	namespace details {

		// input: original image	output: image after census transform
		void census(const void* d_src, uint64_t* d_dst, int width, int height, cudaStream_t cuda_stream);

		// input: census image 		output: matching cost
		void matching_cost(const uint64_t* d_left, const uint64_t* d_right, uint8_t* d_matching_cost, int width, int height, int disp_size);

		// input: matching cost 	output: scan_scost
		// This function consumes the cuda_stream
		void scan_scost(const uint8_t* d_matching_cost, uint16_t* d_scost, int width, int height, int disp_size, cudaStream_t cuda_streams[]);

		// input: scan_scost		output: disparity
		void winner_takes_all(const uint16_t* d_scost, uint16_t* d_left_disp, uint16_t* d_right_disp, int width, int height, int disp_size);

		// input: disparity			output: filtered disparity
		void median_filter(const uint16_t* d_src, uint16_t* d_dst, void* median_filter_buffer, int width, int height);

		// output: updated disparity
		void check_consistency(uint16_t* d_left_disp, const uint16_t* d_right_disp, const void* d_src_left, int width, int height);

		void cast_16bit_8bit_array(const uint16_t* arr16bits, uint8_t* arr8bits, int num_elements);

		inline void cuda_safe_call(cudaError error, const char *file, const int line)
		{
			if (error != cudaSuccess) {
				fprintf(stderr, "cuda error %s : %d %s\n", file, line, cudaGetErrorString(error));
				exit(-1);
			}
		}

	}
}

#endif
