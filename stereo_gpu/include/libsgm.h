/*
	libsgm.h
*/

#pragma once

/**
* @mainpage stereo-sgm
* See sgm::StereoSGM
*/

/**
* @file libsgm.h
* stereo-sgm main header
*/

namespace sgm {
	struct CudaStereoSGMResources;

	/**
	* @enum DST_TYPE
	* Indicates input/output pointer type.
	*/
	enum EXECUTE_INOUT {
		EXECUTE_INOUT_HOST2HOST = (0 << 1) | 0,	// 0
		EXECUTE_INOUT_HOST2CUDA = (1 << 1) | 0, // 2
		EXECUTE_INOUT_CUDA2HOST = (0 << 1) | 1, // 1
		EXECUTE_INOUT_CUDA2CUDA = (1 << 1) | 1, // 3
	};

	/**
	* @brief StereoSGM class
	*/
	class StereoSGM {
	public:
		/**
		* @param width Processed image's width. It must be even.
		* @param height Processed image's height. It must be even.
		* @param disparity_size It must be 64 or 128.
		* @param input_depth_bits Processed image's bits per pixel. It must be 8 or 16.
		* @param output_depth_bits Disparity image's bits per pixel. It must be 8 or 16.
		* @param inout_type 	Specify input/output pointer type. See sgm::EXECUTE_TYPE.
		*/
		StereoSGM(int width, int height, int disparity_size, EXECUTE_INOUT inout_type);

		virtual ~StereoSGM();

		/**
		* Execute stereo semi global matching.
		* @param left_pixels	A pointer stored input left image.
		* @param right_pixels	A pointer stored input right image.
		* @param dst	        Output pointer. User must allocate enough memory.
		* @attention
		* For performance reason, when the instance is created with inout_type == EXECUTE_INOUT_**2CUDA, output_depth_bits == 16, 
		* you don't have to allocate dst memory yourself. It returns internal cuda pointer. You must not free the pointer.
		*/
		void execute(const void* left_pixels, const void* right_pixels, void** dst);

	private:
		StereoSGM(const StereoSGM&);
		StereoSGM& operator=(const StereoSGM&);

		void cuda_resource_allocate();

		CudaStereoSGMResources* cu_res_;

		int width_;
		int height_;
		int disparity_size_;
		EXECUTE_INOUT inout_type_;
	};
}
