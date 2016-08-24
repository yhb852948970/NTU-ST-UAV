/*
	census transform
*/

#include <iostream>

#include "internal.h"

namespace {

	static const int HOR = 9;
	static const int VERT = 7;

	static const int threads_per_block = 16;

	__global__
		void census_kernel(uint8_t* d_source, uint64_t* d_dest, int width, int height)
	{
		const int i = threadIdx.y + blockIdx.y * blockDim.y;
		const int j = threadIdx.x + blockIdx.x * blockDim.x;
		const int offset = j + i * width;

		const int rad_h = HOR / 2;
		const int rad_v = VERT / 2;

		// Modified by Haibo, reduce both the size by 1.
		const int swidth = threads_per_block + HOR - 1;
		const int sheight = threads_per_block + VERT - 1;

		//const int swidth = threads_per_block + HOR;
		//const int sheight = threads_per_block + VERT;

		__shared__ uint8_t s_source[swidth*sheight];

		/**
		*                  *- blockDim.x
		*                 /
		*      +---------+---+ -- swidth (blockDim.x+HOR)
		*      |         |   |
		*      |    1    | 2 |
		*      |         |   |
		*      +---------+---+ -- blockDim.y
		*      |    3    | 4 |
		*      +---------+---+ -- sheight (blockDim.y+VERT)
		*/

		// 1. left-top side
		const int ii = threadIdx.y + blockIdx.y * blockDim.y - rad_v;
		const int jj = threadIdx.x + blockIdx.x * blockDim.x - rad_h;
		if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
			s_source[threadIdx.y*swidth + threadIdx.x] = d_source[ii*width + jj];
		}

		// 2. right side
		// 2 * blockDim.x >= swidth
		{
			const int ii = threadIdx.y + blockIdx.y * blockDim.y - rad_v;
			const int jj = threadIdx.x + blockIdx.x * blockDim.x - rad_h + blockDim.x;
			if (threadIdx.x + blockDim.x < swidth && threadIdx.y < sheight) {
				if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
					s_source[threadIdx.y*swidth + threadIdx.x + blockDim.x] = d_source[ii*width + jj];
				}
			}
		}

		// 3. bottom side
		// 2 * blockDim.y >= sheight
		{
			const int ii = threadIdx.y + blockIdx.y * blockDim.y - rad_v + blockDim.y;
			const int jj = threadIdx.x + blockIdx.x * blockDim.x - rad_h;
			if (threadIdx.x < swidth && threadIdx.y + blockDim.y < sheight) {
				if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
					s_source[(threadIdx.y + blockDim.y)*swidth + threadIdx.x] = d_source[ii*width + jj];
				}
			}
		}

		// 4. right-bottom side
		// 2 * blockDim.x >= swidth && 2 * blockDim.y >= sheight
		{
			const int ii = threadIdx.y + blockIdx.y * blockDim.y - rad_v + blockDim.y;
			const int jj = threadIdx.x + blockIdx.x * blockDim.x - rad_h + blockDim.x;
			if (threadIdx.x + blockDim.x < swidth && threadIdx.y + blockDim.y < sheight) {
				if (ii >= 0 && ii < height && jj >= 0 && jj < width) {
					s_source[(threadIdx.y + blockDim.y)*swidth + threadIdx.x + blockDim.x] = d_source[ii*width + jj];
				}
			}
		}
		__syncthreads();

		// TODO can we remove this condition?
		if (rad_v <= i && i < height - rad_v && rad_h <= j && j < width - rad_h)
		{
			const int ii = threadIdx.y + rad_v;
			const int jj = threadIdx.x + rad_h;
			const int soffset = jj + ii * swidth;
			// const SRC_T c = d_source[offset];
			const uint8_t c = s_source[soffset];
			uint64_t value = 0;

			uint32_t value1 = 0, value2 = 0;

#pragma unroll
			for (int y = -rad_v; y < 0; y++) {
				for (int x = -rad_h; x <= rad_h; x++) {
					// SRC_T result = (c - d_source[width*(i+y)+j+x])>0;
					uint8_t result = (c - s_source[swidth*(ii + y) + jj + x]) > 0;
					value1 <<= 1;
					value1 += result;
				}
			}

			int y = 0;
#pragma unroll
			for (int x = -rad_h; x < 0; x++) {
				// uint8_t result = (c - d_source[width*(i+y)+j+x])>0;
				uint8_t result = (c - s_source[swidth*(ii + y) + jj + x]) > 0;
				value1 <<= 1;
				value1 += result;
			}

#pragma unroll
			for (int x = 1; x <= rad_h; x++) {
				// uint8_t result = (c - d_source[width*(i+y)+j+x])>0;
				uint8_t result = (c - s_source[swidth*(ii + y) + jj + x]) > 0;
				value2 <<= 1;
				value2 += result;
			}

#pragma unroll
			for (int y = 1; y <= rad_v; y++) {
				for (int x = -rad_h; x <= rad_h; x++) {
					// uint8_t result = (c - d_source[width*(i+y)+j+x])>0;
					uint8_t result = (c - s_source[swidth*(ii + y) + jj + x]) > 0;
					value2 <<= 1;
					value2 += result;
				}
			}

			value = (uint64_t)value2;
			value |= (uint64_t)value1 << (rad_v * (2 * rad_h + 1) + rad_h);

			d_dest[offset] = value;
		}
	}
}


namespace sgm {
namespace details {

		void census(
			const void* d_src, uint64_t* d_dst, int width, int height, cudaStream_t cuda_stream) {

			const dim3   blocks((width + threads_per_block - 1) / threads_per_block, (height + threads_per_block - 1) / threads_per_block);
			const dim3   threads(threads_per_block, threads_per_block);

			census_kernel<< <blocks, threads, 0, cuda_stream >> > ((uint8_t*)d_src, d_dst, width, height);

		}

	}
}
