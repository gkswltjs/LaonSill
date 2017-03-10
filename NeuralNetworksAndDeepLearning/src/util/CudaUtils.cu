#include "CudaUtils.h"
#include "Cuda.h"


template <typename Dtype>
__global__ void _soooa_sub_channel_mean(const uint32_t n, const uint32_t singleChannelSize,
		const Dtype* mean, Dtype* data) {
	CUDA_KERNEL_LOOP(index, n) {
		int channel = index / singleChannelSize;
		data[index] = data[index] - mean[channel];
	}
}

template <typename Dtype>
void soooa_sub_channel_mean(const int N, const uint32_t singleChannelSize, const Dtype *mean,
		Dtype *data) {
	_soooa_sub_channel_mean<Dtype><<<SOOOA_GET_BLOCKS(N), SOOOA_CUDA_NUM_THREADS>>>(
			N, singleChannelSize, mean, data);
}

template void soooa_sub_channel_mean(const int N, const uint32_t singleChannelSize,
		const float* mean, float* data);
