#include "MathFunctions.h"
#include "Cuda.h"

template <typename Dtype>
__global__ void _diff_content_loss(const uint32_t n, const Dtype* f,
    const Dtype* p, Dtype* df) {
  CUDA_KERNEL_LOOP(index, n) {
	  if (f[index] > 0)
		  df[index] = f[index] - p[index];
	  else
		  df[index] = 0;
  }
}

template <typename Dtype>
void diff_content_loss(const uint32_t n, const Dtype* f,
    const Dtype* p, Dtype* df) {
	_diff_content_loss<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(
		n, f, p, df);
}

template void diff_content_loss<float>(const uint32_t n, const float* f,
		const float* p, float* df);


template <typename Dtype>
__global__ void _diff_style_loss(const uint32_t n, const Dtype* f, Dtype* a) {
	CUDA_KERNEL_LOOP(index, n) {
		if (f[index] < 0)
			a[index] = 0;
	}
}

template <typename Dtype>
void diff_style_loss(const uint32_t n, const Dtype* f, Dtype* a) {
	_diff_style_loss<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(
			n, f, a);
}

template void diff_style_loss<float>(const uint32_t n, const float* f, float* a);


template <typename Dtype>
__global__ void _fill_channel_mean(const uint32_t n, const uint32_t singleChannelSize,
		const Dtype* mean, Dtype* dst) {
	CUDA_KERNEL_LOOP(index, n) {
		int channel = index / singleChannelSize;
		dst[index] = mean[channel];
	}
}

//ignore_if_le_than_zero<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(n, f, a);

template <typename Dtype>
void fill_channel_mean(const uint32_t n, const uint32_t singleChannelSize,
		const Dtype* mean, Dtype* dst) {
	_fill_channel_mean<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(
			n, singleChannelSize, mean, dst);
}

template void fill_channel_mean(const uint32_t n, const uint32_t singleChannelSize,
		const float* mean, float* dst);



template <typename Dtype>
__global__ void _bound_data(const uint32_t n, const uint32_t singleChannelSize,
		const Dtype* dataMin, const Dtype* dataMax, Dtype* data) {
	CUDA_KERNEL_LOOP(index, n) {
		int channel = index / singleChannelSize;
		if (data[index] > dataMax[channel])
			data[index] = dataMax[channel];
		else if (data[index] < dataMin[channel])
			data[index] = dataMin[channel];
	}
}

template <typename Dtype>
void bound_data(const uint32_t n, const uint32_t singleChannelSize, const Dtype* dataMin,
		const Dtype* dataMax, Dtype* data) {
	_bound_data<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(n, singleChannelSize,
			dataMin, dataMax, data);
}

template void bound_data(const uint32_t n, const uint32_t singleChannelSize,
		const float* dataMin, const float* dataMax, float* data);



















































