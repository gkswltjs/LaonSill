
#include <cfloat>

#include "LogLikelihoodCost.h"

using namespace std;

#ifdef GPU_MODE

/**
 * Computes the backpropagation results of the Softmax loss for each result in a batch.
 * Uses the softmax values obtained from forward propagation to compute the difference.
 *
 * @param label The training batch label values.
 * @param num_labels The number of possible labels.
 * @param batch_size The size of the trained batch.
 * @param diff The resulting gradient.
 */


/*
template <typename Dtype>
__global__ void SoftmaxLossBackprop(const uint32_t* label, int num_labels, int batch_size,
    Dtype *diff) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch_size)
		return;
	const int label_value = static_cast<int>(label[idx]);
	// For each item in the batch, decrease the result of the label's value by 1
	diff[idx * num_labels + label_value] -= 1.0f;
}
*/

template <typename Dtype>
__global__ void SoftmaxLossBackprop(
		const Dtype* z,
		const Dtype* activation,
		const Dtype* target,
		Dtype *delta,
		uint32_t numLabels,
		uint32_t batchsize) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batchsize) return;
	const uint32_t targetValue = static_cast<uint32_t>(target[idx]+0.0001f);
	// For each item in the batch, decrease the result of the label's value by 1

	Dtype ayL = activation[idx*numLabels + targetValue];
	if(ayL < Dtype(FLT_MIN)) {
		ayL = Dtype(FLT_MIN);
	}

	delta[idx * numLabels + targetValue] = -1.0/ayL;
}

#endif


template <typename Dtype>
LogLikelihoodCost<Dtype>::LogLikelihoodCost() {
	this->type = Cost<Dtype>::LogLikelihood;
}

template <typename Dtype>
LogLikelihoodCost<Dtype>::~LogLikelihoodCost() {}

#ifndef GPU_MODE
template <typename Dtype>
double LogLikelihoodCost<Dtype>::fn(const rvec *pA, const rvec *pY) {
	return 0.0;
}

template <typename Dtype>
void LogLikelihoodCost<Dtype>::d_cost(const rcube &z, const rcube &activation,
    const rvec &target, rcube &delta) {
	Util::printCube(activation, "activation:");
	Util::printVec(target, "target:");

	delta.slice(0) = activation.slice(0) - target;
	Util::printCube(delta, "delta:");
}
#else

template <typename Dtype>
double LogLikelihoodCost<Dtype>::forward(const Dtype* output, const Dtype* target,
		const uint32_t numLabels, const uint32_t batchsize) {
	double cost = 0.0;
	uint32_t label;
	for(uint32_t batchIndex = 0; batchIndex < batchsize; batchIndex++) {
		label = static_cast<uint32_t>(target[batchIndex]+0.1);
		cost -= log(max(output[batchIndex*numLabels+label], Dtype(FLT_MIN)));
	}
	return cost;
}

template <typename Dtype>
void LogLikelihoodCost<Dtype>::backward(const Dtype* z, const Dtype* activation,
		const Dtype* target, Dtype* delta, uint32_t numLabels, uint32_t batchsize) {
	SoftmaxLossBackprop<<<SOOOA_GET_BLOCKS(batchsize), SOOOA_CUDA_NUM_THREADS>>>(
			z, activation, target, delta, numLabels, batchsize);
	CUDA_POST_KERNEL_CHECK;
}

template class LogLikelihoodCost<float>;

#endif
