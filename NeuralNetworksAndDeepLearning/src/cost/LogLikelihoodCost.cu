
#include "LogLikelihoodCost.h"
#include <cfloat>


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


template <typename Dtype>
__global__ void SoftmaxLossBackprop(const uint32_t* label, int num_labels, int batch_size, Dtype *diff) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch_size)
		return;
	const int label_value = static_cast<int>(label[idx]);
	// For each item in the batch, decrease the result of the label's value by 1
	diff[idx * num_labels + label_value] -= 1.0f;
}

/*
template <typename Dtype>
__global__ void SoftmaxLossBackprop(
		const Dtype *z,
		const Dtype *activation,
		const uint32_t *target,
		Dtype *delta,
		uint32_t numLabels,
		uint32_t batchsize) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batchsize) return;
	const int targetValue = static_cast<int>(target[idx]);
	// For each item in the batch, decrease the result of the label's value by 1

	Dtype ayL = activation[idx*numLabels + targetValue];
	if(ayL < Dtype(FLT_MIN)) {
		ayL = Dtype(FLT_MIN);
	}
	delta[idx * numLabels + targetValue] = -1.0/ayL;
	//delta[idx * numLabels + targetValue] = -1.0/std::max(activation[idx*numLabels + targetValue], Dtype(FLT_MAX));
}
*/

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
void LogLikelihoodCost<Dtype>::d_cost(const rcube &z, const rcube &activation, const rvec &target, rcube &delta) {
	Util::printCube(activation, "activation:");
	Util::printVec(target, "target:");

	delta.slice(0) = activation.slice(0) - target;
	Util::printCube(delta, "delta:");
}
#else

template <typename Dtype>
double LogLikelihoodCost<Dtype>::forward(const Dtype* output, const uint32_t* target, const uint32_t numLabels, const uint32_t batchsize) {
	double cost = 0.0;
	for(uint32_t batchIndex = 0; batchIndex < batchsize; batchIndex++) {
		cost -= std::log(std::max(output[batchIndex*numLabels+target[batchIndex]], Dtype(FLT_MIN)));
	}
	return cost;
}

template <typename Dtype>
void LogLikelihoodCost<Dtype>::backward(const Dtype *z, const Dtype *activation, const uint32_t *target, Dtype *delta, UINT numLabels, UINT batchsize) {
	checkCudaErrors(cudaMemcpyAsync(delta, activation, sizeof(Dtype)*numLabels*batchsize, cudaMemcpyDeviceToDevice));
	//SoftmaxLossBackprop<<<RoundUp(batchsize, BW), BW>>>(z, activation, target, delta, numLabels, batchsize);
	SoftmaxLossBackprop<<<RoundUp(batchsize, BW), BW>>>(target, numLabels, batchsize, delta);
}


template class LogLikelihoodCost<float>;

#endif



















