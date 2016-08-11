
#include "LogLikelihoodCost.h"


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
__global__ void SoftmaxLossBackprop(const UINT *label, int num_labels, int batch_size, DATATYPE *diff) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch_size)
		return;
	const int label_value = static_cast<int>(label[idx]);
	// For each item in the batch, decrease the result of the label's value by 1
	diff[idx * num_labels + label_value] -= 1.0f;
}

#endif



LogLikelihoodCost::LogLikelihoodCost() {
		this->type = CostType::LogLikelihood;
}
LogLikelihoodCost::~LogLikelihoodCost() {}

#ifndef GPU_MODE
double LogLikelihoodCost::fn(const rvec *pA, const rvec *pY) {
	return 0.0;
}
void LogLikelihoodCost::d_cost(const rcube &z, const rcube &activation, const rvec &target, rcube &delta) {
	Util::printCube(activation, "activation:");
	Util::printVec(target, "target:");

	delta.slice(0) = activation.slice(0) - target;
	Util::printCube(delta, "delta:");
}
#else

double LogLikelihoodCost::fn(const DATATYPE *pA, const DATATYPE *pY) {
	return 0.0;
}

//cost_fn->d_cost(d_z, d_output, target, d_delta, out_dim.rows, out_dim.batches);
void LogLikelihoodCost::d_cost(const DATATYPE *z, DATATYPE *activation, const UINT *target, DATATYPE *delta, UINT numLabels, UINT batchsize) {
	Cuda::refresh();

	Util::printDeviceData(activation, numLabels, 1, 1, batchsize, "activation:");

	checkCudaErrors(cudaMemcpyAsync(delta, activation, sizeof(DATATYPE)*numLabels*batchsize, cudaMemcpyDeviceToDevice));
	SoftmaxLossBackprop<<<RoundUp(batchsize, BW), BW>>>(target, numLabels, batchsize, delta);

	Util::printDeviceData(delta, numLabels, 1, 1, batchsize, "activation:");
	checkCudaErrors(cudaDeviceSynchronize());
}
#endif










