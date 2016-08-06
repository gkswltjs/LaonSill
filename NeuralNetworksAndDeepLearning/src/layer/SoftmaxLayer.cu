/*
 * SoftmaxLayer.cpp
 *
 *  Created on: 2016. 8. 1.
 *      Author: jhkim
 */

#include "SoftmaxLayer.h"



///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
__global__ void Dropout_(const int n, const DATATYPE* in, const DATATYPE* mask,
		const unsigned int threashold, const float scale, DATATYPE *out)
{

	CUDA_KERNEL_LOOP(index, n) {
		//out[index] = in[index] * (mask[index] > threshold) * scale;
		out[index] = in[index] * (mask[index]) * scale;
	}
}



SoftmaxLayer::SoftmaxLayer() {
	this->type = LayerType::Softmax;
}

SoftmaxLayer::SoftmaxLayer(const char *name, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler)
	: OutputLayer(name, n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler,
			ActivationType::Softmax, CostType::LogLikelihood) {
	initialize();
}

SoftmaxLayer::~SoftmaxLayer() {}

void SoftmaxLayer::load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	OutputLayer::load(ifs, layerMap);
	initialize();
	SoftmaxLayer::_shape(false);
}





void SoftmaxLayer::cost(const UINT *target) {
	Util::printMessage("SoftmaxLayer::cost()---"+string(name));
	Cuda::refresh();

	cost_fn->d_cost(d_z, d_output, target, d_delta, out_dim.rows, out_dim.batches);


	//Util::setPrint(true);
	Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "d_delta:");
	//Util::setPrint(false);
	// Accounting for batch size in SGD
	// checkCudaErrors(cublasSscal(cublasHandle, ref_fc2.outputs * m_batchSize, &scalVal, dloss_data, 1));




	/*
	if(Util::train && p_dropout < 1.0f) {
		//Util::setPrint(true);
		Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "delta_input:");
		Dropout_<<<RoundUp(out_dim.batchsize(), BW), BW>>>(out_dim.batchsize(), d_delta, d_mask, 0, scale, d_delta);


		Util::printData(mask, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/mask:"));
		//DATATYPE *next_delta_input = next_layer->getDeltaInput();
		Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "delta_input:");
		//Util::setPrint(false);
	}
	*/




	float alpha = 1.0f, beta = 0.0f;
	Util::printDeviceData(d_input, in_dim.rows, in_dim.batches, 1, 1, "d_input:");
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, out_dim.rows, in_dim.rows, out_dim.batches,
			&alpha, d_delta, out_dim.rows, d_input, in_dim.rows, &beta, d_delta_weight, out_dim.rows));
	Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight:");

	checkCudaErrors(cublasSgemv(Cuda::cublasHandle, CUBLAS_OP_N, out_dim.rows, out_dim.batches,
			&alpha, d_delta, out_dim.rows, d_onevec, 1, &beta, d_delta_bias, 1));
	Util::printDeviceData(d_delta_bias, out_dim.rows, 1, 1, 1, "d_delta_bias:");

	Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "d_delta:");
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, in_dim.rows, out_dim.batches, out_dim.rows,
			&alpha, d_weight, out_dim.rows, d_delta, out_dim.rows, &beta, d_delta_input, in_dim.rows));

	Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.batches, 1, 1, "d_delta_input:");

	propBackpropagation();
}








void SoftmaxLayer::initialize() {
	this->type = LayerType::Softmax;

	//this->cost_fn = CostFactory::create(CostType::LogLikelihood);
	//this->activation_fn = ActivationFactory::create(ActivationType::Softmax);
	//this->activation_fn->initialize_weight(in_dim.size(), weight);

	//weight.zeros();
	//bias.zeros();
}

void SoftmaxLayer::_shape(bool recursive) {
	if(recursive) {
		OutputLayer::_shape();
	}
}

void SoftmaxLayer::_clearShape() {
	OutputLayer::_clearShape();
}
