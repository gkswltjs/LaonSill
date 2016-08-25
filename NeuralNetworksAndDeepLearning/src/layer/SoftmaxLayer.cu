/*
 * SoftmaxLayer.cpp
 *
 *  Created on: 2016. 8. 1.
 *      Author: jhkim
 */

#include "SoftmaxLayer.h"



#ifdef GPU_MODE
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
#endif





SoftmaxLayer::SoftmaxLayer() {
	this->type = Layer::Softmax;
}

SoftmaxLayer::SoftmaxLayer(Builder* builder)
	: OutputLayer(builder) {
	initialize();
}

SoftmaxLayer::SoftmaxLayer(const string name, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler)
	: OutputLayer(name, n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler,
			Activation::Softmax, Cost::LogLikelihood) {
	initialize();
}
#ifndef GPU_MODE
SoftmaxLayer::SoftmaxLayer(const string name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler)
	: OutputLayer(name, n_in, n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler,
			Activation::Softmax, Cost::LogLikelihood) {
	initialize();
}
#endif
SoftmaxLayer::~SoftmaxLayer() {}





#ifndef GPU_MODE
void SoftmaxLayer::cost(const rvec &target) {
	// delta
	cost_fn->d_cost(z, output, target, delta);
	Util::printVec(nabla_b, "bias:");
	Util::printMat(nabla_w, "weight");
	Util::printCube(delta, "delta:");
	Util::printCube(input, "input:");
	nabla_b += delta.slice(0);
	// delta weight
	nabla_w += delta.slice(0)*input.slice(0).t();

	// delta input
	delta_input.slice(0) = weight.t()*delta.slice(0);

	propBackpropagation();
}
#else
void SoftmaxLayer::cost(const uint32_t* target) {
	Util::printMessage("SoftmaxLayer::cost()---"+string(name));

	_target.set_mem(target, SyncMemCopyType::HostToDevice);
	//_target.print("target:");

	const DATATYPE* d_z = _preActivation->device_data();
	const DATATYPE* d_output = _output->device_data();
	const uint32_t* d_target = _target.device_mem();
	DATATYPE* d_delta = _preActivation->mutable_device_grad();

	cost_fn->d_cost(d_z, d_output, d_target, d_delta, out_dim.rows, out_dim.batches);

	//Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "d_delta:");
	_preActivation->print_grad("d_delta:");


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

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.batches, 1, 1, "d_input:");
	_input->print_data("d_input:");
	const DATATYPE* d_input = _input->device_data();
	DATATYPE* d_delta_weight = _params[Weight]->mutable_device_grad();
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, out_dim.rows, in_dim.rows, out_dim.batches,
			&Cuda::alpha, d_delta, out_dim.rows, d_input, in_dim.rows, &Cuda::beta, d_delta_weight, out_dim.rows));
	//Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight:");
	_params[Weight]->print_grad("d_delta_weight:");

	DATATYPE* d_delta_bias = _params[Bias]->mutable_device_grad();
	checkCudaErrors(cublasSgemv(Cuda::cublasHandle, CUBLAS_OP_N, out_dim.rows, out_dim.batches,
			&Cuda::alpha, d_delta, out_dim.rows, d_onevec, 1, &Cuda::beta, d_delta_bias, 1));
	//Util::printDeviceData(d_delta_bias, out_dim.rows, 1, 1, 1, "d_delta_bias:");
	_params[Bias]->print_grad("d_delta_bias:");

	//Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	//Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "d_delta:");
	_params[Weight]->print_data("d_weight:");
	_preActivation->print_grad("d_delta");

	const DATATYPE* d_weight = _params[Weight]->device_data();
	DATATYPE* d_delta_input = _input->mutable_device_grad();
	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, in_dim.rows, out_dim.batches, out_dim.rows,
			&Cuda::alpha, d_weight, out_dim.rows, d_delta, out_dim.rows, &Cuda::beta, d_delta_input, in_dim.rows));

	//Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.batches, 1, 1, "d_delta_input:");
	_input->print_grad("d_delta_input:");

	propBackpropagation();
}
#endif


void SoftmaxLayer::initialize() {
	this->type = Layer::Softmax;

	//this->cost_fn = CostFactory::create(Cost::LogLikelihood);
	//this->activation_fn = ActivationFactory::create(Activation::Softmax);
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

void SoftmaxLayer::_load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	OutputLayer::_load(ifs, layerMap);
	initialize();
	SoftmaxLayer::_shape(false);
}

























