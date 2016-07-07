/*
 * FullyConnectedLayer.cpp
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#include "FullyConnectedLayer.h"
#include "../Util.h"
#include "../exception/Exception.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cublas_v2.h>
#include <cudnn.h>



FullyConnectedLayer::FullyConnectedLayer(const char *name, io_dim in_dim, io_dim out_dim, double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, ActivationType activationType)
	: HiddenLayer(name, in_dim, out_dim) {
	initialize(p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);
}

void FullyConnectedLayer::save(UINT idx, ofstream &ofs) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();
	save(ofs);
	propSave(ofs);
}



#if CPU_MODE

FullyConnectedLayer::FullyConnectedLayer(const char *name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, ActivationType activationType)
	: HiddenLayer(name, n_in, n_out) {
	initialize(p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);
}

FullyConnectedLayer::~FullyConnectedLayer() {
	ActivationFactory::destory(activation_fn);
}

void FullyConnectedLayer::initialize(double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, ActivationType activationType) {
	this->type = LayerType::FullyConnected;
	this->id = Layer::generateLayerId();

	this->p_dropout = p_dropout;

	this->weight_update_param = weight_update_param;
	this->bias_update_param = bias_update_param;
	this->weight_filler = weight_filler;
	this->bias_filler = bias_filler;

	int n_in = in_dim.size();
	int n_out = out_dim.size();

	this->bias.set_size(n_out, 1);
	//this->bias.zeros();
	this->bias_filler.fill(this->bias, n_in);

	this->weight.set_size(n_out, n_in);
	this->weight_filler.fill(this->weight, n_in);


	//this->weight.randn();
	//this->weight *= 1/sqrt(n_in);				// initial point scaling

	this->nabla_b.set_size(n_out, 1);
	this->nabla_w.set_size(n_out, n_in);
	this->nabla_b.zeros();
	this->nabla_w.zeros();

	this->z.set_size(n_out, 1, 1);
	this->output.set_size(n_out, 1, 1);
	this->delta.set_size(n_out, 1, 1);
	this->delta.zeros();
	this->delta_input.set_size(n_in, 1, 1);
	this->delta_input.zeros();

	/**
	 * HiddenLayer에서 activation_fn이 할당되는 곳에서 weight initialize 필요
	 * 잊어버리기 쉬울 것 같으니 대책이 필요
	 */
	this->activation_fn = ActivationFactory::create(activationType);
}





void FullyConnectedLayer::feedforward(UINT idx, const rcube &input) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	Util::printCube(input, "input:");
	Util::convertCube(input, this->input);
	Util::printCube(this->input, "converted input:");

	//Util::dropoutLayer(this->input, this->p_dropout);
	//Util::printCube(this->input, "dropped out:");

	Util::printMat(weight, "weight:");
	Util::printVec(bias, "bias:");

	z.slice(0) = weight*this->input.slice(0) + bias;
	Util::printCube(z, "z:");

	activation_fn->activate(z, output);
	Util::printCube(output, "output:");

	propFeedforward(this->output);
}

void FullyConnectedLayer::backpropagation(UINT idx, HiddenLayer *next_layer) {
	if(!isLastNextLayerRequest(idx)) throw Exception();


	rcube w_next_delta(size(output));
	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);

	Util::printMat(w_next_delta.slice(0), "w_next_delta");
	//Util::printMat(next_w->t(), "next_w");
	//Util::printMat(next_delta.slice(0), "next_delta");

	rcube sp;
	activation_fn->d_activate(output, sp);

	// delta l = dC/dz
	delta.slice(0) = w_next_delta.slice(0) % sp.slice(0);
	Util::printMat(delta.slice(0), "delta:");

	nabla_b += delta.slice(0);
	// delta lw = dC/dw
	nabla_w += delta.slice(0)*input.slice(0).t();



	// delta lx = dC/dx
	delta_input.slice(0) = weight.t()*delta.slice(0);
	//fc_layer->getWeight().t()*fc_layer->getDelta().slice(0)

	propBackpropagation();
	delta.zeros();
	delta_input.zeros();

}




void FullyConnectedLayer::reset_nabla(UINT idx) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	nabla_b.zeros();
	nabla_w.zeros();

	propResetNParam();
}


void FullyConnectedLayer::update(UINT idx, UINT n, UINT miniBatchSize) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	//weight = (1-eta*lambda/n)*weight - (eta/miniBatchSize)*nabla_w;
	//bias -= eta/miniBatchSize*nabla_b;

	weight = (1-weight_update_param.lr_mult*weight_update_param.decay_mult/n)*weight - (weight_update_param.lr_mult/miniBatchSize)*nabla_w;
	bias -= bias_update_param.lr_mult/miniBatchSize*nabla_b;

	propUpdate(n, miniBatchSize);
}




void FullyConnectedLayer::load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	HiddenLayer::load(ifs, layerMap);

	double p_dropout;
	ifs.read((char *)&p_dropout, sizeof(double));

	ActivationType activationType;
	ifs.read((char *)&activationType, sizeof(int));

	update_param weight_update_param;
	ifs.read((char *)&weight_update_param, sizeof(update_param));

	update_param bias_update_param;
	ifs.read((char *)&bias_update_param, sizeof(update_param));

	param_filler weight_filler;
	ifs.read((char *)&weight_filler, sizeof(param_filler));

	param_filler bias_filler;
	ifs.read((char *)&bias_filler, sizeof(param_filler));

	initialize(p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);

	// initialize() 내부에서 weight, bias를 초기화하므로 initialize() 후에 weight, bias load를 수행해야 함


	weight.load(ifs, file_type::arma_binary);
	//weight.print("load-weight:");
	bias.load(ifs, file_type::arma_binary);
	//bias.print("load-bias:");

}







void FullyConnectedLayer::save(ofstream &ofs) {
	HiddenLayer::save(ofs);

	ofs.write((char *)&p_dropout, sizeof(double));

	int activationType = (int)activation_fn->getType();
	ofs.write((char *)&activationType, sizeof(int));
	ofs.write((char *)&weight_update_param, sizeof(update_param));
	ofs.write((char *)&bias_update_param, sizeof(update_param));
	ofs.write((char *)&weight_filler, sizeof(param_filler));
	ofs.write((char *)&bias_filler, sizeof(param_filler));
	//ofs.write((char *)&weight, sizeof(rmat));
	//ofs.write((char *)&bias, sizeof(rvec));

	//weight.print("save-weight:");
	weight.save(ofs, file_type::arma_binary);
	//bias.print("save-bias:");
	bias.save(ofs, file_type::arma_binary);
}

#else




///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
__global__ void FillOnes(DATATYPE *vec, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;
	vec[idx] = 1.0f;
}




void FullyConnectedLayer::initialize(double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, ActivationType activationType) {
	Cuda::refresh();

	this->type = LayerType::FullyConnected;
	this->id = Layer::generateLayerId();

	this->p_dropout = p_dropout;

	this->weight_update_param = weight_update_param;
	this->bias_update_param = bias_update_param;
	this->weight_filler = weight_filler;
	this->bias_filler = bias_filler;

	int u_in = in_dim.unitsize();
	int u_out = out_dim.unitsize();
	int b_in = in_dim.batchsize();
	int b_out = out_dim.batchsize();

	weight = new DATATYPE[u_out*u_in];
	bias = new DATATYPE[u_out];

	weight_filler.fill(weight, u_out*u_in, u_out*u_in);
	bias_filler.fill(bias, u_out, 0);
	Util::printData(weight, u_out, u_in, 1, 1, "weight:");
	Util::printData(bias, u_out, 1, 1, 1, "bias:");

	checkCudaErrors(Util::ucudaMalloc(&this->d_weight, sizeof(DATATYPE)*u_out*u_in));
	checkCudaErrors(Util::ucudaMalloc(&this->d_bias, sizeof(DATATYPE)*u_out));

	checkCudaErrors(Util::ucudaMalloc(&this->d_z, sizeof(DATATYPE)*b_out));
	checkCudaErrors(Util::ucudaMalloc(&this->d_delta, sizeof(DATATYPE)*b_out));
	checkCudaErrors(Util::ucudaMalloc(&this->d_delta_input, sizeof(DATATYPE)*b_in));
	checkCudaErrors(Util::ucudaMalloc(&this->d_delta_weight, sizeof(DATATYPE)*u_out*u_in));
	checkCudaErrors(Util::ucudaMalloc(&this->d_delta_bias, sizeof(DATATYPE)*u_out));

	//DATATYPE *onevec = new DATATYPE[in_dim.batches];
	//for(int i = 0; i < in_dim.batches; i++) onevec[i] = 1;
	checkCudaErrors(Util::ucudaMalloc(&this->d_onevec, sizeof(DATATYPE)*in_dim.batches));
	FillOnes<<<RoundUp(in_dim.batches, BW), BW>>>(this->d_onevec, in_dim.batches);
	//Util::printDeviceData(d_onevec, 1, 1, 1, in_dim.batches, "d_onevec:");
	//checkCudaErrors(cudaMemcpyAsync(this->d_onevec, onevec, sizeof(DATATYPE)*in_dim.batches, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpyAsync(this->d_weight, weight, sizeof(DATATYPE)*u_out*u_in, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(this->d_bias, bias, sizeof(DATATYPE)*u_out, cudaMemcpyHostToDevice));

	this->activation_fn = ActivationFactory::create(activationType);

	checkCudaErrors(cudaDeviceSynchronize());
}

FullyConnectedLayer::~FullyConnectedLayer() {
	Cuda::refresh();

	if(weight) delete [] weight;
	if(bias) delete [] bias;

	checkCudaErrors(cudaFree(d_weight));
	checkCudaErrors(cudaFree(d_bias));

	checkCudaErrors(cudaFree(d_z));
	checkCudaErrors(cudaFree(d_delta));
	checkCudaErrors(cudaFree(d_delta_input));
	checkCudaErrors(cudaFree(d_delta_weight));
	checkCudaErrors(cudaFree(d_delta_bias));

	ActivationFactory::destory(activation_fn);
}







void FullyConnectedLayer::feedforward(UINT idx, const DATATYPE *input) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	Util::printMessage("FullyConnectedLayer::feedforward()---"+string(name));
	Cuda::refresh();

	this->d_input = input;
	float alpha = 1.0f, beta = 0.0f;

	Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	Util::printDeviceData(d_input, in_dim.rows, in_dim.batches, 1, 1, "d_input:");

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			out_dim.rows, out_dim.batches, in_dim.rows,
			&alpha,
			d_weight, out_dim.rows,
			d_input, in_dim.rows,
			&beta,
			d_z, out_dim.rows));

	Util::printDeviceData(d_z, out_dim.rows, out_dim.batches, 1, 1, "d_z:");
	//Util::printDeviceData(d_bias, out_dim.rows, 1, out_dim.channels, out_dim.batches, "d_b:");
	//Util::printDeviceData(d_onevec, 1, 1, 1, in_dim.batches, "d_onevec:");

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			out_dim.rows, out_dim.batches, 1,
	    &alpha,
	    d_bias, out_dim.rows,
	    d_onevec, 1,
	    &alpha,
	    d_z, out_dim.rows));

	Util::printDeviceData(d_z, out_dim.rows, out_dim.batches, 1, 1, "d_z:");

	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_output:");
	activation_fn->activate(d_z, d_output, outputTensorDesc);

	//Util::setPrint(true);
	Util::printDeviceData(d_output, out_dim.rows, out_dim.batches, 1, 1, "d_output:");
	//Util::setPrint(false);

	propFeedforward(this->d_output);
}

void FullyConnectedLayer::backpropagation(UINT idx, HiddenLayer *next_layer) {
	if(!isLastNextLayerRequest(idx)) throw Exception();

	Util::printMessage("FullyConnectedLayer::backpropagation()---"+string(name));
	Cuda::refresh();

	DATATYPE *next_delta_input = next_layer->getDeltaInput();
	Util::printDeviceData(next_delta_input, out_dim.rows, out_dim.batches, 1, 1, "delta_input:");
	Util::printDeviceData(d_output, out_dim.rows, out_dim.batches, 1, 1, "output:");
	activation_fn->d_activate(d_output, next_delta_input, d_z, d_delta, outputTensorDesc);
	Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "d_delta:");

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

	/*
	 * rcube w_next_delta(size(output));
	rcube sp;
	activation_fn->d_activate(output, sp);

	// delta l = dC/dz
	delta.slice(0) = w_next_delta.slice(0) % sp.slice(0);

	nabla_b += delta.slice(0);
	// delta lw = dC/dw
	nabla_w += delta.slice(0)*input.slice(0).t();

	// delta lx = dC/dx
	delta_input.slice(0) = weight.t()*delta.slice(0);
	//fc_layer->getWeight().t()*fc_layer->getDelta().slice(0)
	 */

	propBackpropagation();
}




void FullyConnectedLayer::reset_nabla(UINT idx) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();
	propResetNParam();
}


void FullyConnectedLayer::update(UINT idx, UINT n, UINT miniBatchSize) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	Util::printMessage("FullyConnectedLayer::update()---"+string(name));
	Cuda::refresh();

	//weight = (1-eta*lambda/n)*weight - (eta/miniBatchSize)*nabla_w;
	//bias -= eta/miniBatchSize*nabla_b;
	//weight = (1-weight_update_param.lr_mult*weight_update_param.decay_mult/n)*weight - (weight_update_param.lr_mult/miniBatchSize)*nabla_w;
	//bias -= bias_update_param.lr_mult/miniBatchSize*nabla_b;

	float delta_scale = -weight_update_param.lr_mult/miniBatchSize;
	float param_scale = 1-weight_update_param.lr_mult*weight_update_param.decay_mult/n;

	Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight:");
	Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(in_dim.rows*out_dim.rows), &param_scale, d_weight, 1));
	Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(in_dim.rows*out_dim.rows),
			&delta_scale, d_delta_weight, 1, d_weight, 1));
	Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");

	Util::printDeviceData(d_delta_bias, out_dim.rows, 1, 1, 1, "d_delta_bias:");
	Util::printDeviceData(d_bias, out_dim.rows, 1, 1, 1, "d_bias:");
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(out_dim.rows),
			&delta_scale, d_delta_bias, 1, d_bias, 1));
	Util::printDeviceData(d_bias, out_dim.rows, 1, 1, 1, "d_bias:");

	propUpdate(n, miniBatchSize);
}




void FullyConnectedLayer::load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	HiddenLayer::load(ifs, layerMap);

	double p_dropout;
	ActivationType activationType;
	update_param weight_update_param, bias_update_param;
	param_filler weight_filler, bias_filler;

	ifs.read((char *)&p_dropout, sizeof(double));
	ifs.read((char *)&activationType, sizeof(int));
	ifs.read((char *)&weight_update_param, sizeof(update_param));
	ifs.read((char *)&bias_update_param, sizeof(update_param));
	ifs.read((char *)&weight_filler, sizeof(param_filler));
	ifs.read((char *)&bias_filler, sizeof(param_filler));

	initialize(p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);

	// initialize() 내부에서 weight, bias를 초기화하므로 initialize() 후에 weight, bias load를 수행해야 함
	ifs.read((char *)weight, sizeof(DATATYPE)*out_dim.unitsize()*in_dim.unitsize());
	ifs.read((char *)bias, sizeof(DATATYPE)*out_dim.unitsize());
	checkCudaErrors(cudaMemcpyAsync(d_weight, weight, sizeof(DATATYPE)*out_dim.unitsize()*in_dim.unitsize(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_bias, bias, sizeof(DATATYPE)*out_dim.unitsize(), cudaMemcpyHostToDevice));
}







void FullyConnectedLayer::save(ofstream &ofs) {
	HiddenLayer::save(ofs);

	ofs.write((char *)&p_dropout, sizeof(double));

	int activationType = (int)activation_fn->getType();
	ofs.write((char *)&activationType, sizeof(int));
	ofs.write((char *)&weight_update_param, sizeof(update_param));
	ofs.write((char *)&bias_update_param, sizeof(update_param));
	ofs.write((char *)&weight_filler, sizeof(param_filler));
	ofs.write((char *)&bias_filler, sizeof(param_filler));
	//ofs.write((char *)&weight, sizeof(rmat));
	//ofs.write((char *)&bias, sizeof(rvec));

	checkCudaErrors(cudaMemcpyAsync(weight, d_weight, sizeof(DATATYPE)*out_dim.unitsize()*in_dim.unitsize(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyAsync(bias, d_bias, sizeof(DATATYPE)*out_dim.unitsize(), cudaMemcpyDeviceToHost));
	ofs.write((char *)weight, sizeof(DATATYPE)*out_dim.unitsize()*in_dim.unitsize());
	ofs.write((char *)bias, sizeof(DATATYPE)*out_dim.unitsize());

	//weight.print("save-weight:");
	//weight.save(ofs, file_type::arma_binary);
	//bias.print("save-bias:");
	//bias.save(ofs, file_type::arma_binary);
}







#endif















