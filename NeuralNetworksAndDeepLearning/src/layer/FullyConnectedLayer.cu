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



FullyConnectedLayer::FullyConnectedLayer(const string name, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, ActivationType activationType)
	: HiddenLayer(name) {
	initialize(n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);
}




#ifndef GPU_MODE

FullyConnectedLayer::FullyConnectedLayer(const string name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
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





void FullyConnectedLayer::feedforward(UINT idx, const rcube &input, const char *end) {
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

	propFeedforward(this->output, end);
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







void FullyConnectedLayer::_save(ofstream &ofs) {
	HiddenLayer::_save(ofs);

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
__global__ void FillValues(DATATYPE *vec, int size, DATATYPE value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;
	vec[idx] = value;
}




void FullyConnectedLayer::initialize(int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, ActivationType activationType) {
	Cuda::refresh();
	this->out_dim = io_dim(n_out, 1, 1, 1);
	this->type = LayerType::FullyConnected;

	this->p_dropout = p_dropout;

	this->weight_update_param = weight_update_param;
	this->bias_update_param = bias_update_param;
	this->weight_filler = weight_filler;
	this->bias_filler = bias_filler;

	this->activation_fn = ActivationFactory::create(activationType);





	this->scale = 1. / (1. - p_dropout);
}

void FullyConnectedLayer::_shape(bool recursive) {
	in_dim.rows = in_dim.rows*in_dim.cols*in_dim.channels;
	in_dim.cols = 1;
	in_dim.channels = 1;
	out_dim.batches = in_dim.batches;

	if(recursive) {
		HiddenLayer::_shape();
	}

	int u_in = in_dim.unitsize();
	int u_out = out_dim.unitsize();
	int b_in = in_dim.batchsize();
	int b_out = out_dim.batchsize();



	weight = new DATATYPE[u_out*u_in];
	bias = new DATATYPE[u_out];

	cout << this->name << ", fanin: " << u_out*u_in << endl;
	weight_filler.fill(weight, u_out*u_in, u_in, u_out);
	bias_filler.fill(bias, u_out, u_in, u_out);


	//Util::setPrint(true);
	Util::printData(weight, u_out, u_in, 1, 1, "weight:");
	Util::printData(bias, u_out, 1, 1, 1, "bias:");
	//Util::setPrint(false);


	checkCudaErrors(Util::ucudaMalloc(&this->d_weight, sizeof(DATATYPE)*u_out*u_in));
	checkCudaErrors(Util::ucudaMalloc(&this->d_bias, sizeof(DATATYPE)*u_out));

	checkCudaErrors(Util::ucudaMalloc(&this->d_z, sizeof(DATATYPE)*b_out));
	checkCudaErrors(Util::ucudaMalloc(&this->d_delta, sizeof(DATATYPE)*b_out));
	checkCudaErrors(Util::ucudaMalloc(&this->d_delta_input, sizeof(DATATYPE)*b_in));

	checkCudaErrors(Util::ucudaMalloc(&this->d_delta_weight, sizeof(DATATYPE)*u_out*u_in));
	checkCudaErrors(Util::ucudaMalloc(&this->d_delta_weight_prev, sizeof(DATATYPE)*u_out*u_in));
	//FillValues<<<RoundUp(u_out*u_in, BW), BW>>>(this->d_onevec, u_out*u_in, 0.0f);
	checkCudaErrors(cudaMemset(d_delta_weight_prev, 0, u_out*u_in*sizeof(DATATYPE)));

	checkCudaErrors(Util::ucudaMalloc(&this->d_delta_bias, sizeof(DATATYPE)*u_out));
	checkCudaErrors(Util::ucudaMalloc(&this->d_delta_bias_prev, sizeof(DATATYPE)*u_out));
	checkCudaErrors(cudaMemset(d_delta_bias_prev, 0, u_out*sizeof(DATATYPE)));

	checkCudaErrors(Util::ucudaMalloc(&this->d_onevec, sizeof(DATATYPE)*in_dim.batches));
	FillValues<<<RoundUp(in_dim.batches, BW), BW>>>(this->d_onevec, in_dim.batches, 1.0f);
	//checkCudaErrors(cudaMemset(d_onevec, 1, in_dim.batches));


	checkCudaErrors(cudaMemcpyAsync(this->d_weight, weight, sizeof(DATATYPE)*u_out*u_in, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(this->d_bias, bias, sizeof(DATATYPE)*u_out, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaDeviceSynchronize());





	mask = new DATATYPE[b_out];
	checkCudaErrors(Util::ucudaMalloc(&this->d_mask, sizeof(DATATYPE)*b_out));

}

void FullyConnectedLayer::_clearShape() {
	if(weight) delete [] weight;
	if(bias) delete [] bias;

	checkCudaErrors(cudaFree(d_weight));
	checkCudaErrors(cudaFree(d_bias));

	checkCudaErrors(cudaFree(d_z));
	checkCudaErrors(cudaFree(d_delta));
	checkCudaErrors(cudaFree(d_delta_input));
	checkCudaErrors(cudaFree(d_delta_weight));
	checkCudaErrors(cudaFree(d_delta_weight_prev));
	checkCudaErrors(cudaFree(d_delta_bias));
	checkCudaErrors(cudaFree(d_delta_bias_prev));




	if(mask) delete [] mask;
	checkCudaErrors(cudaFree(d_mask));






	weight = 0;
	bias = 0;

	d_weight = 0;
	d_bias = 0;

	d_z = 0;
	d_delta = 0;
	d_delta_input = 0;
	d_delta_weight = 0;
	d_delta_weight_prev = 0;
	d_delta_bias = 0;
	d_delta_bias_prev = 0;

	HiddenLayer::_clearShape();
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
	checkCudaErrors(cudaFree(d_delta_weight_prev));
	checkCudaErrors(cudaFree(d_delta_bias));
	checkCudaErrors(cudaFree(d_delta_bias_prev));

	checkCudaErrors(cudaFree(d_onevec));

	ActivationFactory::destory(activation_fn);
}






///////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels

/**
 * Fills a floating-point array with ones.
 *
 * @param vec The array to fill.
 * @param size The number of elements in the array.
 */
__global__ void Dropout(const int n, const DATATYPE* in, const DATATYPE* mask,
		const unsigned int threashold, const float scale, DATATYPE *out)
{

	CUDA_KERNEL_LOOP(index, n) {
		//out[index] = in[index] * (mask[index] > threshold) * scale;
		out[index] = in[index] * (mask[index]) * scale;
	}
}






void FullyConnectedLayer::feedforward(UINT idx, const DATATYPE *input, const char *end) {
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
	Util::printDeviceData(d_bias, out_dim.rows, 1, 1, 1, "d_b:");
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
	Util::printDeviceData(d_z, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/d_z:"));
	Util::printDeviceData(d_output, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/d_output:"));
	//Util::setPrint(false);

	//exit(1);








	/*
	// TODO skip when test
	if(Util::train && p_dropout < 1.0f) {
		int b_out = out_dim.batchsize();
		for(int i = 0; i < b_out; i++) {
			mask[i] = ((rand()/(RAND_MAX+1.0) > p_dropout)?1:0);
		}
		checkCudaErrors(cudaMemcpyAsync(d_mask, mask, sizeof(DATATYPE)*b_out, cudaMemcpyHostToDevice));
		//FillOnes<<<RoundUp(in_dim.batches, BW), BW>>>(this->d_onevec, in_dim.batches);
		Dropout<<<RoundUp(b_out, BW), BW>>>(b_out, d_output, d_mask, 0, scale, d_output);

		//Util::setPrint(true);
		Util::printData(mask, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/mask:"));
		Util::printDeviceData(d_output, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/d_output:"));
		//Util::setPrint(false);
	}
	*/




	propFeedforward(this->d_output, end);
}

void FullyConnectedLayer::backpropagation(UINT idx, DATATYPE *next_delta_input) {
	if(!isLastNextLayerRequest(idx)) throw Exception();

	Util::printMessage("FullyConnectedLayer::backpropagation()---"+string(name));
	Cuda::refresh();





	/*
	if(Util::train && p_dropout < 1.0f) {
		//Util::setPrint(true);
		Util::printDeviceData(next_delta_input, out_dim.rows, out_dim.batches, 1, 1, "delta_input:");
		Dropout<<<RoundUp(out_dim.batchsize(), BW), BW>>>(out_dim.batchsize(), next_delta_input, d_mask, 0, scale, next_delta_input);

		Util::printData(mask, out_dim.rows, out_dim.batches, 1, 1, this->name+string("/mask:"));
		//DATATYPE *next_delta_input = next_layer->getDeltaInput();
		Util::printDeviceData(next_delta_input, out_dim.rows, out_dim.batches, 1, 1, "delta_input:");
		Util::setPrint(false);
	}
	*/






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



/*
void FullyConnectedLayer::reset_nabla(UINT idx) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();
	propResetNParam();
}
*/


void FullyConnectedLayer::update(UINT idx, UINT n, UINT miniBatchSize) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();
	Util::printMessage("FullyConnectedLayer::update()---"+string(name));

	/*
	//weight = (1-eta*lambda/n)*weight - (eta/miniBatchSize)*nabla_w;
	//bias -= eta/miniBatchSize*nabla_b;
	//weight = (1-weight_update_param.lr_mult*weight_update_param.decay_mult/n)*weight - (weight_update_param.lr_mult/miniBatchSize)*nabla_w;
	//bias -= bias_update_param.lr_mult/miniBatchSize*nabla_b;

	float delta_scale = -weight_update_param.lr_mult/miniBatchSize;
	float param_scale = 1-weight_update_param.lr_mult*weight_update_param.decay_mult/n;
	float b_delta_scale = -bias_update_param.lr_mult/miniBatchSize;

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
			&b_delta_scale, d_delta_bias, 1, d_bias, 1));
	Util::printDeviceData(d_bias, out_dim.rows, 1, 1, 1, "d_bias:");
	*/

	int weight_size = in_dim.rows*out_dim.rows;
	DATATYPE norm_scale = 1.0/in_dim.batches;
	DATATYPE reg_scale = weight_update_param.decay_mult;
	DATATYPE momentum = 0.0;
	DATATYPE learning_scale = weight_update_param.lr_mult;
	DATATYPE negative_one = -1.0;

	//Util::setPrint(true);
	Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight:");
	Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	Util::printDeviceData(d_delta_weight_prev, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight_prev:");

	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(weight_size), &norm_scale, d_delta_weight, 1));								// normalize by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &reg_scale, d_weight, 1, d_delta_weight, 1));					// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(weight_size), &momentum, d_delta_weight_prev, 1));								//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &learning_scale, d_delta_weight, 1, d_delta_weight_prev, 1));	// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &negative_one, d_delta_weight_prev, 1, d_weight, 1));			// update

	Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight:");
	Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
	Util::printDeviceData(d_delta_weight_prev, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight_prev:");
	//Util::setPrint(false);


	int bias_size = out_dim.rows;
	DATATYPE reg_scale_b = bias_update_param.decay_mult;
	DATATYPE learning_scale_b = bias_update_param.lr_mult;
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(bias_size), &norm_scale, d_delta_bias, 1));								// normalize by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &reg_scale_b, d_bias, 1, d_delta_bias, 1));					// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(bias_size), &momentum, d_delta_bias_prev, 1));								//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &learning_scale_b, d_delta_bias, 1, d_delta_bias_prev, 1));	// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &negative_one, d_delta_bias_prev, 1, d_bias, 1));				// update


	propUpdate(n, miniBatchSize);
}



















void FullyConnectedLayer::load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	HiddenLayer::load(ifs, layerMap);

	UINT n_out = 0;
	double p_dropout;
	ActivationType activationType;
	update_param weight_update_param, bias_update_param;
	param_filler weight_filler, bias_filler;

	ifs.read((char *)&n_out, sizeof(UINT));
	ifs.read((char *)&p_dropout, sizeof(double));
	ifs.read((char *)&activationType, sizeof(int));
	ifs.read((char *)&weight_update_param, sizeof(update_param));
	ifs.read((char *)&bias_update_param, sizeof(update_param));
	ifs.read((char *)&weight_filler, sizeof(param_filler));
	ifs.read((char *)&bias_filler, sizeof(param_filler));

	initialize(n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);
	FullyConnectedLayer::_shape(false);

	// initialize() 내부에서 weight, bias를 초기화하므로 initialize() 후에 weight, bias load를 수행해야 함
	ifs.read((char *)weight, sizeof(DATATYPE)*out_dim.unitsize()*in_dim.unitsize());
	ifs.read((char *)bias, sizeof(DATATYPE)*out_dim.unitsize());
	checkCudaErrors(cudaMemcpyAsync(d_weight, weight, sizeof(DATATYPE)*out_dim.unitsize()*in_dim.unitsize(), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_bias, bias, sizeof(DATATYPE)*out_dim.unitsize(), cudaMemcpyHostToDevice));
}







void FullyConnectedLayer::_save(ofstream &ofs) {
	HiddenLayer::_save(ofs);

	int activationType = (int)activation_fn->getType();

	ofs.write((char *)&out_dim.rows, sizeof(UINT));
	ofs.write((char *)&p_dropout, sizeof(double));
	ofs.write((char *)&activationType, sizeof(int));
	ofs.write((char *)&weight_update_param, sizeof(update_param));
	ofs.write((char *)&bias_update_param, sizeof(update_param));
	ofs.write((char *)&weight_filler, sizeof(param_filler));
	ofs.write((char *)&bias_filler, sizeof(param_filler));

	checkCudaErrors(cudaMemcpyAsync(weight, d_weight, sizeof(DATATYPE)*out_dim.unitsize()*in_dim.unitsize(), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpyAsync(bias, d_bias, sizeof(DATATYPE)*out_dim.unitsize(), cudaMemcpyDeviceToHost));
	ofs.write((char *)weight, sizeof(DATATYPE)*out_dim.unitsize()*in_dim.unitsize());
	ofs.write((char *)bias, sizeof(DATATYPE)*out_dim.unitsize());
}


DATATYPE FullyConnectedLayer::_sumSquareParam() {
	DATATYPE weight_result;
	DATATYPE bias_result;

	int weight_size = out_dim.unitsize()*in_dim.unitsize();
	checkCudaErrors(cublasSdot(Cuda::cublasHandle, weight_size, d_delta_weight, 1, d_delta_weight, 1, &weight_result));

	int bias_size = out_dim.unitsize();
	checkCudaErrors(cublasSdot(Cuda::cublasHandle, bias_size, d_delta_bias, 1, d_delta_bias, 1, &bias_result));

	//cout << weight_result + bias_result << " ";

	return weight_result + bias_result;
}

DATATYPE FullyConnectedLayer::_sumSquareParam2() {
	DATATYPE weight_result;
	DATATYPE bias_result;

	int weight_size = out_dim.unitsize()*in_dim.unitsize();
	checkCudaErrors(cublasSdot(Cuda::cublasHandle, weight_size, d_weight, 1, d_weight, 1, &weight_result));

	int bias_size = out_dim.unitsize();
	checkCudaErrors(cublasSdot(Cuda::cublasHandle, bias_size, d_bias, 1, d_bias, 1, &bias_result));

	return weight_result + bias_result;
}


void FullyConnectedLayer::_scaleParam(DATATYPE scale_factor) {
	int weight_size = out_dim.unitsize()*in_dim.unitsize();
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(weight_size), &scale_factor, d_delta_weight, 1));

	int bias_size = out_dim.unitsize();
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(bias_size), &scale_factor, d_delta_bias, 1));
}



#endif















