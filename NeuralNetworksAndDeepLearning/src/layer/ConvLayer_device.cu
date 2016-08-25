/*
 * ConvLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */


#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "../Util.h"
#include "../exception/Exception.h"
#include "../network/NetworkConfig.h"


ConvLayer::~ConvLayer() {
	delete _params[ParamType::Filter];
	delete _params[ParamType::Bias];
	_params.clear();

	delete _paramsHistory[ParamType::Filter];
	delete _paramsHistory[ParamType::Bias];
	_paramsHistory.clear();

	delete _preActivation;

	//if(filters) delete [] filters;
	//if(biases) delete [] biases;

	//checkCudaErrors(cudaFree(d_z));
	//checkCudaErrors(cudaFree(d_delta));
	//checkCudaErrors(cudaFree(d_delta_input));
	//checkCudaErrors(cudaFree(d_delta_weight));
	//checkCudaErrors(cudaFree(d_delta_weight_prev));
	//checkCudaErrors(cudaFree(d_delta_bias));
	//checkCudaErrors(cudaFree(d_delta_bias_prev));
	if(d_workspace) checkCudaErrors(cudaFree(d_workspace));

	checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));

	ActivationFactory::destory(activation_fn);
}


void ConvLayer::initialize(filter_dim filter_d, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, Activation::Type activationType) {

	this->type = Layer::Conv;
	this->filter_d = filter_d;

	this->weight_update_param = weight_update_param;
	this->bias_update_param = bias_update_param;
	this->weight_filler = weight_filler;
	this->bias_filler = bias_filler;

	const int filter_size = filter_d.size();
	//this->filters = new DATATYPE[filter_size];
	//this->biases = new DATATYPE[filter_d.filters];

	//checkCudaErrors(Util::ucudaMalloc(&this->d_filters, sizeof(DATATYPE)*filter_size));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_biases, sizeof(DATATYPE)*filter_d.filters));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_weight, sizeof(DATATYPE)*filter_size));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_weight_prev, sizeof(DATATYPE)*filter_size));
	//checkCudaErrors(cudaMemset(d_delta_weight_prev, 0, filter_size*sizeof(DATATYPE)));

	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_bias, sizeof(DATATYPE)*filter_d.filters));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_bias_prev, sizeof(DATATYPE)*filter_d.filters));
	//checkCudaErrors(cudaMemset(d_delta_bias_prev, 0, filter_d.filters*sizeof(DATATYPE)));


	this->_params.resize(2);
	this->_params[Filter] = new Data();
	this->_params[Bias] = new Data();
	this->_params[Filter]->reshape({filter_d.filters, filter_d.channels, filter_d.rows, filter_d.cols});
	this->_params[Bias]->reshape({filter_d.filters, 1, 1, 1});

	this->_paramsHistory.resize(2);
	this->_paramsHistory[Filter] = new Data();
	this->_paramsHistory[Bias] = new Data();
	this->_paramsHistory[Filter]->reshape({filter_d.filters, filter_d.channels, filter_d.rows, filter_d.cols});
	this->_paramsHistory[Bias]->reshape({filter_d.filters, 1, 1, 1});

	this->_preActivation = new Data();


	checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

	checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,
			CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			1, filter_d.filters, 1, 1));

	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
			CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			filter_d.filters, filter_d.channels, filter_d.rows, filter_d.cols));

	int pad = (filter_d.rows-1)/2;
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
			pad, pad, filter_d.stride, filter_d.stride, 1, 1,
			CUDNN_CROSS_CORRELATION));

	this->activation_fn = ActivationFactory::create(activationType);
	//checkCudaErrors(cudaDeviceSynchronize());
}

void ConvLayer::_shape(bool recursive) {
	cudnnTensorDescriptor_t tempInputTensorDesc;
	checkCUDNN(cudnnCreateTensorDescriptor(&tempInputTensorDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(tempInputTensorDesc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				in_dim.batches, in_dim.channels, in_dim.rows, in_dim.cols));

	int n = 0, c = 0, h = 0, w = 0;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
			tempInputTensorDesc, filterDesc,
			&n, &c, &h, &w));

	out_dim.batches = n;
	out_dim.channels = c;
	out_dim.rows = h;
	out_dim.cols = w;

	checkCUDNN(cudnnDestroyTensorDescriptor(tempInputTensorDesc));

	if(recursive) {
		HiddenLayer::_shape();
	}

	int u_in = in_dim.unitsize();
	int u_out = out_dim.unitsize();
	int b_in = in_dim.batchsize();
	int b_out = out_dim.batchsize();

	//weight_filler.fill(this->filters, filter_d.size(), filter_d.unitsize(), filter_d.filters);
	//bias_filler.fill(this->biases, filter_d.filters, filter_d.unitsize(), filter_d.filters);
	weight_filler.fill(_params[Filter]->mutable_host_data(), filter_d.size(), filter_d.unitsize(), filter_d.filters);
	bias_filler.fill(_params[Bias]->mutable_host_data(), filter_d.filters, filter_d.unitsize(), filter_d.filters);

	//Util::printData(this->filters, filter_d.rows, filter_d.cols, filter_d.channels, filter_d.filters, this->name+string("/filters:"));
	//Util::printData(this->biases, filter_d.filters, 1, 1, 1, this->name+string("/biases:"));

	_params[Filter]->print_data(this->name+string("/filters:"));
	_params[Bias]->print_data(this->name+string("/biases:"));

	//checkCudaErrors(cudaMemcpyAsync(this->d_filters, filters, sizeof(DATATYPE)*filter_d.size(), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpyAsync(this->d_biases, biases, sizeof(DATATYPE)*filter_d.filters, cudaMemcpyHostToDevice));

	//checkCudaErrors(Util::ucudaMalloc(&this->d_z, sizeof(DATATYPE)*b_out));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta, sizeof(DATATYPE)*b_out));
	_preActivation->reshape({out_dim.batches, out_dim.channels, out_dim.rows, out_dim.cols});

	size_t convFwdWorkspaceSize;
	size_t convBwdFilterWorkspaceSize;
	size_t convBwdDataWorkspaceSize;
	// forward algorithm
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(Cuda::cudnnHandle,
			inputTensorDesc, filterDesc, convDesc, outputTensorDesc,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convFwdAlgo));

	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(Cuda::cudnnHandle,
			inputTensorDesc, filterDesc, convDesc, outputTensorDesc,
			convFwdAlgo, &convFwdWorkspaceSize));

	// backward filter algorithm
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(Cuda::cudnnHandle,
			inputTensorDesc, outputTensorDesc, convDesc, filterDesc,
			CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 32<<20, &convBwdFilterAlgo));

	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(Cuda::cudnnHandle,
			inputTensorDesc, outputTensorDesc, convDesc, filterDesc,
			convBwdFilterAlgo, &convBwdFilterWorkspaceSize));

	// backward data algorithm
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(Cuda::cudnnHandle,
			filterDesc, outputTensorDesc, convDesc, inputTensorDesc,
			CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 32<<20, &convBwdDataAlgo));

	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(Cuda::cudnnHandle,
			filterDesc, outputTensorDesc, convDesc, inputTensorDesc,
			convBwdDataAlgo, &convBwdDataWorkspaceSize));

	workspaceSize = 0;
	workspaceSize = std::max(workspaceSize, convFwdWorkspaceSize);
	workspaceSize = std::max(workspaceSize, convBwdFilterWorkspaceSize);
	workspaceSize = std::max(workspaceSize, convBwdDataWorkspaceSize);
	//cout << workspaceSize << ", " << convFwdWorkspaceSize << ", " << convBwdFilterWorkspaceSize << ", " << convBwdDataWorkspaceSize << endl;

	d_workspace = 0;
	if(workspaceSize > 0) {
		//cout << "workspaceSize: " << workspaceSize << endl;
		checkCudaErrors(Util::ucudaMalloc(&d_workspace, workspaceSize));
	}
}

void ConvLayer::_clearShape() {
	//checkCudaErrors(cudaFree(d_z));
	//checkCudaErrors(cudaFree(d_delta));
	//checkCudaErrors(cudaFree(d_delta_input));

	//d_z = 0;
	//d_delta = 0;
	//d_delta_input = 0;

	delete _params[0];
	delete _params[1];
	//_params.clear();

	delete _paramsHistory[0];
	delete _paramsHistory[1];
	//_paramsHistory.clear();

	delete _preActivation;

	if(d_workspace) {
		checkCudaErrors(cudaFree(d_workspace));
		d_workspace = 0;
	}

	HiddenLayer::_clearShape();
}


void ConvLayer::_save(ofstream &ofs) {
	HiddenLayer::_save(ofs);

	int activationType = (int)activation_fn->getType();

	ofs.write((char *)&filter_d, sizeof(filter_dim));
	ofs.write((char *)&activationType, sizeof(int));
	ofs.write((char *)&weight_update_param, sizeof(update_param));
	ofs.write((char *)&bias_update_param, sizeof(update_param));
	ofs.write((char *)&weight_filler, sizeof(param_filler));
	ofs.write((char *)&bias_filler, sizeof(param_filler));


	const DATATYPE* filters = _params[Filter]->host_data();
	const DATATYPE* biases = _params[Bias]->host_data();
	//checkCudaErrors(cudaMemcpyAsync(filters, d_filters, sizeof(DATATYPE)*filter_d.size(), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpyAsync(biases, d_biases, sizeof(DATATYPE)*filter_d.filters, cudaMemcpyDeviceToHost));
	ofs.write((char *)filters, sizeof(DATATYPE)*filter_d.size());
	ofs.write((char *)biases, sizeof(DATATYPE)*filter_d.filters);
}

void ConvLayer::_load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	HiddenLayer::_load(ifs, layerMap);

	filter_dim filter_d;
	Activation::Type activationType;
	update_param weight_update_param, bias_update_param;
	param_filler weight_filler, bias_filler;

	ifs.read((char *)&filter_d, sizeof(filter_dim));
	ifs.read((char *)&activationType, sizeof(int));
	ifs.read((char *)&weight_update_param, sizeof(update_param));
	ifs.read((char *)&bias_update_param, sizeof(update_param));
	ifs.read((char *)&weight_filler, sizeof(param_filler));
	ifs.read((char *)&bias_filler, sizeof(param_filler));

	initialize(filter_d, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);
	ConvLayer::_shape(false);

	DATATYPE* filters = _params[Filter]->mutable_host_data();
	DATATYPE* biases = _params[Bias]->mutable_host_data();
	// initialize() 내부에서 weight, bias를 초기화하므로 initialize() 후에 weight, bias load를 수행해야 함
	ifs.read((char *)filters, sizeof(DATATYPE)*filter_d.size());
	ifs.read((char *)biases, sizeof(DATATYPE)*filter_d.filters);
	//checkCudaErrors(cudaMemcpyAsync(d_filters, filters, sizeof(DATATYPE)*filter_d.size(), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpyAsync(d_biases, biases, sizeof(DATATYPE)*filter_d.filters, cudaMemcpyHostToDevice));
}

void ConvLayer::update() {
	//Util::setPrint(true);

	//for(UINT i = 0; i < filter_d.filters; i++) {
	//	filters[i] = (1-eta*lambda/n)*filters[i] - (eta/miniBatchSize)*nabla_w[i];
	//}
	//biases -= eta/miniBatchSize*nabla_b;

	/*
	for(UINT i = 0; i < filter_d.filters; i++) {
		filters[i] = (1-weight_update_param.lr_mult*weight_update_param.decay_mult/n)*filters[i] - (weight_update_param.lr_mult/miniBatchSize)*nabla_w[i];
	}
	biases -= bias_update_param.lr_mult/miniBatchSize*nabla_b;
	*/

	/*
	float delta_scale = -weight_update_param.lr_mult/miniBatchSize;
	float param_scale = 1-weight_update_param.lr_mult*weight_update_param.decay_mult/n;
	float b_delta_scale = -bias_update_param.lr_mult/miniBatchSize;

	Util::printDeviceData(d_delta_weight, filter_d.rows, filter_d.cols, filter_d.channels, filter_d.filters, "d_delta_weight:");
	Util::printDeviceData(d_filters, filter_d.rows, filter_d.cols, filter_d.channels, filter_d.filters, "d_filters:");
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(filter_d.size()), &param_scale, d_filters, 1));
	Util::printDeviceData(d_filters, filter_d.rows, filter_d.cols, filter_d.channels, filter_d.filters, "d_filters:");
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(filter_d.size()), &delta_scale, d_delta_weight, 1, d_filters, 1));
	Util::printDeviceData(d_filters, filter_d.rows, filter_d.cols, filter_d.channels, filter_d.filters, "d_filters:");

	Util::printDeviceData(d_delta_bias, 1, 1, filter_d.filters, 1, "d_delta_bias:");
	Util::printDeviceData(d_biases, 1, 1, filter_d.filters, 1, "d_biases:");
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(filter_d.filters),	&b_delta_scale, d_delta_bias, 1, d_biases, 1));
	Util::printDeviceData(d_biases, 1, 1, filter_d.filters, 1, "d_biases:");
	*/

	int weight_size = filter_d.size();
	DATATYPE norm_scale = 1.0/in_dim.batches;
	DATATYPE reg_scale = networkConfig->_weightDecay * weight_update_param.decay_mult;
	DATATYPE momentum = networkConfig->_momentum;
	DATATYPE learning_scale = networkConfig->_baseLearningRate * weight_update_param.lr_mult;
	DATATYPE negative_one = -1.0;

	DATATYPE* d_delta_weight = _params[Filter]->mutable_device_grad();
	DATATYPE* d_filters = _params[Filter]->mutable_device_data();
	DATATYPE* d_delta_weight_prev = _paramsHistory[Filter]->mutable_device_grad();

	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(weight_size), &norm_scale, d_delta_weight, 1));								// normalize by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &reg_scale, d_filters, 1, d_delta_weight, 1));					// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(weight_size), &momentum, d_delta_weight_prev, 1));								//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &learning_scale, d_delta_weight, 1, d_delta_weight_prev, 1));	// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &negative_one, d_delta_weight_prev, 1, d_filters, 1));			// update


	int bias_size = filter_d.filters;
	DATATYPE reg_scale_b = networkConfig->_weightDecay * bias_update_param.decay_mult;
	DATATYPE learning_scale_b = networkConfig->_baseLearningRate * bias_update_param.lr_mult;

	DATATYPE* d_delta_bias = _params[Bias]->mutable_device_grad();
	DATATYPE* d_biases = _params[Bias]->mutable_device_data();
	DATATYPE* d_delta_bias_prev = _paramsHistory[Bias]->mutable_device_grad();

	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(bias_size), &norm_scale, d_delta_bias, 1));								// normalize by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &reg_scale_b, d_biases, 1, d_delta_bias, 1));					// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(bias_size), &momentum, d_delta_bias_prev, 1));								//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &learning_scale_b, d_delta_bias, 1, d_delta_bias_prev, 1));	// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &negative_one, d_delta_bias_prev, 1, d_biases, 1));			// update

}

void ConvLayer::_feedforward() {
	//Util::setPrint(true);
	//this->d_input = input;
	//float alpha = 1.0f, beta = 0.0f;

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	//Util::printDeviceData(d_filters, filter_d.rows, filter_d.cols, filter_d.channels, filter_d.filters, "d_filters:");
	_input->print_data("d_input:");
	_params[Filter]->print_data("d_weight:");

	const DATATYPE* d_filters = _params[Filter]->device_data();
	const DATATYPE* d_input = _input->device_data();
	DATATYPE* d_z = _preActivation->mutable_device_data();

	checkCUDNN(cudnnConvolutionForward(Cuda::cudnnHandle,
			&Cuda::alpha, inputTensorDesc, d_input, filterDesc, d_filters, convDesc,
			convFwdAlgo, d_workspace, workspaceSize, &Cuda::beta, outputTensorDesc, d_z));
	//Util::printDeviceData(d_z, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_z:");
	_preActivation->print_data("d_z:");

	//Util::printDeviceData(d_biases, 1, 1, filter_d.filters, 1, "d_biases:");
	_params[Bias]->print_data("d_b:");
	const DATATYPE* d_biases = _params[Bias]->device_data();
	checkCUDNN(cudnnAddTensor(Cuda::cudnnHandle,
			(void *)&Cuda::alpha, biasTensorDesc,	d_biases, (void *)&Cuda::alpha, outputTensorDesc, d_z));
	Util::printDeviceData(d_z, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_z:");

	DATATYPE* d_output = _output->mutable_device_data();
	activation_fn->activate(d_z, d_output, outputTensorDesc);

	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, 1, 1, this->name+string("/d_output:"));
	_output->print_data(this->name+string("d_output:"));

}

void ConvLayer::_backpropagation() {
	// 여러 source로부터 delta값이 모두 모이면 dw, dx 계산
	Cuda::refresh();

	//Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_output:");
	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_output:");
	_output->print_grad("d_delta_output:");
	_output->print_data("output:");

	const DATATYPE* d_output = _output->device_data();
	const DATATYPE* d_delta_output = _output->device_grad();
	const DATATYPE* d_z = _preActivation->device_data();
	DATATYPE* d_delta = _preActivation->mutable_device_grad();

	activation_fn->d_activate(d_output, d_delta_output, d_z, d_delta, outputTensorDesc);
	//Util::printDeviceData(d_delta, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta:");
	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	_preActivation->print_grad("d_delta:");
	_input->print_data("d_input:");

	const DATATYPE* d_input = _input->device_data();
	DATATYPE* d_delta_weight = _params[Filter]->mutable_device_grad();

	checkCUDNN(cudnnConvolutionBackwardFilter(Cuda::cudnnHandle,
			(void *)&Cuda::alpha, inputTensorDesc, d_input, outputTensorDesc, d_delta, convDesc, convBwdFilterAlgo,
			d_workspace, workspaceSize,
			(void *)&Cuda::beta, filterDesc, d_delta_weight));
	//Util::printDeviceData(d_delta_weight, filter_d.rows, filter_d.cols, filter_d.channels, filter_d.filters, "d_delta_weight:");
	_params[Filter]->print_grad("d_delta_weight:");

	DATATYPE* d_delta_bias = _params[Bias]->mutable_device_grad();
	checkCUDNN(cudnnConvolutionBackwardBias(Cuda::cudnnHandle,
			&Cuda::alpha, outputTensorDesc, d_delta, &Cuda::beta, biasTensorDesc, d_delta_bias));
	//Util::printDeviceData(d_delta_bias, 1, 1, filter_d.filters, 1, "d_delta_bias:");
	//Util::printDeviceData(d_filters, filter_d.rows, filter_d.cols, filter_d.channels, filter_d.filters, "d_filters:");
	_params[Bias]->print_grad("d_delta_bias:");
	_params[Filter]->print_data("d_weight:");

	const DATATYPE* d_filters = _params[Filter]->device_data();
	DATATYPE* d_delta_input = _input->mutable_device_grad();
	checkCUDNN(cudnnConvolutionBackwardData(Cuda::cudnnHandle,
			(void *)&Cuda::alpha, filterDesc, d_filters, outputTensorDesc, d_delta, convDesc, convBwdDataAlgo,
			d_workspace, workspaceSize,
			(void *)&Cuda::beta, inputTensorDesc, d_delta_input));

	//Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_delta_input:");
	//Util::printDeviceData(d_filters, filter_d.rows, filter_d.cols, filter_d.channels, filter_d.filters, "d_filters:");
	_input->print_grad("d_delta_input:");
	_params[Filter]->print_data("d_filters:");
}
