/*
 * ConvLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */


#ifdef GPU_MODE
#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "../Util.h"
#include "../exception/Exception.h"
#include "../network/NetworkConfig.h"

template <typename Dtype>
ConvLayer<Dtype>::~ConvLayer() {
	delete _params[ParamType::Filter];
	delete _params[ParamType::Bias];
	_params.clear();

	delete _paramsHistory[ParamType::Filter];
	delete _paramsHistory[ParamType::Bias];
	_paramsHistory.clear();

	delete _preActivation;

	if(d_workspace) checkCudaErrors(cudaFree(d_workspace));

	checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));

	ActivationFactory<Dtype>::destory(activation_fn);
}

template <typename Dtype>
void ConvLayer<Dtype>::initialize(filter_dim filter_d, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, typename Activation<Dtype>::Type activationType) {

	this->type = Layer<Dtype>::Conv;
	this->filter_d = filter_d;

	this->weight_update_param = weight_update_param;
	this->bias_update_param = bias_update_param;
	this->weight_filler = weight_filler;
	this->bias_filler = bias_filler;

	const int filter_size = filter_d.size();

	this->_params.resize(2);
	this->_params[Filter] = new Data<Dtype>();
	this->_params[Bias] = new Data<Dtype>();
	this->_params[Filter]->reshape({filter_d.filters, filter_d.channels, filter_d.rows, filter_d.cols});
	this->_params[Bias]->reshape({filter_d.filters, 1, 1, 1});

	this->_paramsHistory.resize(2);
	this->_paramsHistory[Filter] = new Data<Dtype>();
	this->_paramsHistory[Bias] = new Data<Dtype>();
	this->_paramsHistory[Filter]->reshape({filter_d.filters, filter_d.channels, filter_d.rows, filter_d.cols});
	this->_paramsHistory[Bias]->reshape({filter_d.filters, 1, 1, 1});

	this->_preActivation = new Data<Dtype>();


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

	this->activation_fn = ActivationFactory<Dtype>::create(activationType);
}

template <typename Dtype>
void ConvLayer<Dtype>::_shape(bool recursive) {
	cudnnTensorDescriptor_t tempInputTensorDesc;
	checkCUDNN(cudnnCreateTensorDescriptor(&tempInputTensorDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(tempInputTensorDesc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				this->in_dim.batches, this->in_dim.channels, this->in_dim.rows, this->in_dim.cols));

	int n = 0, c = 0, h = 0, w = 0;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc,
			tempInputTensorDesc, filterDesc,
			&n, &c, &h, &w));

	this->out_dim.batches = n;
	this->out_dim.channels = c;
	this->out_dim.rows = h;
	this->out_dim.cols = w;

	checkCUDNN(cudnnDestroyTensorDescriptor(tempInputTensorDesc));

	if(recursive) {
		HiddenLayer<Dtype>::_shape();
	}

	int u_in = this->in_dim.unitsize();
	int u_out = this->out_dim.unitsize();
	int b_in = this->in_dim.batchsize();
	int b_out = this->out_dim.batchsize();

	weight_filler.fill(_params[Filter]->mutable_host_data(), filter_d.size(), filter_d.unitsize(), filter_d.filters);
	bias_filler.fill(_params[Bias]->mutable_host_data(), filter_d.filters, filter_d.unitsize(), filter_d.filters);

	/*
	//if(this->name == "inception_3a/conv5x5") {
	if(this->name == "convLayer2") {
		cout << "shape()";
		Data<Dtype>::printConfig = 1;
		_params[Filter]->print_data(this->name+string("/filters:"));
		_params[Bias]->print_data(this->name+string("/biases:"));
		Data<Dtype>::printConfig = 0;
	}
	*/

	_preActivation->reshape({this->out_dim.batches, this->out_dim.channels, this->out_dim.rows, this->out_dim.cols});

	size_t convFwdWorkspaceSize;
	size_t convBwdFilterWorkspaceSize;
	size_t convBwdDataWorkspaceSize;
	// forward algorithm
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(Cuda::cudnnHandle,
			this->inputTensorDesc, filterDesc, convDesc, this->outputTensorDesc,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convFwdAlgo));

	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(Cuda::cudnnHandle,
			this->inputTensorDesc, filterDesc, convDesc, this->outputTensorDesc,
			convFwdAlgo, &convFwdWorkspaceSize));

	// backward filter algorithm
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(Cuda::cudnnHandle,
			this->inputTensorDesc, this->outputTensorDesc, convDesc, filterDesc,
			CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 32<<20, &convBwdFilterAlgo));

	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(Cuda::cudnnHandle,
			this->inputTensorDesc, this->outputTensorDesc, convDesc, filterDesc,
			convBwdFilterAlgo, &convBwdFilterWorkspaceSize));

	// backward data algorithm
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(Cuda::cudnnHandle,
			filterDesc, this->outputTensorDesc, convDesc, this->inputTensorDesc,
			CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 32<<20, &convBwdDataAlgo));

	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(Cuda::cudnnHandle,
			filterDesc, this->outputTensorDesc, convDesc, this->inputTensorDesc,
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

template <typename Dtype>
void ConvLayer<Dtype>::_clearShape() {
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

	HiddenLayer<Dtype>::_clearShape();
}

template <typename Dtype>
void ConvLayer<Dtype>::_save(ofstream &ofs) {
	HiddenLayer<Dtype>::_save(ofs);

	int activationType = (int)activation_fn->getType();

	ofs.write((char *)&filter_d, sizeof(filter_dim));
	ofs.write((char *)&activationType, sizeof(int));
	ofs.write((char *)&weight_update_param, sizeof(update_param));
	ofs.write((char *)&bias_update_param, sizeof(update_param));
	ofs.write((char *)&weight_filler, sizeof(param_filler));
	ofs.write((char *)&bias_filler, sizeof(param_filler));


	const Dtype* filters = _params[Filter]->host_data();
	const Dtype* biases = _params[Bias]->host_data();
	//checkCudaErrors(cudaMemcpyAsync(filters, d_filters, sizeof(Dtype)*filter_d.size(), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpyAsync(biases, d_biases, sizeof(Dtype)*filter_d.filters, cudaMemcpyDeviceToHost));
	ofs.write((char *)filters, sizeof(Dtype)*filter_d.size());
	ofs.write((char *)biases, sizeof(Dtype)*filter_d.filters);
}

template <typename Dtype>
void ConvLayer<Dtype>::_load(ifstream &ifs, map<Layer<Dtype>*, Layer<Dtype>*> &layerMap) {
	HiddenLayer<Dtype>::_load(ifs, layerMap);

	filter_dim filter_d;
	typename Activation<Dtype>::Type activationType;
	update_param weight_update_param, bias_update_param;
	param_filler weight_filler, bias_filler;

	ifs.read((char *)&filter_d, sizeof(filter_dim));
	ifs.read((char *)&activationType, sizeof(int));
	ifs.read((char *)&weight_update_param, sizeof(update_param));
	ifs.read((char *)&bias_update_param, sizeof(update_param));
	ifs.read((char *)&weight_filler, sizeof(param_filler));
	ifs.read((char *)&bias_filler, sizeof(param_filler));

	initialize(filter_d, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);
	ConvLayer<Dtype>::_shape(false);

	Dtype* filters = _params[Filter]->mutable_host_data();
	Dtype* biases = _params[Bias]->mutable_host_data();
	// initialize() 내부에서 weight, bias를 초기화하므로 initialize() 후에 weight, bias load를 수행해야 함
	ifs.read((char *)filters, sizeof(Dtype)*filter_d.size());
	ifs.read((char *)biases, sizeof(Dtype)*filter_d.filters);
	//checkCudaErrors(cudaMemcpyAsync(d_filters, filters, sizeof(Dtype)*filter_d.size(), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpyAsync(d_biases, biases, sizeof(Dtype)*filter_d.filters, cudaMemcpyHostToDevice));
}

template <typename Dtype>
void ConvLayer<Dtype>::update() {
	//Util::setPrint(true);

	//for(uint32_t32_t i = 0; i < filter_d.filters; i++) {
	//	filters[i] = (1-eta*lambda/n)*filters[i] - (eta/miniBatchSize)*nabla_w[i];
	//}
	//biases -= eta/miniBatchSize*nabla_b;

	/*
	for(uint32_t i = 0; i < filter_d.filters; i++) {
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
	Dtype norm_scale = 1.0/this->in_dim.batches;
	Dtype reg_scale = this->networkConfig->_weightDecay * weight_update_param.decay_mult;
	Dtype momentum = this->networkConfig->_momentum;
	Dtype learning_scale = this->networkConfig->_baseLearningRate * weight_update_param.lr_mult;
	Dtype negative_one = -1.0;

	Dtype* d_delta_weight = _params[Filter]->mutable_device_grad();
	Dtype* d_filters = _params[Filter]->mutable_device_data();
	Dtype* d_delta_weight_prev = _paramsHistory[Filter]->mutable_device_grad();


	/*
	//if(this->name == "inception_3a/conv5x5") {
	if(this->name == "convLayer2") {
		cout << "update()" << endl;
		Data<Dtype>::printConfig = 1;
		_params[Filter]->print_data(this->name+string("/filters:"));
		_params[Bias]->print_data(this->name+string("/biases:"));
		Data<Dtype>::printConfig = 0;
	}
	*/



	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(weight_size), &norm_scale, d_delta_weight, 1));								// normalize by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &reg_scale, d_filters, 1, d_delta_weight, 1));					// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(weight_size), &momentum, d_delta_weight_prev, 1));								//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &learning_scale, d_delta_weight, 1, d_delta_weight_prev, 1));	// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(weight_size), &negative_one, d_delta_weight_prev, 1, d_filters, 1));			// update


	int bias_size = filter_d.filters;
	Dtype reg_scale_b = this->networkConfig->_weightDecay * bias_update_param.decay_mult;
	Dtype learning_scale_b = this->networkConfig->_baseLearningRate * bias_update_param.lr_mult;

	Dtype* d_delta_bias = _params[Bias]->mutable_device_grad();
	Dtype* d_biases = _params[Bias]->mutable_device_data();
	Dtype* d_delta_bias_prev = _paramsHistory[Bias]->mutable_device_grad();

	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(bias_size), &norm_scale, d_delta_bias, 1));								// normalize by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &reg_scale_b, d_biases, 1, d_delta_bias, 1));					// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(bias_size), &momentum, d_delta_bias_prev, 1));								//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &learning_scale_b, d_delta_bias, 1, d_delta_bias_prev, 1));	// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(bias_size), &negative_one, d_delta_bias_prev, 1, d_biases, 1));			// update

}

template <typename Dtype>
void ConvLayer<Dtype>::_feedforward() {

	/*
	//if(this->name == "inception_3a/conv5x5") {
	if(this->name == "convLayer2") {
		cout << "feedforward()" << endl;
		Data<Dtype>::printConfig = 1;
		_params[Filter]->print_data(this->name+string("/filters:"));
		_params[Bias]->print_data(this->name+string("/biases:"));
		Data<Dtype>::printConfig = 0;
	}
	*/

	// Apply filters to input data
	const Dtype* d_inputData = this->_input->device_data();
	const Dtype* d_filtersData = _params[Filter]->device_data();
	Dtype* d_preActivationData = _preActivation->mutable_device_data();

	checkCUDNN(cudnnConvolutionForward(Cuda::cudnnHandle,
			&Cuda::alpha, this->inputTensorDesc, d_inputData, filterDesc, d_filtersData, convDesc, convFwdAlgo, d_workspace, workspaceSize,
			&Cuda::beta, this->outputTensorDesc, d_preActivationData));
	_preActivation->print_data("d_preActivationData:");


	// Add bias to filtered input data
	_params[Bias]->print_data("d_biasData:");
	const Dtype* d_biasesData = _params[Bias]->device_data();
	checkCUDNN(cudnnAddTensor(Cuda::cudnnHandle,
			&Cuda::alpha, biasTensorDesc, d_biasesData,
			&Cuda::alpha, this->outputTensorDesc, d_preActivationData));
	_preActivation->print_data("d_preActivationData:");


	// Activate filtered result
	Dtype* d_output = this->_output->mutable_device_data();
	activation_fn->forward(this->outputTensorDesc, d_preActivationData, d_output);
	this->_output->print_data(this->name+string("d_output:"));
}


template <typename Dtype>
void ConvLayer<Dtype>::_activationBackward() {
	this->_output->print_grad("d_delta_output:");
	this->_output->print_data("output:");

	const Dtype* d_output = this->_output->device_data();
	const Dtype* d_delta_output = this->_output->device_grad();
	const Dtype* d_z = _preActivation->device_data();
	Dtype* d_delta = _preActivation->mutable_device_grad();

	activation_fn->backward(this->outputTensorDesc, d_output, d_delta_output, d_z, d_delta);
	this->_input->print_data("d_inputData:");
	_preActivation->print_grad("d_preActivationGrad:");
}


template <typename Dtype>
void ConvLayer<Dtype>::_backpropagation() {
	// 여러 source로부터 delta값이 모두 모이면 dw, dx 계산
	//Cuda::refresh();

	/*
	//Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_output:");
	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_output:");
	this->_output->print_grad("d_delta_output:");
	this->_output->print_data("output:");

	const Dtype* d_output = this->_output->device_data();
	const Dtype* d_delta_output = this->_output->device_grad();
	const Dtype* d_z = _preActivation->device_data();
	Dtype* d_delta = _preActivation->mutable_device_grad();

	//activation_fn->backward(d_output, d_delta_output, d_z, d_delta, outputTensorDesc);
	activation_fn->backward(this->outputTensorDesc, d_output, d_delta_output, d_z, d_delta);
	//Util::printDeviceData(d_delta, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta:");
	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	_preActivation->print_grad("d_delta:");
	this->_input->print_data("d_input:");
	*/

	_computeFiltersGrad();
	_computeBiasesGrad();
	_computeInputGrad();

}


template <typename Dtype>
void ConvLayer<Dtype>::_computeFiltersGrad() {
	// d(Cost)/d(Filters)
	const Dtype* d_inputData = this->_input->device_data();
	const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	Dtype* d_filtersGrad = _params[Filter]->mutable_device_grad();

	/*
	if(this->name == "convLayer1") {
		Data<Dtype>::printConfig = 1;
		_preActivation->print_grad("preActivationGrad:");
		Data<Dtype>::printConfig = 0;
	}
	*/


	checkCUDNN(cudnnConvolutionBackwardFilter(Cuda::cudnnHandle,
			&Cuda::alpha, this->inputTensorDesc, d_inputData, this->outputTensorDesc, d_preActivationGrad, convDesc, convBwdFilterAlgo, d_workspace, workspaceSize,
			&Cuda::beta, filterDesc, d_filtersGrad));

	/*
	if(this->name == "convLayer1") {
		Data<Dtype>::printConfig = 1;
		_params[Filter]->print_grad("d_filtersGrad:");
		Data<Dtype>::printConfig = 0;
	}
	*/

	/*
	if(_params[Filter]->is_nan_grad()) {
		Data<Dtype>::printConfig = 1;
		this->_input->print_data("input:");
		_preActivation->print_grad("delta:");
		_params[Filter]->print_grad("deltaWeight:");
		Data<Dtype>::printConfig = 0;
		exit(1);
	}
	*/
	/*
	if(this->_output->is_nan_grad()) {
		cout << this->name << " output gradient nan ... " << endl;
		exit(1);
	}
	*/
}

template <typename Dtype>
void ConvLayer<Dtype>::_computeBiasesGrad() {
	// d(Cost)/d(Biases)
	const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	Dtype* d_biasGrad = _params[Bias]->mutable_device_grad();

	/*
	if(this->name == "convLayer1") {
		Data<Dtype>::printConfig = 1;
		_preActivation->print_grad("d_preActivationGrad:");
		Data<Dtype>::printConfig = 0;
	}
	*/

	checkCUDNN(cudnnConvolutionBackwardBias(Cuda::cudnnHandle,
			&Cuda::alpha, this->outputTensorDesc, d_preActivationGrad,
			&Cuda::beta, biasTensorDesc, d_biasGrad));

	/*
	if(this->name == "convLayer1") {
		Data<Dtype>::printConfig = 1;
		_params[Bias]->print_grad("d_biasesGrad:");
		Data<Dtype>::printConfig = 0;
	}
	*/
}

template <typename Dtype>
void ConvLayer<Dtype>::_computeInputGrad() {
	// d(Cost)/d(Input)
	const Dtype* d_filtersData = _params[Filter]->device_data();
	const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	Dtype* d_inputGrad = this->_input->mutable_device_grad();
	checkCUDNN(cudnnConvolutionBackwardData(Cuda::cudnnHandle,
			&Cuda::alpha, filterDesc, d_filtersData, this->outputTensorDesc, d_preActivationGrad, convDesc, convBwdDataAlgo, d_workspace, workspaceSize,
			&Cuda::beta, this->inputTensorDesc, d_inputGrad));
	this->_input->print_grad("d_inputGrad:");
	_params[Filter]->print_data("d_filtersData:");

	/*
	//if(this->name == "inception_3a/conv5x5reduce") {
	if(this->name == "inception_3a/conv1x1") {
		double grad = _params[Filter]->sumsq_device_grad();
		double data = _params[Filter]->sumsq_device_data();
		//cout << "inception_3a/conv5x5reduce grad: " << grad << ", data:" << data << endl;
		cout << "inception_3a/conv1x1 grad: " << grad << ", data:" << data << endl;
	}
	*/
}





template ConvLayer<float>::~ConvLayer();
template void ConvLayer<float>::initialize(filter_dim filter_d, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, typename Activation<float>::Type activationType);
template void ConvLayer<float>::_shape(bool recursive);
template void ConvLayer<float>::_clearShape();
template void ConvLayer<float>::_save(ofstream &ofs);
template void ConvLayer<float>::_load(ifstream &ifs, map<Layer<float>*, Layer<float>*> &layerMap);
template void ConvLayer<float>::update();
template void ConvLayer<float>::_feedforward();
template void ConvLayer<float>::_activationBackward();
template void ConvLayer<float>::_backpropagation();



#endif
