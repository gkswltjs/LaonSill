/*
 * ConvLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */


#ifdef GPU_MODE
#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "Util.h"
#include "Exception.h"
#include "NetworkConfig.h"
#include "cuda_runtime.h"
#include <algorithm>

using namespace std;

/**
 * dst array에 src array를 더한다.
 *
 * @param dst dst array, dst + src가 저장이 될 장소
 * @param src src array
 * @param N The number of elements in the array.
 */
template <typename Dtype>
__global__ void AddArrayOfConvLayer(Dtype* dst, const Dtype* src, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= N)
		return;

	dst[idx] = dst[idx] + src[idx];
}

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

	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));

	ActivationFactory<Dtype>::destory(activation_fn);
}

template <typename Dtype>
void ConvLayer<Dtype>::initialize(filter_dim filter_d, update_param weight_update_param, update_param bias_update_param,
		param_filler<Dtype> weight_filler, param_filler<Dtype> bias_filler, typename Activation<Dtype>::Type activationType) {

	this->type = Layer<Dtype>::Conv;
	this->filter_d = filter_d;

	this->weight_update_param = weight_update_param;
	this->bias_update_param = bias_update_param;
	this->weight_filler = weight_filler;
	this->bias_filler = bias_filler;

	const int filter_size = filter_d.size();

	this->_params.resize(2);
	this->_params[Filter] = new Data<Dtype>("Filter");
	this->_params[Bias] = new Data<Dtype>("Bias");
	this->_params[Filter]->shape({filter_d.filters, filter_d.channels, filter_d.rows, filter_d.cols});
	this->_params[Bias]->shape({filter_d.filters, 1, 1, 1});

	this->_paramsHistory.resize(2);
	this->_paramsHistory[Filter] = new Data<Dtype>("FilterHistory");
	this->_paramsHistory[Bias] = new Data<Dtype>("BiasHistory");
	this->_paramsHistory[Filter]->shape({filter_d.filters, filter_d.channels, filter_d.rows, filter_d.cols});
	this->_paramsHistory[Bias]->shape({filter_d.filters, 1, 1, 1});

	this->_preActivation = new Data<Dtype>("PreActivation");

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

	checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc,
			CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			1, filter_d.filters, 1, 1));

	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
			CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
			filter_d.filters, filter_d.channels, filter_d.rows, filter_d.cols));

	//int pad = (filter_d.rows-1)/2;
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
			filter_d.pad, filter_d.pad, filter_d.stride, filter_d.stride, 1, 1,
			CUDNN_CROSS_CORRELATION));

	this->activation_fn = ActivationFactory<Dtype>::create(activationType);
}

template <typename Dtype>
void ConvLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batches, channels, rows, cols));

	int n = 0, c = 0, h = 0, w = 0;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
			convDesc,
			this->inputTensorDesc,
			filterDesc,
			&n, &c, &h, &w));

	checkCUDNN(cudnnSetTensor4dDescriptor(
			this->outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			n, c, h, w));

	const uint32_t obatches = static_cast<uint32_t>(n);
	const uint32_t ochannels = static_cast<uint32_t>(c);
	const uint32_t orows = static_cast<uint32_t>(h);
	const uint32_t ocols = static_cast<uint32_t>(w);

	printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
			this->name.c_str(), obatches, ochannels, orows, ocols);

	this->_inputShape[0] = inputShape;
	this->_preActivation->shape({obatches, ochannels, orows, ocols});
	this->_outputData[0]->shape({obatches, ochannels, orows, ocols});


	//int u_in = this->in_dim.unitsize();
	//int u_out = this->out_dim.unitsize();
	//int b_in = this->in_dim.batchsize();
	//int b_out = this->out_dim.batchsize();
	int u_in = channels * rows * cols;
	int u_out = c * h * w;
	int b_in = batches * channels * rows * cols;
	int b_out = n * c * h * w;

	weight_filler.fill(_params[Filter]);
	bias_filler.fill(_params[Bias]);


	size_t convFwdWorkspaceSize;
	size_t convBwdFilterWorkspaceSize;
	size_t convBwdDataWorkspaceSize;

	// forward algorithm
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			filterDesc,
			convDesc,
			this->outputTensorDesc,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			8<<20,
			&convFwdAlgo));

	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			filterDesc,
			convDesc,
			this->outputTensorDesc,
			convFwdAlgo,
			&convFwdWorkspaceSize));

	// backward filter algorithm
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			convDesc,
			filterDesc,
			CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
			8<<20,
			&convBwdFilterAlgo));

	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
			Cuda::cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			convDesc,
			filterDesc,
			convBwdFilterAlgo,
			&convBwdFilterWorkspaceSize));

	// backward data algorithm
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
			Cuda::cudnnHandle,
			filterDesc,
			this->outputTensorDesc,
			convDesc,
			this->inputTensorDesc,
			CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
			8<<20,
			&convBwdDataAlgo));

	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
			Cuda::cudnnHandle,
			filterDesc,
			this->outputTensorDesc,
			convDesc,
			this->inputTensorDesc,
			convBwdDataAlgo,
			&convBwdDataWorkspaceSize));

	workspaceSize = 0;
	workspaceSize = max(workspaceSize, convFwdWorkspaceSize);
	workspaceSize = max(workspaceSize, convBwdFilterWorkspaceSize);
	workspaceSize = max(workspaceSize, convBwdDataWorkspaceSize);

	d_workspace = 0;
	if(workspaceSize > 0) {
		checkCudaErrors(Util::ucudaMalloc(&d_workspace, workspaceSize));
	}

	/*
	this->setInDimension(this->_inputData[0]->getShape());

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

	weight_filler.fill(_params[Filter]);
	bias_filler.fill(_params[Bias]);


	_params[Filter]->print_data(this->name + " filter: ");



	_preActivation->shape({this->out_dim.batches, this->out_dim.channels, this->out_dim.rows, this->out_dim.cols});

	size_t convFwdWorkspaceSize;
	size_t convBwdFilterWorkspaceSize;
	size_t convBwdDataWorkspaceSize;
	// forward algorithm
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(Cuda::cudnnHandle,
			this->inputTensorDesc, filterDesc, convDesc, this->outputTensorDesc,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 8<<20, &convFwdAlgo));
			//CUDNN_CONVOLUTION_FWD_NO_WORKSPACE, 0, &convFwdAlgo));
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(Cuda::cudnnHandle,
			this->inputTensorDesc, filterDesc, convDesc, this->outputTensorDesc,
			convFwdAlgo, &convFwdWorkspaceSize));


	// backward filter algorithm
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(Cuda::cudnnHandle,
			this->inputTensorDesc, this->outputTensorDesc, convDesc, filterDesc,
			CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 8<<20, &convBwdFilterAlgo));
			//CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, 32<<20, &convBwdFilterAlgo));
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(Cuda::cudnnHandle,
			this->inputTensorDesc, this->outputTensorDesc, convDesc, filterDesc,
			convBwdFilterAlgo, &convBwdFilterWorkspaceSize));


	// backward data algorithm
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(Cuda::cudnnHandle,
			filterDesc, this->outputTensorDesc, convDesc, this->inputTensorDesc,
			CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 8<<20, &convBwdDataAlgo));
			//CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, 32<<20, &convBwdDataAlgo));
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(Cuda::cudnnHandle,
			filterDesc, this->outputTensorDesc, convDesc, this->inputTensorDesc,
			convBwdDataAlgo, &convBwdDataWorkspaceSize));

	workspaceSize = 0;
	workspaceSize = max(workspaceSize, convFwdWorkspaceSize);
	workspaceSize = max(workspaceSize, convBwdFilterWorkspaceSize);
	workspaceSize = max(workspaceSize, convBwdDataWorkspaceSize);
	//cout << workspaceSize << ", " << convFwdWorkspaceSize << ", " << convBwdFilterWorkspaceSize << ", " << convBwdDataWorkspaceSize << endl;

	d_workspace = 0;
	if(workspaceSize > 0) {
		//cout << "workspaceSize: " << workspaceSize << endl;
		checkCudaErrors(Util::ucudaMalloc(&d_workspace, workspaceSize));
	}
	*/
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
void ConvLayer<Dtype>::update() {
	// update filters ...
	const uint32_t weightSize = filter_d.size();
	const Dtype regScale = this->networkConfig->_weightDecay * weight_update_param.decay_mult;
	const Dtype learnScale = this->networkConfig->getLearningRate() * weight_update_param.lr_mult;
	_updateParam(weightSize, regScale, learnScale, _paramsHistory[Filter], _params[Filter]);

	// update biases ...
	const uint32_t biasSize = filter_d.filters;
	const Dtype regScale_b = this->networkConfig->_weightDecay * bias_update_param.decay_mult;
	const Dtype learnScale_b = this->networkConfig->getLearningRate() * bias_update_param.lr_mult;
	_updateParam(biasSize, regScale_b, learnScale_b, _paramsHistory[Bias], _params[Bias]);
}




template <typename Dtype>
void ConvLayer<Dtype>::_updateParam(const uint32_t paramSize, const Dtype regScale, const Dtype learnScale, Data<Dtype>* dataHistory, Data<Dtype>* data) {
	const uint32_t batches = this->_inputData[0]->getShape(0);
	const Dtype normScale = 1.0/batches;
	const Dtype momentum = this->networkConfig->_momentum;
	const Dtype negativeOne = -1.0;

    data->mutable_host_grad();
	Dtype* d_paramGrad = data->mutable_device_grad();   // should update grad
	Dtype* d_paramData = data->mutable_device_data();
	Dtype* d_paramHistoryData = dataHistory->mutable_device_data();

	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(paramSize), &normScale, d_paramGrad, 1));							// normalized by batch size
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &regScale, d_paramData, 1, d_paramGrad, 1));			// regularize
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(paramSize), &momentum, d_paramHistoryData, 1));					//
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &learnScale, d_paramGrad, 1, d_paramHistoryData, 1));	// momentum
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &negativeOne, d_paramHistoryData, 1, d_paramData, 1));	// update

}

template <typename Dtype>
void ConvLayer<Dtype>::applyChanges(LearnableLayer<Dtype> *targetLayer) {
	const uint32_t weightSize = filter_d.size();
	const uint32_t biasSize = filter_d.filters;
    ConvLayer<Dtype>* _targetLayer = (ConvLayer<Dtype>*)targetLayer;

    int blockSize = BW;
    int gridSize = (weightSize + blockSize -1) / blockSize;

    AddArrayOfConvLayer<<<gridSize, blockSize>>>(
        _targetLayer->_params[Filter]->mutable_device_grad(),
        _params[Filter]->device_grad(), weightSize);

    gridSize = (biasSize + blockSize -1) / blockSize;

    AddArrayOfConvLayer<<<gridSize, blockSize>>>(
        _targetLayer->_params[Bias]->mutable_device_grad(),
        _params[Bias]->device_grad(), biasSize);
}

template <typename Dtype>
void ConvLayer<Dtype>::syncParams(LearnableLayer<Dtype> *targetLayer) {
	const uint32_t weightSize = filter_d.size();
	const uint32_t biasSize = filter_d.filters;
    ConvLayer<Dtype>* _targetLayer = (ConvLayer<Dtype>*)targetLayer;

    memcpy(_params[Filter]->mutable_host_grad(), _targetLayer->_params[Filter]->host_grad(),
        weightSize);
    memcpy(_params[Bias]->mutable_host_grad(), _targetLayer->_params[Bias]->host_grad(),
        biasSize);
#if 0
    for (uint32_t paramIdx = 0; paramIdx < weightSize; paramIdx++) {
        _params[Filter]->mutable_host_grad()[paramIdx] = 
            _targetLayer->_params[Filter]->host_grad()[paramIdx];
    }

    for (uint32_t paramIdx = 0; paramIdx < biasSize; paramIdx++) {
        _params[Bias]->mutable_host_grad()[paramIdx] = 
            _targetLayer->_params[Bias]->host_grad()[paramIdx];
    }
#endif
}

template <typename Dtype>
void ConvLayer<Dtype>::syncMutableMem() {
	_params[Filter]->mutable_device_grad();
	_params[Filter]->host_grad();
	_params[Bias]->mutable_device_grad();
	_params[Bias]->host_data();
}













template <typename Dtype>
void ConvLayer<Dtype>::feedforward() {
	reshape();

	_computeFiltersConvolutionData();
	_computeActivationData();
}



template <typename Dtype>
void ConvLayer<Dtype>::_computeFiltersConvolutionData() {
	// Apply filters to input data
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	const Dtype* d_filtersData = _params[Filter]->device_data();
	Dtype* d_preActivationData = _preActivation->mutable_device_data();

	this->_inputData[0]->print_data();
	_params[Filter]->print_data();

	checkCUDNN(cudnnConvolutionForward(Cuda::cudnnHandle,
			&Cuda::alpha, this->inputTensorDesc, d_inputData, filterDesc, d_filtersData, convDesc, convFwdAlgo, d_workspace, workspaceSize,
			&Cuda::beta, this->outputTensorDesc, d_preActivationData));

	_preActivation->print_data();

	// Add bias to filtered input data
	_params[Bias]->print_data();

	const Dtype* d_biasesData = _params[Bias]->device_data();

	checkCUDNN(cudnnAddTensor(Cuda::cudnnHandle,
			&Cuda::alpha, biasTensorDesc, d_biasesData,
			&Cuda::alpha, this->outputTensorDesc, d_preActivationData));

	_preActivation->print_data();
}

template <typename Dtype>
void ConvLayer<Dtype>::_computeActivationData() {
	// Activate filtered result
	const Dtype* d_preActivationData = _preActivation->device_data();
	Dtype* d_output = this->_outputData[0]->mutable_device_data();

	_preActivation->print_data();

	if (activation_fn)
		activation_fn->forward(this->outputTensorDesc, d_preActivationData, d_output);

	this->_outputData[0]->print_data();
}



template <typename Dtype>
void ConvLayer<Dtype>::backpropagation() {
	// 여러 source로부터 delta값이 모두 모이면 dw, dx 계산

	/*
	//Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "delta_output:");
	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "output:");
	this->_output->print_grad("delta_output:");
	this->_output->print_data("output:");

	const Dtype* d_output = this->_output->device_data();
	const Dtype* d_delta_output = this->_output->device_grad();
	const Dtype* d_z = _preActivation->device_data();
	Dtype* d_delta = _preActivation->mutable_device_grad();

	//activation_fn->backward(d_output, d_delta_output, d_z, d_delta, outputTensorDesc);
	activation_fn->backward(this->outputTensorDesc, d_output, d_delta_output, d_z, d_delta);
	//Util::printDeviceData(d_delta, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "delta:");
	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "input:");
	_preActivation->print_grad("delta:");
	this->_input->print_data("input:");
	*/




	//if(this->name == "inception_3a/convProjection") {
	//	Data<Dtype>::printConfig = 1;
	//}


	_computePreActivationGrad();
	_computeFiltersGrad();

	//if(this->name == "inception_3a/convProjection") {
	//	exit(1);
	//}


	_computeBiasesGrad();
	_computeInputGrad();


	/*
	if(_params[0]->is_nan_grad()) {
		cout << this->name << " filter is nan grad ... " << endl;
	}
	if(_params[1]->is_nan_grad()) {
		cout << this->name << " bias is nan grad ... " << endl;
	}
	*/
}


template <typename Dtype>
void ConvLayer<Dtype>::_computePreActivationGrad() {
	this->_outputData[0]->print_grad("outputGrad:");
	this->_outputData[0]->print_data("outputData:");

	const Dtype* d_outputData = this->_outputData[0]->device_data();
	const Dtype* d_outputGrad = this->_outputData[0]->device_grad();
	const Dtype* d_preActivationData = _preActivation->device_data();
	Dtype* d_preActivationGrad = _preActivation->mutable_device_grad();

	activation_fn->backward(this->outputTensorDesc, d_outputData, d_outputGrad, d_preActivationData, d_preActivationGrad);
}



template <typename Dtype>
void ConvLayer<Dtype>::_computeFiltersGrad() {
	this->_inputData[0]->print_data("inputData:");
	this->_preActivation->print_grad("preActivationGrad:");

	// d(Cost)/d(Filters)
	const Dtype* d_inputData = this->_inputData[0]->device_data();
	const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	Dtype* d_filtersGrad = _params[Filter]->mutable_device_grad();

	checkCUDNN(cudnnConvolutionBackwardFilter(Cuda::cudnnHandle,
			&Cuda::alpha, this->inputTensorDesc, d_inputData, this->outputTensorDesc, d_preActivationGrad, convDesc, convBwdFilterAlgo, d_workspace, workspaceSize,
			&Cuda::beta, filterDesc, d_filtersGrad));

	this->_params[Filter]->print_grad("filtersGrad:");
}

template <typename Dtype>
void ConvLayer<Dtype>::_computeBiasesGrad() {
	// d(Cost)/d(Biases)
	const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	Dtype* d_biasGrad = _params[Bias]->mutable_device_grad();

	checkCUDNN(cudnnConvolutionBackwardBias(Cuda::cudnnHandle,
			&Cuda::alpha, this->outputTensorDesc, d_preActivationGrad,
			&Cuda::beta, biasTensorDesc, d_biasGrad));
}

template <typename Dtype>
void ConvLayer<Dtype>::_computeInputGrad() {
	// d(Cost)/d(Input)
	const Dtype* d_filtersData = _params[Filter]->device_data();
	const Dtype* d_preActivationGrad = this->_preActivation->device_grad();
	Dtype* d_inputGrad = this->_inputData[0]->mutable_device_grad();
	checkCUDNN(cudnnConvolutionBackwardData(Cuda::cudnnHandle,
			&Cuda::alpha, filterDesc, d_filtersData, this->outputTensorDesc, d_preActivationGrad, convDesc, convBwdDataAlgo, d_workspace, workspaceSize,
			&Cuda::beta, this->inputTensorDesc, d_inputGrad));
	this->_inputData[0]->print_grad("inputGrad:");
	_params[Filter]->print_data("filtersData:");

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




/*
template <typename Dtype>
double ConvLayer<Dtype>::testParamAbnormality() {
	const Dtype* weightGrad = _params[Filter]->host_grad();
	const size_t count = _params[Filter]->getCount();

	double mean = 0.0;
	for(uint32_t i = 0; i < count; i++) {
		mean += weightGrad[i];
	}
	mean /= count;

	double sd = 0.0;
	for(uint32_t i = 0; i < count; i++) {
		sd += (weightGrad[i]-mean)*(weightGrad[i]-mean);
	}
	sd = sqrt(sd/(count-1));


	cout << this->name << ": mean: " << mean << ", sd: " << sd << endl;

	for(uint32_t i = 0; i < count; i++) {
		if(abs(weightGrad[i]-mean) > 10000*sd) {
			return weightGrad[i];
		}
	}
	return DBL_MAX;
}
*/








template ConvLayer<float>::~ConvLayer();
template void ConvLayer<float>::initialize(filter_dim filter_d, update_param weight_update_param, update_param bias_update_param,
		param_filler<float> weight_filler, param_filler<float> bias_filler, typename Activation<float>::Type activationType);
template void ConvLayer<float>::reshape();
template void ConvLayer<float>::_clearShape();
//template void ConvLayer<float>::_save(ofstream &ofs);
//template void ConvLayer<float>::_load(ifstream &ifs, map<Layer<float>*, Layer<float>*> &layerMap);
template void ConvLayer<float>::update();
template void ConvLayer<float>::feedforward();
template void ConvLayer<float>::backpropagation();


#endif
