#include <CImg.h>
#include <stddef.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda/Cuda.h"
#include "dataset/DataSet.h"
#include "dataset/MockDataSet.h"
#include "debug/Debug.h"
#include "evaluation/Top1Evaluation.h"
#include "evaluation/Top5Evaluation.h"
#include "monitor/NetworkMonitor.h"
#include "network/Network.h"
#include "network/NetworkConfig.h"
#include "Util.h"

using namespace std;

void network_test();
void cuda_gemm_test();
void cuda_conv_test();




int main(int argc, char** argv) {
	cout << "main" << endl;
	cout.precision(11);
	cout.setf(ios::fixed);
	Util::setOutstream(&cout);
	//Util::setOutstream("./log");

	network_test();

	cout << "end" << endl;
	return 0;
}


void network_test() {
	Cuda::create(0);
	cout << "Cuda creation done ... " << endl;
	Util::setPrint(false);

	const uint32_t maxEpoch = 1000;
	const uint32_t batchSize = 10;
	const float baseLearningRate = 0.001f;
	const float weightDecay = 0.002f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;

	//DataSet* dataSet = new MockDataSet(28, 28, 1, 10, 10, 10);
	//DataSet* dataSet = new MockDataSet(56, 56, 3, 10, 10, 10);
	//DataSet* dataSet = new MnistDataSet(0.8);
	//DataSet* dataSet = new MockDataSet(224, 224, 3, 100, 100, 100);
	//DataSet* dataSet = createImageNet10CatDataSet();
	//DataSet* dataSet = createImageNet100CatDataSet();
	DataSet* dataSet = createMnistDataSet();
	dataSet->load();
	dataSet->zeroMean(true);

	Evaluation* top1Evaluation = new Top1Evaluation();
	Evaluation* top5Evaluation = new Top5Evaluation();
	NetworkListener* top1Listener = new NetworkMonitor(maxEpoch);
	NetworkListener* top5Listener = new NetworkMonitor(maxEpoch);

	//LayersConfig* layersConfig = createCNNSimpleLayersConfig();
	LayersConfig* layersConfig = createCNNDoubleLayersConfig();
	//LayersConfig* layersConfig = createGoogLeNetLayersConfig();
	//LayersConfig* layersConfig = createInceptionLayersConfig();
	//LayersConfig* layersConfig = createGoogLeNetInception3ALayersConfig();
	//LayersConfig* layersConfig = createGoogLeNetInception3ASimpleLayersConfig();
	NetworkConfig* networkConfig =
			(new NetworkConfig::Builder())
			->batchSize(batchSize)
			->baseLearningRate(baseLearningRate)
			->weightDecay(weightDecay)
			->momentum(momentum)
			->clipGradientsLevel(clipGradientsLevel)
			->dataSet(dataSet)
			->evaluations({top1Evaluation, top5Evaluation})
			->networkListeners({top1Listener, top5Listener})
			->layersConfig(layersConfig)
			->build();

	Network* network = new Network(networkConfig);
	network->shape();
	network->sgd(maxEpoch);

	Cuda::destroy();
}







void cuda_gemm_test() {
	float alpha = 1.0f, beta = 0.0f;
	float *d_a, *d_b, *d_c, *d_result;
	int m = 3, k = 5, n = 2;

	float a_r[] = {
			0.1, 0.2, 0.3, 0.4, 0.5,
			0.6, 0.7, 0.8, 0.9, 1.0,
			1.1, 1.2, 1.3, 1.4, 1.5
	};
	float a_c[] = {
			0.1, 0.6, 1.1,
			0.2, 0.7, 1.2,
			0.3, 0.8, 1.3,
			0.4, 0.9, 1.4,
			0.5, 1.0, 1.5
	};

	/*
	float b_r[] = {
			0.1, 0.2,
			0.4, 0.5,
			0.7, 0.8,
			1.0, 1.1,
			1.3, 1.4,
	};
	float b_c[] = {
			0.1, 0.4, 0.7, 1.0, 1.3,
			0.2, 0.5, 0.8, 1.1, 1.4
	};
	*/

	float c_r[] = {
			0.1, 0.2, 0.3, 0.4,
			0.5, 0.6, 0.7, 0.8,
			0.9, 1.0, 1.1, 1.2
	};
	float c_c[] = {
			0.1, 0.5, 0.9,
			0.2, 0.6, 1.0,
			0.3, 0.7, 1.1,
			0.4, 0.8, 1.2
	};



	Cuda::create(0);

	checkCudaErrors(cudaMalloc(&d_a, sizeof(float)*3*5));
	checkCudaErrors(cudaMalloc(&d_c, sizeof(float)*3*4));
	checkCudaErrors(cudaMalloc(&d_result, sizeof(float)*5*4));

	checkCudaErrors(cudaMemcpyAsync(d_a, a_c, sizeof(float)*3*5, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_c, c_c, sizeof(float)*3*4, cudaMemcpyHostToDevice));

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
				5, 4, 3,
				&alpha,
				d_a, 3,
				d_c, 3,
				&beta,
				d_result, 5));

	float *host = new float[5*4];
	checkCudaErrors(cudaMemcpyAsync(host, d_result, sizeof(DATATYPE)*5*4, cudaMemcpyDeviceToHost));
	for(int i = 0; i < 5*4; i++) {
		cout << host[i] << ", ";
	}
	cout << endl;

	checkCudaErrors(cudaFree(d_a));
	//checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_c));
	checkCudaErrors(cudaFree(d_result));

	/*
	checkCudaErrors(cudaMalloc(&d_a, sizeof(float)*m*k));
	checkCudaErrors(cudaMalloc(&d_b, sizeof(float)*k*n));
	checkCudaErrors(cudaMalloc(&d_c, sizeof(float)*m*n));

	checkCudaErrors(cudaMemcpyAsync(d_a, a_c, sizeof(float)*m*k, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyAsync(d_b, b_c, sizeof(float)*k*n, cudaMemcpyHostToDevice));

	checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				m, n, k,
				&alpha,
				d_a, m,
				d_b, k,
				&beta,
				d_c, m));

	float *host = new float[m*n];
	checkCudaErrors(cudaMemcpyAsync(host, d_c, sizeof(DATATYPE)*m*n, cudaMemcpyDeviceToHost));
	for(int i = 0; i < m*n; i++) {
		cout << host[i] << ", ";
	}
	cout << endl;

	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));
	checkCudaErrors(cudaFree(d_c));
	*/

	Cuda::destroy();




	/*
	Cuda::create(0);

	MockDataSet *dataSet = new MockDataSet(3, 3, 1, 10, 10);
	dataSet->load();

	int epoch = 10;
	int batchSize = 1;
	double lr_mult = 0.1;
	double decay_mult = 0.1;

	InputLayer *inputLayer = new InputLayer(
			"input",
			io_dim(3, 3, 1, batchSize));

	HiddenLayer *fc1Layer = new FullyConnectedLayer(
			"fc1",
			io_dim(3*3*1, 1, 1, batchSize),
			io_dim(5, 1, 1, batchSize),
			0.5,
			update_param(lr_mult, decay_mult),
			update_param(lr_mult, decay_mult),
			param_filler(ParamFillerType::Xavier),
			param_filler(ParamFillerType::Gaussian, 1),
			Activation::ReLU);

	OutputLayer *softmaxLayer = new SoftmaxLayer(
			"softmax",
			io_dim(5, 1, 1, batchSize),
			io_dim(2, 1, 1, batchSize),
			0.5,
			update_param(lr_mult, decay_mult),
			update_param(lr_mult, decay_mult),
			param_filler(ParamFillerType::Constant, 0.1),
			param_filler(ParamFillerType::Gaussian, 1));

	Network::addLayerRelation(inputLayer, fc1Layer);
	Network::addLayerRelation(fc1Layer, softmaxLayer);

	Network network(inputLayer, softmaxLayer, dataSet, 0);
	network.sgd(epoch, batchSize);

	Cuda::destroy();
	*/
}


void cuda_conv_test() {
		Cuda::create(0);

		float alpha = 1.0f, beta = 0.0f;
		int n = 1, c = 1, h = 8, w = 8;

		// 입력 데이터
		float *data = new float[n*c*h*w];
		for(int i = 0; i < n*c*h*w; i++) {
			data[i] = (float)rand() / RAND_MAX;
			//cout << data[i] << ", ";
		}
		//cout << endl;
		Util::printData(data, h, w, c, n, "data:");
		float *d_data;
		checkCudaErrors(cudaMalloc(&d_data, sizeof(float)*n*c*h*w));
		checkCudaErrors(cudaMemcpyAsync(d_data, data,	sizeof(float)*n*c*h*w, cudaMemcpyHostToDevice));

		// 입력 텐서
		cudnnTensorDescriptor_t dataTensor;
		checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
		checkCUDNN(cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

		int kernel = 3;
		int conv_in_channel = 1;
		int conv_out_channel = 1;
		// 컨볼루션 필터 디스크립터
		cudnnFilterDescriptor_t conv1filterDesc;
		checkCUDNN(cudnnCreateFilterDescriptor(&conv1filterDesc));
		checkCUDNN(cudnnSetFilter4dDescriptor(conv1filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, conv_out_channel, conv_in_channel, kernel, kernel));

		// 컨볼루션 필터
		int pconv1size = kernel*kernel*conv_in_channel*conv_out_channel;
		float *pconv1 = new float[pconv1size];
		for(int i = 0; i < n*c*h*w; i++) {
			pconv1[i] = (float)rand() / RAND_MAX;
			//cout << pconv1[i] << ", ";
		}
		//cout << endl;
		Util::printData(pconv1, kernel, kernel, conv_in_channel, conv_out_channel, "pconv1:");
		float *d_pconv1;
		checkCudaErrors(cudaMalloc(&d_pconv1, sizeof(float)*pconv1size));
		checkCudaErrors(cudaMemcpyAsync(d_pconv1, pconv1, sizeof(float)*pconv1size, cudaMemcpyHostToDevice));

		// 컨볼루션 디스크립터
		cudnnConvolutionDescriptor_t conv1Desc;
		checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1Desc));
		// pad-h, pad-w, stride-v, stride-h, upscale-x, upscale-y
		checkCUDNN(cudnnSetConvolution2dDescriptor(conv1Desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));

		// 출력 텐서
		cudnnTensorDescriptor_t conv1Tensor;
		checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));
		checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv1Desc, dataTensor, conv1filterDesc, &n, &c, &h, &w));
		cout << "n: " << n << ", c: " << c << ", h: " << h << ", w: " << w << endl;
		checkCUDNN(cudnnSetTensor4dDescriptor(conv1Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

		// 컨볼루션 알고리즘
		cudnnConvolutionFwdAlgo_t conv1algo;
		checkCUDNN(cudnnGetConvolutionForwardAlgorithm(Cuda::cudnnHandle,
				dataTensor,
				conv1filterDesc,
				conv1Desc,
				conv1Tensor,
			  CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			    0,
			  &conv1algo));

		//
		size_t sizeInBytes = 0;
		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(Cuda::cudnnHandle,
				dataTensor,
				conv1filterDesc,
				conv1Desc,
				conv1Tensor,
				conv1algo,
				&sizeInBytes));

		void *d_cudnn_workspace = nullptr;
		size_t m_workspaceSize = sizeInBytes;
		if(m_workspaceSize > 0) {
				checkCudaErrors(cudaMalloc(&d_cudnn_workspace, m_workspaceSize));
		}

		//float *conv1 = new float[n*c*h*w];
		float *d_conv1;
		checkCudaErrors(cudaMalloc(&d_conv1, sizeof(float)*n*c*h*w));

		// 컨볼루션 수행
		checkCUDNN(cudnnConvolutionForward(Cuda::cudnnHandle, &alpha, dataTensor,
			   d_data, conv1filterDesc, d_pconv1, conv1Desc,
			   conv1algo, d_cudnn_workspace, m_workspaceSize, &beta,
			   conv1Tensor, d_conv1));



		//float *conv1 = new float[n*c*h*w];
		//checkCudaErrors(cudaMemcpyAsync(conv1, d_conv1, sizeof(DATATYPE)*n*c*h*w, cudaMemcpyDeviceToHost));
		//Util::printData(conv1, h, w, c, n, "conv1:");
		Util::printDeviceData(d_conv1, h, w, c, n, "conv1:");

		//////////////////////////////////////////////////////////////////////////

		int pool_stride = 2;
		int pool_size = 2;
		cudnnPoolingDescriptor_t poolDesc;
		checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
		checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, pool_size, pool_size, 0, 0, pool_stride, pool_stride));

		cudnnTensorDescriptor_t pool1Tensor;
		checkCUDNN(cudnnCreateTensorDescriptor(&pool1Tensor));
		checkCUDNN(cudnnSetTensor4dDescriptor(pool1Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h/pool_stride, w/pool_stride));

		float *d_pool1;
		int pool1_size = n*c*(h/pool_stride)*(w/pool_stride);
		checkCudaErrors(cudaMalloc(&d_pool1, sizeof(float)*pool1_size));

		checkCUDNN(cudnnPoolingForward(Cuda::cudnnHandle, poolDesc, &alpha, conv1Tensor,
					d_conv1, &beta, pool1Tensor, d_pool1));

		//float *pool1 = new float[n*c*h*w];
		//checkCudaErrors(cudaMemcpyAsync(pool1, d_pool1, sizeof(DATATYPE)*pool1_size, cudaMemcpyDeviceToHost));
		//Util::printData(pool1, h/pool_stride, w/pool_stride, c, n, "pool1:");
		Util::printDeviceData(d_pool1, h/pool_stride, w/pool_stride, c, n, "pool1:");


		Cuda::destroy();

}









