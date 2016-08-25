/*
 * PoolingLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */

#include "PoolingLayer.h"


PoolingLayer::PoolingLayer() {
	this->type = Layer::Pool;
}

PoolingLayer::PoolingLayer(Builder* builder)
	: HiddenLayer(builder) {
	initialize(builder->_poolDim, builder->_poolingType);
}

PoolingLayer::PoolingLayer(const string name, pool_dim pool_d, Pooling::Type poolingType)
	: HiddenLayer(name) {
	initialize(pool_d, poolingType);
}

#ifndef GPU_MODE
PoolingLayer::~PoolingLayer() {
	PoolingFactory::destroy(pooling_fn);
}
#else
PoolingLayer::~PoolingLayer() {
	//checkCudaErrors(cudaFree(d_delta));
	//checkCudaErrors(cudaFree(d_delta_input));

	PoolingFactory::destroy(pooling_fn);
}
#endif


#ifndef GPU_MODE
void PoolingLayer::initialize(pool_dim pool_d, Pooling::Type poolingType) {
	this->type = Layer::Pool;

	this->out_dim.rows = in_dim.rows / pool_d.rows;
	this->out_dim.cols = in_dim.rows / pool_d.cols;
	this->out_dim.channels = in_dim.channels;

	//this->output.set_size(out_dim.rows, out_dim.cols, out_dim.channels);

	this->pool_d = pool_d;

	this->pooling_fn = PoolingFactory::create(poolingType);

	this->pool_map.set_size(in_dim.rows/pool_d.stride, in_dim.cols/pool_d.stride, in_dim.channels);
	this->output.set_size(size(pool_map));
	this->delta_input.set_size(size(input));
	this->delta_input.zeros();
}
#else
void PoolingLayer::initialize(pool_dim pool_d, Pooling::Type poolingType) {
	this->type = Layer::Pool;
	this->pool_d = pool_d;
	this->pooling_fn = PoolingFactory::create(poolingType, pool_d);
}
#endif


void PoolingLayer::_save(ofstream &ofs) {
	HiddenLayer::_save(ofs);

	int poolingType = (int)pooling_fn->getType();

	ofs.write((char *)&pool_d, sizeof(pool_dim));
	ofs.write((char *)&poolingType, sizeof(int));
}

void PoolingLayer::_load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	HiddenLayer::_load(ifs, layerMap);

	pool_dim pool_d;
	Pooling::Type poolingType;

	ifs.read((char *)&pool_d, sizeof(pool_dim));
	ifs.read((char *)&poolingType, sizeof(Pooling::Type));

	initialize(pool_d, poolingType);

	PoolingLayer::_shape(false);
}

#ifndef GPU_MODE
void PoolingLayer::_feedforward(UINT idx, const rcube &input, const char *end=0) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	Util::convertCube(input, this->input);
	pooling_fn->pool(pool_d, this->input, pool_map, output);

	propFeedforward(this->output, end);
}

void PoolingLayer::backpropagation(UINT idx, HiddenLayer *next_layer) {
	// TODO w_next_delta를 모두 합하여 한 번에 d_pool하는 것이 연산적으로 유리, 수정 필요
	rcube w_next_delta(size(output));

	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);
	Util::printCube(next_layer->getDeltaInput(), "delta input:");
	Util::printCube(w_next_delta, "w_next_delta:");

	rcube temp(size(delta_input));
	pooling_fn->d_pool(pool_d, w_next_delta, pool_map, temp);
	delta_input += temp;
	Util::printCube(delta_input, "delta_input:");


	// dx가 모두 aggregate된 후 이전 레이어로 back propagate한다.
	if(!isLastNextLayerRequest(idx)) return;

	propBackpropagation();
	delta_input.zeros();
}
#else
void PoolingLayer::_shape(bool recursive) {
	cudnnTensorDescriptor_t tempInputTensorDesc;
	checkCUDNN(cudnnCreateTensorDescriptor(&tempInputTensorDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(tempInputTensorDesc,
				CUDNN_TENSOR_NCHW,
				CUDNN_DATA_FLOAT,
				in_dim.batches, in_dim.channels, in_dim.rows, in_dim.cols));

	int n, c, h, w;
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(pooling_fn->getPoolDesc(),
			tempInputTensorDesc,
			&n, &c, &h, &w));

	out_dim.batches = n;
	out_dim.channels = c;
	out_dim.rows = h;
	out_dim.cols = w;

	checkCUDNN(cudnnDestroyTensorDescriptor(tempInputTensorDesc));

	if(recursive) {
		HiddenLayer::_shape();
	}

	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta, sizeof(DATATYPE)*out_dim.batchsize()));
	//checkCudaErrors(Util::ucudaMalloc(&this->d_delta_input, sizeof(DATATYPE)*in_dim.batchsize()));
}

void PoolingLayer::_clearShape() {
	//checkCudaErrors(cudaFree(d_delta));
	//checkCudaErrors(cudaFree(d_delta_input));

	//d_delta = NULL;
	//d_delta_input = 0;

	HiddenLayer::_clearShape();
}

void PoolingLayer::_feedforward() {
	//this->d_input = input;

	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	_input->print_data("d_input:");
	const DATATYPE* d_input = _input->gpu_data();
	DATATYPE* d_output = _output->mutable_gpu_data();
	pooling_fn->pool(inputTensorDesc, d_input, outputTensorDesc, d_output);

	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, 1, 1, this->name+string("/d_output:"));
	_output->print_data(this->name+string("/d_output:"));
}

void PoolingLayer::_backpropagation() {
	// backpropagate delta to delta_input
	//Util::printDeviceData(d_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_output:");
	//Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
	_output->print_data("d_output:");
	_input->print_data("d_input:");

	const DATATYPE* d_output = _output->gpu_data();
	const DATATYPE* d_delta_output = _output->gpu_grad();
	const DATATYPE* d_input = _input->gpu_data();
	DATATYPE* d_delta_input = _input->mutable_gpu_grad();
	pooling_fn->d_pool(outputTensorDesc, d_output, d_delta_output, inputTensorDesc, d_input, d_delta_input);

	//Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_delta_input:");
	_input->print_grad("d_delta_input:");
}
#endif




































