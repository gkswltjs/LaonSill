/*
 * Layer.cpp
 *
 *  Created on: 2016. 6. 10.
 *      Author: jhkim
 */

#include "Layer.h"
#include "LayerFactory.h"
#include "../exception/Exception.h"
#include "../network/NetworkConfig.h"


int Layer::layerCount = 0;

int Layer::generateLayerId() {
	return layerCount++;
}





Layer::Layer(const string name) {
	initialize(name);
}

Layer::Layer(Builder* builder) {
	for(uint32_t i = 0; i < builder->_nextLayerIndices.size(); i++) {
		this->nextLayers.push_back((Layer*)((size_t)builder->_nextLayerIndices[i]));
	}
	initialize(builder->_name);
}

#ifndef GPU_MODE
Layer::Layer(const string name, int n_in, int n_out) {
	initialize(name, io_dim(n_in, 1, 1), io_dim(n_out, 1, 1));
}

Layer::~Layer() {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		if(nextLayers[i] && nextLayers[i]->isLastPrevLayerRequest(i)) {
			delete nextLayers[i];
		}
	}
	nextLayers.clear();
	//cout << "destroying " << name << " layer ... " << endl;
}
#else
Layer::~Layer() {
	// 다음 레이어들에 대해 소멸을 요청
	// 현재의 레이어가 요청하는 다음 레이어에 대해 마지막 이전 레이어인 경우,
	// 다음 레이어에 대해 소멸을 요청하게 된다. (multi-branch인 경우 복수의 소멸 요청을 피하기 위해)
	for(UINT i = 0; i < nextLayers.size(); i++) {
		if(nextLayers[i] && nextLayers[i]->isLastPrevLayerRequest(id)) {
			delete nextLayers[i];
		}
	}
	nextLayers.clear();

	_clearShape();
	//checkCudaErrors(cudaFree(d_output));
	//checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
	//checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));
	//cout << "destroying " << name << " layer ... " << endl;
}
#endif



void Layer::addPrevLayer(Layer* prevLayer) {
	prevLayers.push_back(prevLayer);
}

void Layer::addNextLayer(Layer* nextLayer) {
	nextLayers.push_back(nextLayer);
}

bool Layer::isFirstPrevLayerRequest(UINT idx) {
	if(prevLayers.size() < 1) return true;
	if(prevLayers[0]->getId() != idx) return false;
	else return true;
}

bool Layer::isLastPrevLayerRequest(UINT idx) {
	uint32_t numPrevLayers = prevLayers.size();
	if(numPrevLayers < 1) return true;
	if(prevLayers[numPrevLayers-1]->getId() != idx) return false;
	else return true;
}

bool Layer::isFirstNextLayerRequest(UINT idx) {
	if(nextLayers.size() < 1) return true;
	if(nextLayers[0]->getId() != idx) return false;
	else return true;
}

bool Layer::isLastNextLayerRequest(UINT idx) {
	uint32_t numNextLayers = nextLayers.size();
	if(numNextLayers < 1) return true;
	if(nextLayers[numNextLayers-1]->getId() != idx) return false;
	else return true;
}



bool Layer::w_isLastPrevLayerRequest(UINT idx, const string method) {
#ifdef PRINT_CALLSTACK
	cout << method << this->name << "-";
#endif
	bool result = isLastPrevLayerRequest(idx);
#ifdef PRINT_CALLSTACK
	if (!result) cout << "skipped ... " << endl;
	else cout << "entered ... " << endl;
#endif
	return result;
}

bool Layer::w_isLastNextLayerRequest(UINT idx, const string method) {
#ifdef PRINT_CALLSTACK
	cout << method << this->name << "-";
#endif
	bool result = isLastNextLayerRequest(idx);
#ifdef PRINT_CALLSTACK
	if (!result) cout << "skipped ... " << endl;
	else cout << "entered ... " << endl;
#endif
	return result;
}



Layer* Layer::find(UINT idx, const string name) {
	// 레이어 찾기의 경우 마지막 branch의 요청에 대해서만 처리하면 된다.
	if (!w_isLastPrevLayerRequest(idx, "Layer::find()")) return 0;

	Layer* layer = _find(name);
	if(layer) return layer;

	return propFind();
}

void Layer::shape(UINT idx, io_dim in_dim) {
	if (!w_isLastPrevLayerRequest(idx, "Layer::shape()")) return;

	this->in_dim = in_dim;
	_shape();
	propShape();
}

void Layer::reshape(UINT idx, io_dim in_dim) {
	if (!w_isLastPrevLayerRequest(idx, "Layer::reshape()")) return;

	this->in_dim = in_dim;
	_reshape();
	propReshape();
}

void Layer::clearShape(UINT idx) {
	if (!w_isLastPrevLayerRequest(idx, "Layer::clearShape()")) return;

	_clearShape();
	propClearShape();
}

double Layer::sumSquareGrad(UINT idx) {
	if(!w_isLastPrevLayerRequest(idx, "Layer::sumSquareGrad()")) return 0.0;

	double result =_sumSquareGrad();
	result += propSumSquareGrad();
	return result;
}

double Layer::sumSquareParam(UINT idx) {
	if(!w_isLastPrevLayerRequest(idx, "Layer::sumSquareParam()")) return 0.0;

	double result =_sumSquareParam();
	result += propSumSquareParam();
	return result;
}

void Layer::scaleParam(UINT idx, DATATYPE scale_factor) {
	if(!w_isLastPrevLayerRequest(idx, "Layer::scaleParam()")) return;

	_scaleParam(scale_factor);
	propScaleParam(scale_factor);
}

void Layer::save(UINT idx, ofstream &ofs) {
	if(!w_isLastPrevLayerRequest(idx, "Layer::save()")) return;

	_save(ofs);
	propSave(ofs);
}

void Layer::saveHeader(UINT idx, ofstream &ofs) {
	if(!w_isLastPrevLayerRequest(idx, "Layer::saveHeader()")) return;

	Layer *p = this;
	ofs.write((char *)&type, sizeof(int));
	ofs.write((char *)&p, sizeof(Layer *));

	//cout << "save header for " << name << ", type: " << (int)type << ", address: " << p << endl;
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->saveHeader(i, ofs);
	}
}

void Layer::load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	_load(ifs, layerMap);
}

void Layer::update(UINT idx, UINT n, UINT miniBatchSize) {
	if (!w_isLastPrevLayerRequest(idx, "Layer::update()")) return;

	_update(n, miniBatchSize);
	propUpdate(n, miniBatchSize);
}

#ifndef GPU_MODE
void Layer::feedforward(UINT idx, const rcube &input, const char *end=0) {
	propFeedforward(input, end);
}
#else
void Layer::feedforward(UINT idx, const DATATYPE *input, const char *end) {
	_concat(idx, input);
	if (!w_isLastPrevLayerRequest(idx, "Layer::feedforward()")) return;

	//_scaleInput();
	_feedforward();
	propFeedforward(end);
}
#endif









#ifndef GPU_MODE
void Layer::initialize(const string name, io_dim in_dim, io_dim out_dim) {
	strcpy(this->name, name);
	//this->name = name;
	this->in_dim = in_dim;
	this->out_dim = out_dim;
	this->input.set_size(in_dim.rows, in_dim.cols, in_dim.channels);
	this->output.set_size(out_dim.rows, out_dim.cols, out_dim.channels);
}
#else
void Layer::initialize(const string name) {
	this->name = name;
	this->id = Layer::generateLayerId();
}
#endif

void Layer::loadNetwork(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	// fill layer map
	while(true) {
		Layer::Type layerType;
		Layer *address;

		ifs.read((char *)&layerType, sizeof(int));
		ifs.read((char *)&address, sizeof(Layer *));

		//int loc = ifs.tellg();
		//cout << loc << ", " << (int)layerType << endl;

		if(address == 0) break;
		if(layerType == Layer::Input) {
			layerMap.insert(pair<Layer *, Layer *>(address, this));
		}
		else {
			Layer *layer = LayerFactory::create(layerType);
			layerMap.insert(pair<Layer *, Layer *>(address, layer));
			//cout << "created layer type: " << (int)layerType << ", address: " << layer << endl;
		}
	}
	//cout << "map size: " << layerMap.size() << endl;

	Layer *layerKey;
	//ifs.read((char *)&layerKey, sizeof(Layer *));
	//initialize();

	ifs.read((char *)&layerKey, sizeof(Layer *));
	while(ifs && layerKey) {
		Layer *layer = layerMap.find(layerKey)->second;
		if(!layer) throw Exception();

		if(layer->getType() == Layer::Input) {
			Layer::load(ifs, layerMap);
		} else {
			layer->load(ifs, layerMap);
		}
		ifs.read((char *)&layerKey, sizeof(Layer *));
	}
}

void Layer::updateLayerRelation(map<Layer*, Layer*> &layerMap) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i] = layerMap.find((Layer*)nextLayers[i])->second;
	}

	// 학습된 네트워크를 load하는 경우 backward pass가 없으므로 불필요
	//HiddenLayer *hiddenLayer = dynamic_cast<HiddenLayer *>(this);
	//if(hiddenLayer) {
		for(uint32_t i = 0; i < prevLayers.size(); i++) {
			prevLayers[i] = layerMap.find(prevLayers[i])->second;
		}
	//}
}








Layer* Layer::_find(const string name) {
	// 현재 레이어가 찾는 레이어인 경우
	if (this->name == name) {
		return this;
	} else {
		return 0;
	}
}

void Layer::_shape(bool recursive) {
	char message[256];
	sprintf(message, "%s---_shape():in-%dx%dx%dx%d, out-%dx%dx%dx%d", name.c_str(), in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches,
			out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches);
	Util::printMessage(string(message));

	checkCudaErrors(Util::ucudaMalloc(&this->d_input, sizeof(DATATYPE)*in_dim.batchsize()));		//batch size 고려
	checkCudaErrors(Util::ucudaMalloc(&this->d_output, sizeof(DATATYPE)*out_dim.batchsize()));		//batch size 고려

	checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));

	checkCUDNN(cudnnSetTensor4dDescriptor(inputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			in_dim.batches, in_dim.channels, in_dim.rows, in_dim.cols));
	checkCUDNN(cudnnSetTensor4dDescriptor(outputTensorDesc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			out_dim.batches, out_dim.channels, out_dim.rows, out_dim.cols));
}

void Layer::_reshape() {
	// 이전의 input, output 설정과 관련된 memory 정리
	_clearShape();
	_shape();
}

void Layer::_clearShape() {
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	checkCUDNN(cudnnDestroyTensorDescriptor(inputTensorDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(outputTensorDesc));

	d_input = NULL;
	d_output = NULL;
	inputTensorDesc = NULL;
	outputTensorDesc = NULL;
}

double Layer::_sumSquareGrad() {
	return 0.0;
}

double Layer::_sumSquareParam() {
	return 0.0;
}

void Layer::_scaleParam(DATATYPE scale_factor) {
	return;
}

void Layer::_save(ofstream &ofs) {
	Layer *address = this;
	UINT nextLayerSize = nextLayers.size();
	UINT prevLayerSize = prevLayers.size();

	ofs.write((char *)&address, sizeof(Layer *));							// layer address
	ofs.write((char *)&id, sizeof(int));									// layer id
	ofs.write(name.c_str(), LAYER_NAME_LENGTH);								// layer name
	ofs.write((char *)&in_dim, sizeof(io_dim));								// layer in_dim
	ofs.write((char *)&out_dim, sizeof(io_dim));							// layer out_dim
	ofs.write((char *)&nextLayerSize, sizeof(UINT));						// layer next layer size
	for(UINT i = 0; i < nextLayerSize; i++) {								// layer next layers
		ofs.write((char *)&nextLayers[i], sizeof(Layer*));
	}
	ofs.write((char *)&prevLayerSize, sizeof(UINT));						// layer prev layer size
	for(UINT i = 0; i < prevLayers.size(); i++) {							// layer prev layers
		ofs.write((char *)&prevLayers[i], sizeof(Layer*));
	}
}

void Layer::_load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	int layerId;
	char name[LAYER_NAME_LENGTH];
	UINT nextLayerSize, prevLayerSize;

	ifs.read((char *)&layerId, sizeof(int));
	ifs.read(name, LAYER_NAME_LENGTH);
	ifs.read((char *)&in_dim, sizeof(io_dim));
	ifs.read((char *)&out_dim, sizeof(io_dim));
	ifs.read((char *)&nextLayerSize, sizeof(UINT));
	for(UINT i = 0; i < nextLayerSize; i++) {
		Layer* nextLayer;
		ifs.read((char *)&nextLayer, sizeof(Layer*));
		nextLayers.push_back(nextLayer);
	}
	ifs.read((char *)&prevLayerSize, sizeof(UINT));
	for(UINT i = 0; i < prevLayerSize; i++) {
		Layer* prevLayer;
		ifs.read((char *)&prevLayer, sizeof(Layer*));
		prevLayers.push_back(prevLayer);
	}
	initialize(name);

	Layer::_shape(false);
	updateLayerRelation(layerMap);
}

void Layer::_update(UINT n, UINT miniBatchSize) {
	return;
}

void Layer::_feedforward() {
	checkCudaErrors(cudaMemcpyAsync(this->d_output, this->d_input, sizeof(DATATYPE)*in_dim.batchsize(), cudaMemcpyDeviceToDevice));
}

void Layer::_concat(UINT idx, const DATATYPE* input) {
	Util::printDeviceData(input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "input:");
	Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");

	// 첫번째 branch로부터의 input, 그대로 copy
	if(isFirstPrevLayerRequest(idx)) {
		checkCudaErrors(cudaMemcpyAsync(d_input, input, sizeof(DATATYPE)*in_dim.batchsize(), cudaMemcpyDeviceToDevice));
	}
	// 첫번째 이후의 branch로부터의 input, accumulate input
	else {
		checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(in_dim.batchsize()),
				&Cuda::alpha, input, 1, d_input, 1));
	}
	Util::printDeviceData(d_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_input:");
}

void Layer::_scaleInput() {
	if(prevLayers.size() > 1) {
		float branchFactor = 1.0f / prevLayers.size();
		//cout << this->name << "'s feedforward branch factor is " << branchFactor << endl;
		checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(in_dim.batchsize()), &branchFactor, d_input, 1));
	}
}











Layer* Layer::propFind() {
	for (uint32_t i = 0; i < nextLayers.size(); i++) {
		Layer* result = nextLayers[i]->find(id, name);
		if(result != 0) return result;
	}
	return 0;
}

void Layer::propShape() {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->shape(id, out_dim);
	}
}

void Layer::propReshape() {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->reshape(id, out_dim);
	}
}

void Layer::propClearShape() {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->clearShape(id);
	}
}

double Layer::propSumSquareGrad() {
	double result = 0.0;
	for(UINT i = 0; i < nextLayers.size(); i++) {
		result += nextLayers[i]->sumSquareGrad(id);
	}
	return result;
}

double Layer::propSumSquareParam() {
	double result = 0.0;
	for(UINT i = 0; i < nextLayers.size(); i++) {
		result += nextLayers[i]->sumSquareParam(id);
	}
	return result;
}

void Layer::propScaleParam(DATATYPE scale_factor) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->scaleParam(id, scale_factor);
	}
}

void Layer::propSave(ofstream &ofs) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->save(id, ofs);
	}
}

void Layer::propUpdate(UINT n, UINT miniBatchSize) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->update(id, n, miniBatchSize);
	}
}

#ifndef GPU_MODE
void Layer::propFeedforward(const rcube output, const char *end=0) {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->feedforward(i, output, end);
	}
}

void Layer::propResetNParam() {
	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->reset_nabla(i);
	}
}
#else
void Layer::propFeedforward(const char *end) {
	if(end != 0 && name == end) return;

	for(UINT i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->feedforward(id, d_output, end);
	}
}
#endif



