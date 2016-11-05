/*
 * InputLayer.cpp
 *
 *  Created on: 2016. 9. 12.
 *      Author: jhkim
 */


#include "InputLayer.h"
#include "../network/NetworkConfig.h"

using namespace std;

template <typename Dtype>
InputLayer<Dtype>::InputLayer() {
	initialize();
}

template <typename Dtype>
InputLayer<Dtype>::InputLayer(const string name) : Layer<Dtype>(name) {
	initialize();
}

template <typename Dtype>
InputLayer<Dtype>::InputLayer(Builder* builder) : Layer<Dtype>(builder) {
	initialize();
}

template <typename Dtype>
InputLayer<Dtype>::~InputLayer() {}


template <typename Dtype>
int InputLayer<Dtype>::getInputSize() const {
	return this->in_dim.rows*this->in_dim.cols*this->in_dim.channels;
}


template <typename Dtype>
//void InputLayer<Dtype>::shape(uint32_t idx, io_dim in_dim) {
void InputLayer<Dtype>::shape() {
	//if (!Layer<Dtype>::w_isLastPrevLayerRequest(idx, "Layer::shape()")) return;

	//this->in_dim = in_dim;

	// class member shared_ptr을 초기화하는 방법이 마땅하지 않아서 ...
	//shared_ptr<Data<Dtype>> _tmp_input(new Data<Dtype>());
	//this->_input = _tmp_input;

	_shape();
	//Layer<Dtype>::propShape();
}

template <typename Dtype>
void InputLayer<Dtype>::feedforward() {
	Layer<Dtype>::feedforward();
}


template <typename Dtype>
void InputLayer<Dtype>::feedforward(DataSet<Dtype>* dataSet, const uint32_t baseIndex, const char* end) {
	//_input->set_data(input, Data::HostToDevice);
	const uint32_t unitSize = this->in_dim.unitsize();

	if(this->networkConfig->_status == NetworkStatus::Train) {
		for(uint32_t i = 0; i < this->in_dim.batches; i++) {
			//cout << "baseIndex: " << baseIndex << ", inBatch: " << i << endl;
			//cout << "src: " << baseIndex+i << ", dst: " << i*unitSize << ", size: " << unitSize << endl;
			const Dtype* ptr = dataSet->getTrainDataAt(baseIndex+i);
			this->_inputData[0]->set_device_with_host_data(ptr, i*unitSize, unitSize);
		}
	} else if(this->networkConfig->_status == NetworkStatus::Test) {
		for(uint32_t i = 0; i < this->in_dim.batches; i++) {
			this->_inputData[0]->set_device_with_host_data(dataSet->getTestDataAt(baseIndex+i), i*unitSize, unitSize);
		}
	}

	this->_feedforward();
	//this->propFeedforward(end);
}

template <typename Dtype>
void InputLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::Input;
}

template <typename Dtype>
void InputLayer<Dtype>::_shape(bool recursive) {
	this->out_dim = this->in_dim;
	if(recursive) {
		Layer<Dtype>::_shape();
	}
}

template <typename Dtype>
void InputLayer<Dtype>::_clearShape() {
	Layer<Dtype>::_clearShape();
}

/*
template <typename Dtype>
void InputLayer<Dtype>::_save(ofstream &ofs) {
	this->saveHeader(0, ofs);
	// header boundary (dummy layer)
	int type = 0;
	Layer<Dtype>* layer = 0;
	ofs.write((char*)&type, sizeof(int));
	ofs.write((char*)&layer, sizeof(Layer<Dtype>*));

	Layer<Dtype>::_save(ofs);
}

template <typename Dtype>
void InputLayer<Dtype>::_load(ifstream& ifs, map<Layer<Dtype>*, Layer<Dtype>*>& layerMap) {
	initialize();
	InputLayer::_shape(false);
	this->loadNetwork(ifs, layerMap);
}
*/


template class InputLayer<float>;







