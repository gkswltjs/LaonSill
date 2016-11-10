/*
 * InputLayer.cpp
 *
 *  Created on: 2016. 9. 12.
 *      Author: jhkim
 */


#include "InputLayer.h"
#include "../network/NetworkConfig.h"
#include "../dataset/ImagePackDataSet.h"

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

	if (builder->_sourceType == "ImagePack") {
		_dataSet = new ImagePackDataSet<Dtype>(
				builder->_source+"/train_data",
				builder->_source+"/train_label",
				1,
				builder->_source+"/test_data",
				builder->_source+"/test_label",
				1);
		_dataSet->setMean({0.13066047740});
		_dataSet->load();
	} else {
		cout << "Unsuppored Input Source Type: " << builder->_sourceType;
		exit(1);
	}

	initialize();
}

template <typename Dtype>
InputLayer<Dtype>::~InputLayer() {}


template <typename Dtype>
int InputLayer<Dtype>::getInputSize() const {
	return this->in_dim.rows*this->in_dim.cols*this->in_dim.channels;
}


template <typename Dtype>
void InputLayer<Dtype>::shape() {
	this->in_dim.batches = this->networkConfig->_batchSize;
	this->in_dim.channels = this->_dataSet->getChannels();
	this->in_dim.rows = this->_dataSet->getRows();
	this->in_dim.cols = this->_dataSet->getCols();

	_shape();
}

template <typename Dtype>
void InputLayer<Dtype>::feedforward() {
	Layer<Dtype>::feedforward();
}


template <typename Dtype>
void InputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
	const uint32_t unitSize = this->in_dim.unitsize();

	if (this->networkConfig->_status == NetworkStatus::Train) {
		// data
		for (uint32_t i = 0; i < this->in_dim.batches; i++) {
			const Dtype* ptr = _dataSet->getTrainDataAt(baseIndex+i);
			this->_inputData[0]->set_device_with_host_data(ptr, i*unitSize, unitSize);
		}

		// label
		if (this->_inputs.size() > 1) {
			for (uint32_t i = 0; i < this->in_dim.batches; i++) {
				const Dtype* ptr = _dataSet->getTrainLabelAt(baseIndex+i);
				this->_inputData[1]->set_device_with_host_data(ptr, i, 1);
			}
		}

	} else if (this->networkConfig->_status == NetworkStatus::Test) {
		for(uint32_t i = 0; i < this->in_dim.batches; i++) {
			const Dtype* ptr = _dataSet->getTestDataAt(baseIndex+i);
			this->_inputData[0]->set_device_with_host_data(ptr, i*unitSize, unitSize);
		}

		if (this->_inputs.size() > 1) {
			for (uint32_t i = 0; i < this->in_dim.batches; i++) {
				const Dtype* ptr = _dataSet->getTestLabelAt(baseIndex+i);
				this->_inputData[1]->set_device_with_host_data(ptr, i, 1);
			}
		}
	}
	Layer<Dtype>::feedforward();
}

template <typename Dtype>
void InputLayer<Dtype>::initialize() {
	this->type = Layer<Dtype>::Input;
}

template <typename Dtype>
void InputLayer<Dtype>::_shape(bool recursive) {
	this->out_dim = this->in_dim;

	for (uint32_t i = 0; i < this->_outputs.size(); i++) {
		//this->_inputs.insert(this->_inputs.begin(), this->_outputs[0]);
		//this->_inputData.insert(this->_inputData.begin(), this->_outputData[0]);
		this->_inputs.push_back(this->_outputs[i]);
		this->_inputData.push_back(this->_outputData[i]);
	}

	if (this->_outputs.size() > 1) {
		this->_inputData[1]->shape({this->in_dim.batches, 1, 1, 1});
	}

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







