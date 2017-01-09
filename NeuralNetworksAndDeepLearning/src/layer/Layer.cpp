/*
 * Layer.cpp
 *
 *  Created on: 2016. 6. 10.
 *      Author: jhkim
 */



/*
#include "Layer.h"
#include "LayerFactory.h"
#include "../exception/Exception.h"
#include "../network/NetworkConfig.h"
*/

#include "Layer.h"

#include <stddef.h>
#include <utility>

#include "Exception.h"
#include "Layer_device.cu"
#include "LayerFactory.h"
#include "NetworkConfig.h"

using namespace std;

template <typename Dtype>
Layer<Dtype>::Layer(const string& name) {
	initialize(0, name);
}

template <typename Dtype>
Layer<Dtype>::Layer(Builder* builder) {
	this->_inputs = builder->_inputs;
	this->_outputs = builder->_outputs;

	// 사용자가 지정한 propDown이 있는 경우, input의 수와 일치하면 그대로 사용
	if (builder->_propDown.size() > 0) {
		assert(builder->_propDown.size() == this->_inputs.size());
		this->_propDown = builder->_propDown;
	}
	// 사용자가 지정한 propDown이 없는 경우, 첫번째 입력에 대해 propDown, 나머지 입력에 대해 propDown을 막음
	else {
		this->_propDown.resize(this->_inputs.size());
		for (uint32_t i = 0; i < this->_inputs.size(); i++) {
			if (i == 0)
				this->_propDown[i] = true;
			else
				this->_propDown[i] = false;
		}
	}

	initialize(builder->_id, builder->_name);

	cout << this->name << " propDown: ";
	for (uint32_t i = 0; i < this->_propDown.size(); i++) {
		cout << this->_propDown[i] << ", ";
	}
	cout << endl;
}

template <typename Dtype>
Layer<Dtype>::~Layer() {
	// 다음 레이어들에 대해 소멸을 요청
	// 현재의 레이어가 요청하는 다음 레이어에 대해 마지막 이전 레이어인 경우,
	// 다음 레이어에 대해 소멸을 요청하게 된다. (multi-branch인 경우 복수의 소멸 요청을 피하기 위해)
	//for(uint32_t i = 0; i < nextLayers.size(); i++) {
	//	if(nextLayers[i] && nextLayers[i]->isLastPrevLayerRequest(id)) {
	//		delete nextLayers[i];
	//	}
	//}
	//nextLayers.clear();

	_clearShape();
}


template <typename Dtype>
void Layer<Dtype>::reshape() {}


template <typename Dtype>
void Layer<Dtype>::clearShape(uint32_t idx) {
	//if (!w_isLastPrevLayerRequest(idx, "Layer::clearShape()")) return;

	_clearShape();
	//propClearShape();
}


/*
template <typename Dtype>
void Layer<Dtype>::save(uint32_t idx, ofstream &ofs) {
	if(!w_isLastPrevLayerRequest(idx, "Layer::save()")) return;

	_save(ofs);
	propSave(ofs);
}

template <typename Dtype>
void Layer<Dtype>::saveHeader(uint32_t idx, ofstream &ofs) {
	if(!w_isLastPrevLayerRequest(idx, "Layer::saveHeader()")) return;

	Layer<Dtype>* p = this;
	ofs.write((char *)&type, sizeof(int));
	ofs.write((char *)&p, sizeof(Layer<Dtype>*));

	//cout << "save header for " << name << ", type: " << (int)type << ", address: " << p << endl;
	for(uint32_t i = 0; i < nextLayers.size(); i++) {
		nextLayers[i]->saveHeader(i, ofs);
	}
}

template <typename Dtype>
void Layer<Dtype>::load(ifstream &ifs, map<Layer<Dtype>*, Layer<Dtype>*> &layerMap) {
	_load(ifs, layerMap);
}
*/

template <typename Dtype>
void Layer<Dtype>::feedforward() {
	_outputData[0]->set_device_data(_inputData[0]);
}

template <typename Dtype>
void Layer<Dtype>::initialize(uint32_t id, const string name) {
	this->id = id;
	this->name = name;

	//checkCUDNN(cudnnCreateTensorDescriptor(&inputTensorDesc));
	//checkCUDNN(cudnnCreateTensorDescriptor(&outputTensorDesc));
}



template <typename Dtype>
bool Layer<Dtype>::_adjustInputShape() {
	const uint32_t inputSize = _inputData.size();

	// 입력 shape가 입력 데이터만큼 할당되지 않은 경우 해당 사이즈만큼 재할당
	if (_inputShape.size() != inputSize) {
		_inputShape.resize(inputSize);
		for (uint32_t i = 0; i < inputSize; i++) {
			_inputShape[i].resize(4);
		}
		return true;
	} else
		return false;
}


template <typename Dtype>
bool Layer<Dtype>::_isInputShapeChanged(uint32_t index) {
	assert(index < this->_inputData.size());
	assert(this->_inputData.size() == this->_inputShape.size());

	return (this->_inputData[index]->getCount() == 0 ||
			this->_inputData[index]->getShape() != this->_inputShape[index]);
}














#ifndef GPU_MODE
void Layer<Dtype>::_shape(bool recursive) {
	cout << "Layer::_shape() is not implemented in CPU_MODE" << endl;
	exit(1);
}

void Layer<Dtype>::_clearShape() {
	cout << "Layer::_clearShape() is not implemented in CPU_MODE" << endl;
	exit(1);
}

/*
void Layer<Dtype>::_save(ofstream &ofs) {
	cout << "Layer::_save() is not implemented in CPU_MODE" << endl;
	exit(1);
}

void Layer<Dtype>::_load(ifstream &ifs, map<Layer<Dtype>*, Layer<Dtype>*> &layerMap) {
	cout << "Layer::_load() is not implemented in CPU_MODE" << endl;
	exit(1);
}
*/

#endif






template class Layer<float>;




