/*
 * SoftmaxWithLossLayer.cpp
 *
 *  Created on: Dec 3, 2016
 *      Author: jkim
 */

#include <cfloat>
#include <vector>

#include "SoftmaxWithLossLayer.h"
#include "MathFunctions.h"
#include "SysLog.h"
#include "PropMgmt.h"
#include "PlanParser.h"

#define SOFTMAXWITHLOSSLAYER_LOG 0

using namespace std;


template <typename Dtype>
SoftmaxWithLossLayer<Dtype>::SoftmaxWithLossLayer()
: SoftmaxWithLossLayer(NULL) {}

template <typename Dtype>
SoftmaxWithLossLayer<Dtype>::SoftmaxWithLossLayer(_SoftmaxWithLossPropLayer* prop)
: LossLayer<Dtype>(),
  prob("prob") {
	this->type = Layer<Dtype>::SoftmaxWithLoss;
	if (prop) {
		this->prop = new _SoftmaxWithLossPropLayer();
		*(this->prop) = *(prop);
	} else {
		this->prop = NULL;
	}

	const bool hasNormalization = GET_PROP(prop, Loss, hasNormalization);
	const bool hasNormalize = GET_PROP(prop, Loss, hasNormalize);
	const bool normalize = GET_PROP(prop, Loss, normalize);
	if (!hasNormalization && hasNormalize)
		GET_PROP(prop, Loss, normalization) = normalize ?
				NormalizationMode::Valid :
				NormalizationMode::BatchSize;

	//InnerLayerFunc::initLayer(0);
	int softmaxAxis = GET_PROP(prop, SoftmaxWithLoss, softmaxAxis);
	stringstream softmaxDef;
	softmaxDef << "{\n";
	softmaxDef << "\t\"name\" : \"inner_softmax\",\n";
	softmaxDef << "\t\"id\" : 0,\n";
	softmaxDef << "\t\"layer\" : \"Softmax\",\n";
	softmaxDef << "\t\"input\" : [\"inner_softmax_7001_input\"],\n";
	softmaxDef << "\t\"output\" : [\"inner_softmax_7001_output\"],\n";
	softmaxDef << "\t\"softmaxAxis\" : " << softmaxAxis << "\n";
	softmaxDef << "}\n";

	_SoftmaxPropLayer* innerProp = new _SoftmaxPropLayer();

	Json::Reader reader;
	Json::Value layer;
	reader.parse(softmaxDef, layer);

	vector<string> keys = layer.getMemberNames();
	string layerType = layer["layer"].asCString();

	for (int j = 0; j < keys.size(); j++) {
		string key = keys[j];
		Json::Value val = layer[key.c_str()];
		if (strcmp(key.c_str(), "layer") == 0) continue;
		if (strcmp(key.c_str(), "innerLayer") == 0) continue;

		PlanParser::setPropValue(val, true, layerType, key,  (void*)innerProp);
	}
	this->softmaxLayer = new SoftmaxLayer<Dtype>(innerProp);

	delete innerProp;
#if 0
#endif

#if SOFTMAXWITHLOSSLAYER_LOG
	cout << "for " << GET_PROP(prop, SoftmaxWithLoss, name) << endl;

	cout << "lossWeight: " << GET_PROP(prop, Loss, lossWeight) << endl;
	cout << "hasIgnoreLabel: " << GET_PROP(prop, Loss, hasIgnoreLabel) << endl;
	cout << "ignoreLabel: " << GET_PROP(prop, Loss, ignoreLabel) << endl;
	cout << "hasNormalize: " << GET_PROP(prop, Loss, hasNormalize) << endl;
	cout << "normalize: " << GET_PROP(prop, Loss, normalize) << endl;
	cout << "hasNormalization: " << GET_PROP(prop, Loss, hasNormalization) << endl;
	cout << "normalization: " << GET_PROP(prop, Loss, normalization) << endl;

	cout << "softmaxAxis: " << GET_PROP(prop, SoftmaxWithLoss, softmaxAxis) << endl;
#endif
}



template <typename Dtype>
SoftmaxWithLossLayer<Dtype>::~SoftmaxWithLossLayer() {
	//InnerLayerFunc::destroyLayer(0);
}


template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::reshape() {
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		this->_outputData[0]->reshape({1, 1, 1, 1});
		this->_outputData[0]->mutable_host_grad()[0] = GET_PROP(prop, Loss, lossWeight);

#if SOFTMAXWITHLOSSLAYER_LOG
		printf("<%s> layer' output-0 has reshaped as: %dx%dx%dx%d\n",
				this->name.c_str(), 1, 1, 1, 1);
#endif
        //InnerLayerFunc::setInOutTensor(0, (void*)this->_inputData[0], true, 0);
        //InnerLayerFunc::setInOutTensor(0, (void*)&this->prob, false, 0);

		SASSERT0(this->softmaxLayer->_inputData.size() == 0);
		SASSERT0(this->softmaxLayer->_outputData.size() == 0);
		this->softmaxLayer->_inputData.push_back(this->_inputData[0]);
		this->softmaxLayer->_outputData.push_back(&this->prob);
	}

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		const vector<uint32_t>& inputDataShape = this->_inputData[i]->getShape();
		this->_inputShape[i] = inputDataShape;

		// "data"
		if (i == 0) {
			this->prob.reshape(inputDataShape);

			//this->softmaxAxis = 1;
			uint32_t softmaxAxis = GET_PROP(prop, SoftmaxWithLoss, softmaxAxis);
			this->outerNum = this->_inputData[0]->getCountByAxis(0, softmaxAxis);
			this->innerNum = this->_inputData[0]->getCountByAxis(softmaxAxis+1);

			//this->_inputData[0]->print_shape();
			//this->_inputData[1]->print_shape();
			//cout << "softmaxAxis=" << softmaxAxis << endl;

			SASSERT(this->outerNum*this->innerNum == this->_inputData[1]->getCount(),
					"Number of labels must match number of predictions ... : %d - %d",
					this->outerNum*this->innerNum, this->_inputData[1]->getCount());

			if (this->_outputData.size() > 1) {
				// softmax output ...
			}


			this->softmaxLayer->reshape();
		}
		// "labels"
	}
}






template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::cost() {
	return this->_outputData[0]->host_data()[0];
}


/****************************************************************************
 * layer callback functions
 ****************************************************************************/
template<typename Dtype>
void* SoftmaxWithLossLayer<Dtype>::initLayer() {
    SoftmaxWithLossLayer* layer = new SoftmaxWithLossLayer<Dtype>();
    return (void*)layer;
}

template<typename Dtype>
void SoftmaxWithLossLayer<Dtype>::destroyLayer(void* instancePtr) {
    SoftmaxWithLossLayer<Dtype>* layer = (SoftmaxWithLossLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void SoftmaxWithLossLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {

    SoftmaxWithLossLayer<Dtype>* layer = (SoftmaxWithLossLayer<Dtype>*)instancePtr;

    if (isInput) {
        SASSERT0(index < 2);
        SASSERT0(layer->_inputData.size() == index);
        layer->_inputData.push_back((Data<Dtype>*)tensorPtr);
    } else {
        SASSERT0(index == 0);
        SASSERT0(layer->_outputData.size() == 0);
        layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
    }
}

template<typename Dtype>
bool SoftmaxWithLossLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    SoftmaxWithLossLayer<Dtype>* layer = (SoftmaxWithLossLayer<Dtype>*)instancePtr;
    layer->reshape();
    return true;
}

template<typename Dtype>
void SoftmaxWithLossLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    SoftmaxWithLossLayer<Dtype>* layer = (SoftmaxWithLossLayer<Dtype>*)instancePtr;
    layer->feedforward();
}

template<typename Dtype>
void SoftmaxWithLossLayer<Dtype>::backwardTensor(void* instancePtr) {
    SoftmaxWithLossLayer<Dtype>* layer = (SoftmaxWithLossLayer<Dtype>*)instancePtr;
    layer->backpropagation();
}

template<typename Dtype>
void SoftmaxWithLossLayer<Dtype>::learnTensor(void* instancePtr) {
    SASSERT0(false);
}

template class SoftmaxWithLossLayer<float>;
