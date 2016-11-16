/*
 * HiddenLayer.cpp
 *
 *  Created on: 2016. 9. 6.
 *      Author: jhkim
 */


#include "HiddenLayer.h"
#include "NetworkConfig.h"

using namespace std;

template <typename Dtype>
HiddenLayer<Dtype>::HiddenLayer() {}

template <typename Dtype>
HiddenLayer<Dtype>::HiddenLayer(Builder* builder) : Layer<Dtype>(builder) {}

template <typename Dtype>
HiddenLayer<Dtype>::HiddenLayer(const string& name) : Layer<Dtype>(name) {}


template <typename Dtype>
HiddenLayer<Dtype>::~HiddenLayer() {}


template <typename Dtype>
void HiddenLayer<Dtype>::backpropagation() {

	//if(!Layer<Dtype>::isSharedOutput()) {
	//	_deconcat(idx, next_input, offset);
	//	if (!this->w_isLastNextLayerRequest(idx, "HiddenLayer::backpropagation()")) return;
	//}

	//_scaleGradient();
	_backpropagation();
	//propBackpropagation();
}



template <typename Dtype>
void HiddenLayer<Dtype>::_backpropagation() {
	this->_inputData[0]->set_device_grad(this->_outputData[0]);
}


template <typename Dtype>
void HiddenLayer<Dtype>::shape() {
	Layer<Dtype>::shape();
}

template <typename Dtype>
void HiddenLayer<Dtype>::_clearShape() {
	Layer<Dtype>::_clearShape();
}


/*
template <typename Dtype>
void HiddenLayer<Dtype>::_deconcat(uint32_t idx, Data<Dtype>* next_delta_input, uint32_t offset) {
	next_delta_input->print_grad("next_delta_input:");
	this->_outputData[0]->print_grad("outputGrad");
	// 첫번째 branch로부터의 backpropagation, 그대로 copy
	if(this->isFirstNextLayerRequest(idx)) {
		this->_outputData[0]->set_device_grad(next_delta_input, offset);
	}
	// 첫번째 이후의 branch로부터의 backpropagation, accumulate gradient
	else {
		this->_outputData[0]->add_device_grad(next_delta_input, offset);
	}
	this->_outputData[0]->print_grad("outputGrad:");
}
*/

/*
template <typename Dtype>
void HiddenLayer<Dtype>::_scaleGradient() {
	if(this->nextLayers.size() > 1) {
		float branchFactor = 1.0f / this->nextLayers.size();
		//cout << this->name << "'s backpropagation branch factor is " << branchFactor << endl;
		this->_outputData[0]->print_grad("before scaling output grad: ");
		this->_outputData[0]->scale_device_grad(branchFactor);
		this->_outputData[0]->print_grad("after scaling output grad: ");
	}
}
*/

/*
template <typename Dtype>
void HiddenLayer<Dtype>::propBackpropagation() {
	HiddenLayer *hiddenLayer;
	for(uint32_t i = 0; i < this->prevLayers.size(); i++) {
		hiddenLayer = dynamic_cast<HiddenLayer *>(this->prevLayers[i]);

		// !!! 대부분의 경우 _backpropagation에서 사용한 inputGrad을 그대로 사용하므로 문제가 없지만
		// DepthConcatLayer와 같이 inputGrad을 분배해야 하는 케이스가 있으므로 inputGrad을 그대로 사용하지 말고
		// getter를 사용하여 이전 레이어에 inputGrad을 전달해야 한다.
		if(hiddenLayer) {
			//_distGradToPrev(i, hiddenLayer);
			hiddenLayer->backpropagation(this->id, this->getInput(), 0);
		}
	}
}
*/




template class HiddenLayer<float>;







