/*
 * SplitLayer.cpp
 *
 *  Created on: Nov 8, 2016
 *      Author: jkim
 */

#include <vector>

#include "SplitLayer.h"

#define SPLITLAYER_LOG 0

using namespace std;

template <typename Dtype>
SplitLayer<Dtype>::SplitLayer(const std::string& name)
	: Layer<Dtype>(name) {

}

template <typename Dtype>
SplitLayer<Dtype>::SplitLayer(Builder* builder)
	: Layer<Dtype>(builder) {
	initialize();

	tempCount = 0;
}



template <typename Dtype>
SplitLayer<Dtype>::~SplitLayer() {}



template <typename Dtype>
void SplitLayer<Dtype>::initialize() {

}

template <typename Dtype>
void SplitLayer<Dtype>::reshape() {
	Layer<Dtype>::_adjustInputShape();

	if (!Layer<Dtype>::_isInputShapeChanged(0))
		return;

	const vector<uint32_t>& inputShape = this->_inputData[0]->getShape();
	uint32_t batches 	= inputShape[0];
	uint32_t channels 	= inputShape[1];
	uint32_t rows 		= inputShape[2];
	uint32_t cols 		= inputShape[3];

	this->_inputShape[0] = inputShape;

	for (uint32_t i = 0; i < this->_outputData.size(); i++) {
		this->_outputData[i]->reshape(inputShape);
	}

	/*
	this->setInDimension(this->_inputData[0]->getShape());
	this->out_dim = this->in_dim;

	// 일단 SplitLayer는 forward pass에서
	// 하나의 output이 여러 input으로 전달되는 경우를 담당한다고 전제
	// 입력은 항상 1개로 상정

	//for (uint32_t i = 1; i < this->_inputs.size(); i++) {
	//	if (this->_inputData[i]->getCount() == 0)
	//		this->_inputData[i]->reshape({this->in_dim.batches, this->in_dim.channels,
	//		this->in_dim.rows, this->in_dim.cols});
	//}

	// 모든 output data에 대해 shape처리한다.
	//	ㄴLayer의 _shape()가 호출되어도 이미지 shape처리되었기 때문에 무시된다.
	for (uint32_t i = 0; i < this->_outputs.size(); i++) {
		if (this->_outputData[i]->getCount() == 0)
			this->_outputData[i]->reshape({this->out_dim.batches, this->out_dim.channels,
			this->out_dim.rows, this->out_dim.cols});
	}

	if(recursive) {
		Layer<Dtype>::_shape();
	}
	*/
}

template <typename Dtype>
void SplitLayer<Dtype>::feedforward() {
	reshape();

	for (uint32_t i = 0; i < this->_outputs.size(); i++) {
		this->_outputData[i]->set_device_data(this->_inputData[0]);
	}
}

template <typename Dtype>
void SplitLayer<Dtype>::backpropagation() {

	/*
	if (this->name == "conv4_3_norm_conv4_3_norm_0_split") {
		this->tempCount++;
		if (this->tempCount == 2) {
			for (int i = 0; i < this->_outputData.size(); i++) {
				cout << this->_outputData[i]->asum_device_grad() << endl;
			}
		}
	}
	*/

#if SPLITLAYER_LOG
	const string targetLayer = "rpn/output-split";
#endif

	this->_inputData[0]->reset_device_grad();
	for (uint32_t i = 0; i < this->_outputs.size(); i++) {

#if SPLITLAYER_LOG
		if (this->name == targetLayer) {
			Data<Dtype>::printConfig = true;
			this->_outputData[i]->print_grad({}, false);
			Data<Dtype>::printConfig = false;
		}
#endif
		this->_inputData[0]->add_device_grad(this->_outputData[i]);
#if SPLITLAYER_LOG
		if (this->name == targetLayer) {
			Data<Dtype>::printConfig = true;
			this->_inputData[0]->print_grad({}, false);
			Data<Dtype>::printConfig = false;
		}
#endif
	}
#if SPLITLAYER_LOG
	if (this->name == targetLayer) {
		exit(1);
	}
#endif

	/*
	if (this->name == "conv4_3_norm_conv4_3_norm_0_split") {
		if (this->tempCount == 2) {
			cout << "input asum: " << this->_inputData[0]->asum_device_grad() << endl;
			exit(1);
		}
	}
	*/

}

template class SplitLayer<float>;
