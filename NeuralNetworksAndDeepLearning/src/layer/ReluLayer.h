/*
 * ReluLayer.h
 *
 *  Created on: Jan 25, 2017
 *      Author: jkim
 */

#ifndef RELULAYER_H_
#define RELULAYER_H_

#include "common.h"
#include "Layer.h"

template <typename Dtype>
class ReluLayer : public Layer<Dtype> {
public:
	class Builder : public Layer<Dtype>::Builder {
	public:
		Builder() {
			this->type = Layer<Dtype>::Relu;
		}
		virtual Builder* name(const std::string name) {
			Layer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			Layer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			Layer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			Layer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			Layer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		Layer<Dtype>* build() {
			return new ReluLayer(this);
		}
	};

	ReluLayer(Builder* builder);
	virtual ~ReluLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

protected:
	void initialize();



protected:
	// input, output tensor의 desc가 동일하므로 하나만 사용
	cudnnTensorDescriptor_t tensorDesc;	///< cudnn 입력 데이터(n-D 데이터셋) 구조 정보

	cudnnActivationDescriptor_t activationDesc;	///< cudnn 활성화 관련 자료구조에 대한 포인터
};

#endif /* RELULAYER_H_ */
