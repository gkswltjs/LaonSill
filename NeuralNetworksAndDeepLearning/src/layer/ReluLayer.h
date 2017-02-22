/*
 * ReluLayer.h
 *
 *  Created on: Jan 25, 2017
 *      Author: jkim
 */

#ifndef RELULAYER_H_
#define RELULAYER_H_

#include "common.h"
#include "HiddenLayer.h"

template <typename Dtype>
class ReluLayer : public HiddenLayer<Dtype> {
public:
	class Builder : public HiddenLayer<Dtype>::Builder {
	public:
        bool    _useLeaky;
        double  _leaky;

		Builder() {
			this->type = Layer<Dtype>::Relu;
            this->_useLeaky = false;
            this->_leaky = 0.0;
		}
		virtual Builder* name(const std::string name) {
			HiddenLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			HiddenLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			HiddenLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			HiddenLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			HiddenLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}
        Builder* leaky(double leaky) {
            this->_leaky = leaky;
            this->_useLeaky = true;
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
	void initialize(bool useLeaky, double leaky);
    void applyLeakyForward();
    void applyLeakyBackward();

protected:
	// input, output tensor의 desc가 동일하므로 하나만 사용
	cudnnTensorDescriptor_t tensorDesc;	///< cudnn 입력 데이터(n-D 데이터셋) 구조 정보

	cudnnActivationDescriptor_t activationDesc;	///< cudnn 활성화 관련 자료구조에 대한 포인터

    bool useLeaky;      // leaky Relu 사용 여부
    double leaky;       // leaky value
};

#endif /* RELULAYER_H_ */
