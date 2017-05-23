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
        bool    _useLeaky;
        double  _leaky;

		Builder() {
			this->type = Layer<Dtype>::Relu;
            this->_useLeaky = false;
            this->_leaky = 0.0;
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
    ReluLayer(const std::string& name);
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

public:
    /****************************************************************************
     * layer callback functions 
     ****************************************************************************/
    static void* initLayer();
    static void destroyLayer(void* instancePtr);
    static void setInOutTensor(void* instancePtr, void* tensorPtr, bool isInput, int index);
    static bool allocLayerTensors(void* instancePtr);
    static void forwardTensor(void* instancePtr, int miniBatchIndex);
    static void backwardTensor(void* instancePtr);
    static void learnTensor(void* instancePtr);
};

#endif /* RELULAYER_H_ */
