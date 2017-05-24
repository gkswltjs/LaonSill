/**
 * @file CrossEntropyWithLossLayer.h
 * @date 2017-02-06
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef SIGMOIDWITHLOSSLAYER_H
#define SIGMOIDWITHLOSSLAYER_H 

#include "common.h"
#include "LossLayer.h"
#include "LayerConfig.h"
#include "Activation.h"
#include "ActivationFactory.h"
#include "Cost.h"

template <typename Dtype>
class CrossEntropyWithLossLayer : public LossLayer<Dtype> {
public: 
	class Builder : public LossLayer<Dtype>::Builder {
	public:
        Dtype _targetValue;
        bool _withSigmoid;

		Builder() {
			this->type = Layer<Dtype>::CrossEntropyWithLoss;
            this->_targetValue = 0;
            this->_withSigmoid = false;
		}
		virtual Builder* name(const std::string name) {
			LossLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			LossLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			LossLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			LossLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			LossLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		virtual Builder* lossWeight(const float lossWeight) {
			LossLayer<Dtype>::Builder::lossWeight(lossWeight);
			return this;
		}
		virtual Builder* ignoreLabel(const int ignoreLabel) {
			LossLayer<Dtype>::Builder::ignoreLabel(ignoreLabel);
			return this;
		}
		virtual Builder* normalize(const bool normalize) {
			LossLayer<Dtype>::Builder::normalize(normalize);
			return this;
		}
        /// XXX: 현재는 target value로 구성된 districution과 crosss entropy loss를 구하도록 
        //      되어 있다.
        //       추후에 임의의  value distribution에 대한 cross entropy loss를 구할 수 있어야
        //      한다.
		Builder* targetValue(const Dtype targetValue) {
            this->_targetValue = targetValue;
			return this;
		}
		Builder* withSigmoid(bool withSigmoid) {
            this->_withSigmoid = withSigmoid;
			return this;
		}
		Layer<Dtype>* build() {
			return new CrossEntropyWithLossLayer(this);
		}
	};

    CrossEntropyWithLossLayer();
    CrossEntropyWithLossLayer(Builder* builder);
    CrossEntropyWithLossLayer(const std::string& name);
    virtual ~CrossEntropyWithLossLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	virtual Dtype cost();
    void setTargetValue(Dtype value);

private:
	void initialize(Dtype targetValue, bool withSigmoid);
    Dtype   targetValue;
    int     depth;
    bool    withSigmoid;

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

#endif /* SIGMOIDWITHLOSSLAYER_H */
