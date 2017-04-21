/**
 * @file YOLOOutputLayer.h
 * @date 2017-04-21
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef YOLOOUTPUTLAYER_H
#define YOLOOUTPUTLAYER_H 

#include "common.h"
#include "LossLayer.h"
#include "LayerConfig.h"
#include "Activation.h"
#include "ActivationFactory.h"
#include "Cost.h"

template<typename Dtype>
class YOLOOutputLayer : public LossLayer<Dtype> {
public: 
	class Builder : public LossLayer<Dtype>::Builder {
	public:
        float _noobj;
        float _coord;

		Builder() {
			this->type = Layer<Dtype>::YoloLoss;
            this->_noobj = 0.5;
            this->_coord = 5.0;
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
        Builder* lambda(float noobj, float coord) {
            this->_noobj = noobj;
            this->_coord = coord;
            return this;
        }
		Layer<Dtype>* build() {
			return new YOLOOutputLayer(this);
		}
	};

    YOLOOutputLayer();
    YOLOOutputLayer(Builder* builder);
    virtual ~YOLOOutputLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();
	virtual Dtype cost();

private:
	void initialize(float coord, float noobj);
    float noobj;
    float coord;
};

#endif /* YOLOOUTPUTLAYER_H */
