/**
 * @file HyperTangentLayer.h
 * @date 2017-03-03
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef HYPERTANGENTLAYER_H
#define HYPERTANGENTLAYER_H 

#include "common.h"
#include "LearnableLayer.h"
#include "LayerConfig.h"
#include "Activation.h"
#include "ActivationFactory.h"
#include "Cost.h"

template<typename Dtype>
class HyperTangentLayer : public Layer<Dtype> {
public: 
	class Builder : public Layer<Dtype>::Builder {
	public:
		Builder() {
			this->type = Layer<Dtype>::Sigmoid2;
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
			return new HyperTangentLayer(this);
		}
	};

	/**
	 * @details FullyConnectedLayer 기본 생성자
	 *          내부적으로 레이어 타입만 초기화한다.
	 */
	HyperTangentLayer();
	HyperTangentLayer(Builder* builder);

    HyperTangentLayer(const std::string name);
    virtual ~HyperTangentLayer() {}

	virtual void backpropagation();
	virtual void reshape();
	virtual void feedforward();

private:
    void initialize();
};

#endif /* HYPERTANGENTLAYER_H */