/**
 * @file AAALayer.h
 * @date 2017-01-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef BATCHNORMLAYER_H
#define BATCHNORMLAYER_H 

#include "common.h"
#include "HiddenLayer.h"
#include "LearnableLayer.h"
#include "LayerConfig.h"
#include "Activation.h"
#include "ActivationFactory.h"
#include "Cost.h"

template <typename Dtype>
class AAALayer : public HiddenLayer<Dtype>, public LearnableLayer<Dtype> {
public: 

	class Builder : public HiddenLayer<Dtype>::Builder {
	public:
		typename Activation<Dtype>::Type _activationType;

		uint32_t _var1;
        double   _var2;
                                        
		Builder() {
			this->type = Layer<Dtype>::AAA;
            _var1 = 0;
            _var2 = 0.0;
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
		virtual Builder* var1(uint32_t var1) {
			this->_var1 = var1;
			return this;
		}
		virtual Builder* var2(double var2) {
			this->_var2 = var2;
			return this;
		}
		Layer<Dtype>* build() {
			return new AAALayer(this);
		}
	};

	/**
	 * @details FullyConnectedLayer 기본 생성자
	 *          내부적으로 레이어 타입만 초기화한다.
	 */
	AAALayer();
	AAALayer(Builder* builder);

    AAALayer(const std::string name, int var1, double var2);
    virtual ~AAALayer();

	//////////////////////////////////////////
	// Learnable Layer Method
	//////////////////////////////////////////
	using HiddenLayer<Dtype>::getName;
	virtual const std::string getName() { return this->name; }
	virtual void update();
	virtual double sumSquareParamsData();
	virtual double sumSquareParamsGrad();
	virtual void scaleParamsGrad(float scale);
	virtual uint32_t boundParams();
	virtual uint32_t numParams();
	virtual void saveParams(std::ofstream& ofs);
	virtual void loadParams(std::ifstream& ifs);
	virtual void loadParams(std::map<std::string, Data<Dtype>*>& dataMap);
	//////////////////////////////////////////

	virtual void backpropagation();
	virtual void reshape();
	virtual void feedforward();

private:
    void initialize(int var1, double var2);

    int         var1;
    double      var2;
};
#endif /* BATCHNORMLAYER_H */
