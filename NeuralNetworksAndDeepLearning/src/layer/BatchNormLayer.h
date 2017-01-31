/**
 * @file BatchNormLayer.h
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
class BatchNormLayer : public HiddenLayer<Dtype>, public LearnableLayer<Dtype> {
public: 

	class Builder : public HiddenLayer<Dtype>::Builder {
	public:
		uint32_t _nOut;										///< 출력 노드의 수
		typename Activation<Dtype>::Type _activationType;	///< weighted sum에 적용할 활성화

        /* batch normalization related variables */
        double      _epsilon;   // Small value added to variance to avoid dividing 
                                // by zero. default value = 0.001
                                        
		Builder() {
			this->type = Layer<Dtype>::BatchNorm;
			_nOut = 0;
            _epsilon = 0.001;
		}
		Builder* nOut(uint32_t nOut) {
			this->_nOut = nOut;
			return this;
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
		virtual Builder* epsilon(double epsilon) {
			this->_epsilon = epsilon;
			return this;
		}
		Layer<Dtype>* build() {
			return new BatchNormLayer(this);
		}
	};

	/**
	 * @details FullyConnectedLayer 기본 생성자
	 *          내부적으로 레이어 타입만 초기화한다.
	 */
	BatchNormLayer();
	BatchNormLayer(Builder* builder);

    BatchNormLayer(const std::string name, int n_out, double epsilon);
    virtual ~BatchNormLayer();

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
    void initialize(int n_out, double epsilon);

    int         n_out;
    double      epsilon;          // Small value added to variance to avoid dividing 
                                    // by zero. default value = 0.001
    Dtype      *gammaSets;        // scaled normalized value sets
    Dtype      *betaSets;         // shift normalized value sets
    Dtype      *meanSets;         // moving mean value sets
    Dtype      *varianceSets;     // moving variance value sets
};
#endif /* BATCHNORMLAYER_H */
