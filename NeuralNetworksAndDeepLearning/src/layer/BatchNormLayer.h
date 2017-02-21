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
#include "Layer.h"
#include "LearnableLayer.h"
#include "LayerConfig.h"
#include "Activation.h"
#include "ActivationFactory.h"
#include "Cost.h"

template <typename Dtype>
class BatchNormLayer : public LearnableLayer<Dtype> {
public: 

	class Builder : public LearnableLayer<Dtype>::Builder {
	public:
		uint32_t    _kernelMapCount;
		typename Activation<Dtype>::Type _activationType;	///< weighted sum에 적용할 활성화

        /* batch normalization related variables */
        double      _epsilon;   // Small value added to variance to avoid dividing 
                                // by zero. default value = 0.001
                                        
		Builder() {
			this->type = Layer<Dtype>::BatchNorm;
            _kernelMapCount = 1;
            _epsilon = 0.001;
		}
		virtual Builder* name(const std::string name) {
			LearnableLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			LearnableLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			LearnableLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			LearnableLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			LearnableLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		Builder* kernelMapCount(uint32_t kernelMapCount) {
			this->_kernelMapCount = kernelMapCount;
			return this;
		}
		Builder* epsilon(double epsilon) {
			this->_epsilon = epsilon;
			return this;
		}
		Layer<Dtype>* build() {
			return new BatchNormLayer(this);
		}
	};

	BatchNormLayer(Builder* builder);

    BatchNormLayer(const std::string name, int kernelMapCount, double epsilon);
    virtual ~BatchNormLayer();

	//////////////////////////////////////////
	// Learnable Layer Method
	//////////////////////////////////////////
	using Layer<Dtype>::getName;
	virtual const std::string getName() { return this->name; }
	virtual void update();
	//virtual double sumSquareParamsData();
	//virtual double sumSquareParamsGrad();
	//virtual void scaleParamsGrad(float scale);
	//virtual uint32_t boundParams();
	//virtual uint32_t numParams();
	//virtual void saveParams(std::ofstream& ofs);
	//virtual void loadParams(std::ifstream& ifs);
	//virtual void loadParams(std::map<std::string, Data<Dtype>*>& dataMap);
	//////////////////////////////////////////

	virtual void backpropagation();
	virtual void reshape();
	virtual void feedforward();

private:
    void initialize(int kernelMapCount, double epsilon);

    int         kernelMapCount;
    double      epsilon;                // Small value added to variance to avoid dividing 
                                        // by zero. default value = 0.001
    int         depth;
    int         batchSetCount;
    Dtype      *gammaSets;              // scaled normalized value sets
    Dtype      *betaSets;               // shift normalized value sets
    Dtype      *meanSumSets;            // summed mean value sets
    Dtype      *varianceSumSets;        // summed variance value sets

    Dtype      *localMeanSets;          // mean sets for each mini-batch
    Dtype      *localVarianceSets;      // variance sets for each mini-batch

    Dtype      *normInputValues;        // normalized input values. 
                                        // backward 과정에서 gamma 학습에 활용이 됨.
                                        
    Dtype      *normInputGradValues;    // gradient of normalized input
    Dtype      *varianceGradValues;     // gradient of variance
    Dtype      *meanGradValues;         // gradient of mean
    Dtype      *gammaGradValues;        // gradient of scaled normalized value
    Dtype      *betaGradValues;         // gradient of scaled normalized value

    void        computeNormInputGrad();
    void        computeVarianceGrad();
    void        computeMeanGrad();
    void        computeInputGrad();
    void        computeScaleGrad();
    void        computeShiftGrad();
};
#endif /* BATCHNORMLAYER_H */
