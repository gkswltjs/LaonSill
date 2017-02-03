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

	/**
	 * @details FullyConnectedLayer 기본 생성자
	 *          내부적으로 레이어 타입만 초기화한다.
	 */
	BatchNormLayer();
	BatchNormLayer(Builder* builder);

    BatchNormLayer(const std::string name, int kernelMapCount, double epsilon);
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
    void initialize(int kernelMapCount, double epsilon);

    int         kernelMapCount;
    double      epsilon;                // Small value added to variance to avoid dividing 
                                        // by zero. default value = 0.001
    int         depth;
    int         batchSetCount;

	Data<Dtype>    *gammaSet;           // scale factor
    Data<Dtype>    *betaSet;            // shift factor
    Data<Dtype>    *meanSet;            // mean
    Data<Dtype>    *varSet;             // variance
    Data<Dtype>    *normInputSet;       // normalized input value

    std::shared_ptr<SyncMem<Dtype>>  meanSumSet;    // meanSet들의 합
    std::shared_ptr<SyncMem<Dtype>>  varSumSet;     // varSet들의 합

    void        computeNormInputGrad();
    void        computeVarianceGrad();
    void        computeMeanGrad();
    void        computeInputGrad();
    void        computeScaleGrad();
    void        computeShiftGrad();
};
#endif /* BATCHNORMLAYER_H */
