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
		typename Activation<Dtype>::Type _activationType;	///< weighted sum에 적용할 활성화

        /* batch normalization related variables */
        double      _epsilon;   // Small value added to variance to avoid dividing 
                                // by zero. default value = 0.00001
        bool        _train;
                                        
		Builder() {
			this->type = Layer<Dtype>::BatchNorm;
            _epsilon = 0.00001;
            _train = true;
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
		Builder* epsilon(double epsilon) {
			this->_epsilon = epsilon;
			return this;
		}
        Builder* train(bool train) {
            this->_train = train;
            return this;
        }
		Layer<Dtype>* build() {
			return new BatchNormLayer(this);
		}
	};

	BatchNormLayer(Builder* builder);

    BatchNormLayer(const std::string name, double epsilon, bool train);
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
    void initialize(double epsilon, bool train);

    void applyChanges(LearnableLayer<Dtype> *targetLayer);
    void syncParams(LearnableLayer<Dtype> *targetLayer);

    double      epsilon;                // Small value added to variance to avoid dividing 
                                        // by zero. default value = 0.001
    bool        train;
    int         depth;

    Data<Dtype>    *meanSet;            // mean
    Data<Dtype>    *varSet;             // variance
    Data<Dtype>    *normInputSet;       // normalized input value

    void        computeNormInputGrad();
    void        computeVarianceGrad();
    void        computeMeanGrad();
    void        computeInputGrad();
    void        computeScaleGrad();
    void        computeShiftGrad();
public:
    void        donateParam(BatchNormLayer<Dtype>* receiver);
    void        setTrain(bool train) { this->train = train; }

protected:
    enum ParamType {
        Gamma = 0,
        Beta = 1,
        GlobalMean,
        GlobalVar,
        GlobalCount
    };

	void _updateParam(const uint32_t paramSize, const Dtype regScale, const Dtype learnScale,
        const Dtype epsilon, const Dtype decayRate, const Dtype beta1, const Dtype beta2,
        Data<Dtype>* dataHistory, Data<Dtype>* dataHistory2, Data<Dtype>* data);
};
#endif /* BATCHNORMLAYER_H */
