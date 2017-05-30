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
    BatchNormLayer();
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
    void applyChanges(LearnableLayer<Dtype> *targetLayer);
    void syncParams(LearnableLayer<Dtype> *targetLayer);

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
    void        setTrain(bool train);

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
#endif /* BATCHNORMLAYER_H */
