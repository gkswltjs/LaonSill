/**
 * @file NoiseInputLayer.cpp
 * @date 2017-02-16
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <time.h>

#include <boost/random.hpp>
#include <boost/math/special_functions/next.hpp>

#include <cblas.h>

#include "common.h"
#include "NoiseInputLayer.h"
#include "InputLayer.h"
#include "NetworkConfig.h"
#include "SysLog.h"
#include "PropMgmt.h"

using namespace std;

typedef boost::mt19937 RNGType;

template<typename Dtype>
NoiseInputLayer<Dtype>::NoiseInputLayer() {
    initialize(0, 0.0, 0.0, false, 0, 0, 0, 0.0, 0.0, false);
}

template<typename Dtype>
NoiseInputLayer<Dtype>::NoiseInputLayer(const string name, int noiseDepth,
    double noiseRangeLow, double noiseRangeHigh, bool useLinearTrans, int tranChannels,
    int tranRows, int tranCols, double tranMean, double tranVariance, bool regenerateNoise) :
    InputLayer<Dtype>(name) {
    initialize(noiseDepth, noiseRangeLow, noiseRangeHigh, useLinearTrans, tranChannels,
        tranRows, tranCols, tranMean, tranVariance, regenerateNoise);
}

template<typename Dtype>
NoiseInputLayer<Dtype>::NoiseInputLayer(const string& name) : InputLayer<Dtype>(name) {

}

template<typename Dtype>
NoiseInputLayer<Dtype>::NoiseInputLayer(Builder* builder) : InputLayer<Dtype>(builder) {
	initialize(builder->_noiseDepth, builder->_noiseRangeLow, builder->_noiseRangeHigh,
        builder->_useLinearTrans, builder->_tranChannels, builder->_tranRows,
        builder->_tranCols, builder->_tranMean, builder->_tranVariance,
        builder->_regenerateNoise);
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::setRegenerateNoise(bool regenerate) {
    this->regenerateNoise = regenerate;
}

template<typename Dtype>
NoiseInputLayer<Dtype>::~NoiseInputLayer() {
    if (this->uniformArray != NULL) {
        free(this->uniformArray);
    }
}

template <typename Dtype>
bool NoiseInputLayer<Dtype>::prepareUniformArray() {
    uint32_t batchSize = this->networkConfig->_batchSize;
	RNGType rng;
    unsigned int seedValue = static_cast<unsigned int>(time(NULL)+getpid());
    rng.seed(seedValue);

    bool firstGenerate = false;

    if (this->uniformArray == NULL) {
        int allocSize = sizeof(Dtype) * this->noiseDepth * batchSize;
        this->uniformArray = (Dtype*)malloc(allocSize);
        SASSERT0(this->uniformArray != NULL);
        firstGenerate = true;
    }

    if (firstGenerate || this->regenerateNoise) {
        boost::random::uniform_real_distribution<float> random_distribution(
            this->noiseRangeLow, this->noiseRangeHigh);
        boost::variate_generator<RNGType, boost::random::uniform_real_distribution<float>>
            variate_generator(rng, random_distribution);

        for (int i = 0; i < this->noiseDepth * batchSize; ++i) {
            this->uniformArray[i] = (Dtype)variate_generator();
        }

        return true;
    }

    return false;
}

template <typename Dtype>
void NoiseInputLayer<Dtype>::prepareLinearTranMatrix() {
	/*
    uint32_t batchSize = this->networkConfig->_batchSize;

	RNGType rng;
    rng.seed(static_cast<unsigned int>(time(NULL)+getpid()));
    boost::normal_distribution<float> random_distribution(this->tranMean, this->tranVariance);
    boost::variate_generator<RNGType, boost::normal_distribution<float> >
    variate_generator(rng, random_distribution);

    Dtype* tempMatrix;
    int tempMatrixCount = this->noiseDepth * this->tranChannels * this->tranRows *
        this->tranCols * batchSize;

    int tempMatrixAllocSize = sizeof(Dtype) * tempMatrixCount;
    tempMatrix = (Dtype*)malloc(tempMatrixAllocSize);
    SASSERT0(tempMatrix != NULL);

    for (int i = 0; i < tempMatrixCount; i++) {
        tempMatrix[i] = (Dtype)variate_generator();
    }

    int linearTransMatrixAllocSize = sizeof(Dtype) * this->tranChannels * this->tranRows *
        this->tranCols * batchSize;
    SASSERT0(this->linearTransMatrix == NULL);
    this->linearTransMatrix = (Dtype*)malloc(linearTransMatrixAllocSize);
    SASSERT0(this->linearTransMatrix != NULL);

    int m = 1;
    int n = this->tranChannels * this->tranRows * this->tranCols * batchSize;
    int k = this->noiseDepth * batchSize;

    if (sizeof(Dtype) == sizeof(float)) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1,
            this->uniformArray, k, tempMatrix, n, 0, this->linearTransMatrix, n);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1,
            (const double*)this->uniformArray, k, (const double*)tempMatrix, n, 0,
            (double*)this->linearTransMatrix, n);
    }

    free(tempMatrix);
    */
}

template <typename Dtype>
void NoiseInputLayer<Dtype>::reshape() {
    uint32_t batchSize = this->networkConfig->_batchSize;

    bool isNoiseGenerated = prepareUniformArray();

    if ((this->uniformArray == NULL) && (this->useLinearTrans)) {
        prepareLinearTranMatrix();
    }

	if (this->_inputData.size() < 1) {
		for (uint32_t i = 0; i < this->_outputs.size(); i++) {
			this->_inputs.push_back(this->_outputs[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}

    Layer<Dtype>::_adjustInputShape();

    this->batchSize = batchSize;
    if (!this->useLinearTrans) {
        this->_inputShape[0][0] = batchSize;
        this->_inputShape[0][1] = 1;
        this->_inputShape[0][2] = (unsigned int)this->noiseDepth;
        this->_inputShape[0][3] = 1;

        this->_inputData[0]->reshape(this->_inputShape[0]);
    } else {
        this->_inputShape[0][0] = batchSize;
        this->_inputShape[0][1] = (unsigned int)this->tranChannels;
        this->_inputShape[0][2] = (unsigned int)this->tranRows;
        this->_inputShape[0][3] = (unsigned int)this->tranCols;

        this->_inputData[0]->reshape(this->_inputShape[0]);
    }

    if (isNoiseGenerated) {
        int copyElemCount;
        if (this->useLinearTrans) {
            copyElemCount = this->tranChannels * this->tranRows * this->tranCols * batchSize;
            this->_inputData[0]->set_device_with_host_data(this->linearTransMatrix,
                0, copyElemCount); 
        } else {
            copyElemCount = this->noiseDepth * batchSize;
            this->_inputData[0]->set_device_with_host_data(this->uniformArray,
                0, copyElemCount); 
        }
    }
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::feedforward() {
	//Layer<Dtype>::feedforward();
	cout << "unsupported ... " << endl;
	exit(1);
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
    reshape();

}

template<typename Dtype>
void NoiseInputLayer<Dtype>::initialize(int noiseDepth, double noiseRangeLow,
    double noiseRangeHigh, bool useLinearTrans, int tranChannels, int tranRows,
    int tranCols, double tranMean, double tranVariance, bool regenerateNoise) {

    this->type = Layer<Dtype>::NoiseInput;
    this->batchSize = 0;
    this->uniformArray = NULL;
    this->linearTransMatrix = NULL;

    this->noiseDepth = noiseDepth;
    this->noiseRangeLow = noiseRangeLow;
    this->noiseRangeHigh = noiseRangeHigh;
    this->useLinearTrans = useLinearTrans;
    this->tranChannels = tranChannels;
    this->tranRows = tranRows;
    this->tranCols = tranCols;
    this->tranMean = tranMean;
    this->tranVariance = tranVariance;

    this->regenerateNoise = regenerateNoise;
}

template<typename Dtype>
int NoiseInputLayer<Dtype>::getNumTrainData() {
    if (this->_dataSet != NULL) {
        return this->_dataSet->getNumTrainData();
    } else {    
        uint32_t batches = this->networkConfig->_batchSize;
        return batches;
    }
}

template<typename Dtype>
int NoiseInputLayer<Dtype>::getNumTestData() {
    if (this->_dataSet != NULL) {
        return this->_dataSet->getNumTestData();
    } else {
        return 1;
    }
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::shuffleTrainDataSet() {
    if (this->_dataSet != NULL) {
        return this->_dataSet->shuffleTrainDataSet();
    }
}

/****************************************************************************
 * layer callback functions 
 ****************************************************************************/
template<typename Dtype>
void* NoiseInputLayer<Dtype>::initLayer() {
    NoiseInputLayer* layer = new NoiseInputLayer<Dtype>(SLPROP_BASE(name));
    return (void*)layer;
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::destroyLayer(void* instancePtr) {
    NoiseInputLayer<Dtype>* layer = (NoiseInputLayer<Dtype>*)instancePtr;
    delete layer;
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::setInOutTensor(void* instancePtr, void* tensorPtr,
    bool isInput, int index) {
    NoiseInputLayer<Dtype>* layer = (NoiseInputLayer<Dtype>*)instancePtr;

    SASSERT0(!isInput);
    SASSERT0(index == 0);
    SASSERT0(layer->_outputData.size() == 0);
    layer->_outputData.push_back((Data<Dtype>*)tensorPtr);
}

template<typename Dtype>
bool NoiseInputLayer<Dtype>::allocLayerTensors(void* instancePtr) {
    NoiseInputLayer<Dtype>* layer = (NoiseInputLayer<Dtype>*)instancePtr;
    //layer->reshape();
    return true;
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::forwardTensor(void* instancePtr, int miniBatchIdx) {
    cout << "NoiseInputLayer.. forward(). miniBatchIndex : " << miniBatchIdx << endl;
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::backwardTensor(void* instancePtr) {
    cout << "NoiseInputLayer.. backward()" << endl;
}

template<typename Dtype>
void NoiseInputLayer<Dtype>::learnTensor(void* instancePtr) {
    cout << "NoiseInputLayer.. learn()" << endl;
}

template class NoiseInputLayer<float>;
