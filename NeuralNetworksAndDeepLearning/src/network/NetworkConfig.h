/*
 * NetworkParam.h
 *
 *  Created on: 2016. 8. 12.
 *      Author: jhkim
 */

#ifndef NETWORKPARAM_H_
#define NETWORKPARAM_H_

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common.h"
#include "BaseLayer.h"
#include "InputLayer.h"
#include "LearnableLayer.h"
#include "SplitLayer.h"
#include "LossLayer.h"
#include "NetworkListener.h"
#include "Worker.h"
#include "SysLog.h"
#include "Donator.h"


//template <typename Dtype> class DataSet;
template <typename Dtype> class Worker;

#if 0 
template <typename Dtype>
class LayersConfig {
public:
	std::map<std::string, Data<Dtype>*> _layerDataMap;
    std::vector<Layer<Dtype>*> _firstLayers;
    std::vector<Layer<Dtype>*> _lastLayers;
    std::vector<Layer<Dtype>*> _layers;
    std::vector<LearnableLayer<Dtype>*> _learnableLayers;
    std::vector<LossLayer<Dtype>*> _lossLayers;
    InputLayer<Dtype>* _inputLayer;

    std::map<std::string, Layer<Dtype>*> _nameLayerMap;
	Builder* _builder;

	LayersConfig(Builder* builder);
	LayersConfig<Dtype>* firstLayers(std::vector<Layer<Dtype>*>& firstLayers);
	LayersConfig<Dtype>* lastLayers(std::vector<Layer<Dtype>*>& lastLayers);
	LayersConfig<Dtype>* layers(std::vector<Layer<Dtype>*>& layers);
	LayersConfig<Dtype>* learnableLayers(std::vector<LearnableLayer<Dtype>*>& learnableLayers);
	LayersConfig<Dtype>* nameLayerMap(std::map<std::string, Layer<Dtype>*>& nameLayerMap);
	LayersConfig<Dtype>* layerDataMap(std::map<std::string, Data<Dtype>*>& layerDataMap);
};
#endif

#if 0
enum NetworkStatus : int {
	Train = 0,
	Test = 1
};

enum NetworkPhase : int {
	TrainPhase = 0,
	TestPhase = 1
};

enum LRPolicy : int {
	Fixed = 0,
	Step,
	Exp,
	Inv,
	Multistep,
	Poly
};

enum Optimizer : int {
    Momentum = 0,
    Vanilla,
    Nesterov,
    Adagrad,
    RMSprop,
    Adam
};
#endif

template <typename Dtype>
class NetworkConfig {
public:
#if 0
	NetworkStatus _status;
    Optimizer _optimizer;
	LRPolicy _lrPolicy;
	NetworkPhase _phase;
	
    std::vector<NetworkListener*> _networkListeners;
    std::vector<LayersConfig<Dtype>*> layersConfigs;

	uint32_t _batchSize;
	uint32_t _dop;
	uint32_t _epochs;
	uint32_t _testInterval;
	uint32_t _saveInterval;
	uint32_t _iterations;
	uint32_t _stepSize;
	float _baseLearningRate;
    float _power;
	float _momentum;
	float _weightDecay;
	float _clipGradientsLevel;
	float _gamma;
    float _epsilon;     // for Adam, Adagrad, Rmsprop learning policy
    float _beta1;       // for Adam learning policy
    float _beta2;       // for Adam learning policy
    float _decayRate;   // for RMSprop learning policy

    std::string _savePathPrefix;
    std::string _loadPath;

    std::vector<std::string> _lossLayers;

	// save & load를 위해서 builder도 일단 저장해 두자.
	Builder* _builder;

	NetworkConfig(Builder* builder) {
		this->_builder = builder;
		this->_iterations = 0;
		this->_rate = -1.0f;
        this->_optimizer = Optimizer::Momentum;
	}
	NetworkConfig* lossLayers(const std::vector<std::string>& lossLayers) {
		this->_lossLayers = lossLayers;
		return this;
	}
	NetworkConfig* networkListeners(const std::vector<NetworkListener*> networkListeners) {
		this->_networkListeners = networkListeners;
		return this;
	}
	NetworkConfig* batchSize(uint32_t batchSize) {
		this->_batchSize = batchSize;
		return this;
	}
	NetworkConfig* dop(uint32_t dop) {
		this->_dop = dop;
		return this;
	}
	NetworkConfig* epochs(uint32_t epochs) {
		this->_epochs = epochs;
		return this;
	}
	NetworkConfig* testInterval(uint32_t testInterval) {
		this->_testInterval = testInterval;
		return this;
	}
	NetworkConfig* saveInterval(uint32_t saveInterval) {
		this->_saveInterval = saveInterval;
		return this;
	}
	NetworkConfig* stepSize(uint32_t stepSize) {
		this->_stepSize = stepSize;
		return this;
	}
	NetworkConfig* savePathPrefix(std::string savePathPrefix) {
		this->_savePathPrefix = savePathPrefix;
		return this;
	}
	NetworkConfig* loadPath(const std::string loadPath) {
		this->_loadPath = loadPath;
		return this;
	}
	NetworkConfig* clipGradientsLevel(float clipGradientsLevel) {
		this->_clipGradientsLevel = clipGradientsLevel;
		return this;
	}
	NetworkConfig* gamma(float gamma) {
		this->_gamma = gamma;
		return this;
	}
	NetworkConfig* baseLearningRate(float baseLearningRate) {
		this->_baseLearningRate = baseLearningRate;
		return this;
	}
	NetworkConfig* power(float power) {
		this->_power = power;
		return this;
	}
	NetworkConfig* momentum(float momentum) {
		this->_momentum = momentum;
		return this;
	}
	NetworkConfig* weightDecay(float weightDecay) {
		this->_weightDecay = weightDecay;
		return this;
	}
	NetworkConfig* epsilon(float epsilon) {
		this->_epsilon = epsilon;
		return this;
	}
	NetworkConfig* beta(float beta1, float beta2) {
		this->_beta1 = beta1;
		this->_beta2 = beta2;
		return this;
	}
	NetworkConfig* decayRate(float decayRate) {
		this->_decayRate = decayRate;
		return this;
	}
	NetworkConfig* lrPolicy(LRPolicy lrPolicy) {
		this->_lrPolicy = lrPolicy;
		return this;
	}
	NetworkConfig* optimizer(Optimizer opt) {
		this->_optimizer = opt;
		return this;
	}
	NetworkConfig* networkPhase(NetworkPhase phase) {
		this->_phase = phase;
		return this;
	}

	void save();
	void load();
	bool doTest();
	bool doSave();
	float getLearningRate();

    static float calcLearningRate();
#endif

#if 0
private:
	float _rate;
#endif
};


#endif /* NETWORKPARAM_H_ */
