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
#include "Evaluation.h"
#include "Layer.h"
#include "InputLayer.h"
#include "LearnableLayer.h"
#include "SplitLayer.h"
#include "LossLayer.h"
#include "AccuracyLayer.h"
#include "NetworkListener.h"
#include "Worker.h"
#include "Top1Evaluation.h"
#include "Top5Evaluation.h"
#include "SysLog.h"
#include "Donator.h"


//template <typename Dtype> class DataSet;
template <typename Dtype> class Worker;



struct WeightsArg {
	std::string weightsPath;
	//std::vector<std::string> weights;
	std::map<std::string, std::string> weightsMap;
};



template <typename Dtype>
class LayersConfig {
public:
	class Builder {
	public:
		// layer의 등록 순서를 보장하기 위해 vector를 사용해야 함.
		std::vector<typename Layer<Dtype>::Builder*> _layerWise;
		std::set<uint32_t> _layerIdSet;

		Builder* layer(typename Layer<Dtype>::Builder* layerBuilder);
		LayersConfig<Dtype>* build();

	private:
		void initializeLayers(std::vector<Layer<Dtype>*>& firstLayers,
				std::vector<Layer<Dtype>*>& lastLayers,
				std::vector<AccuracyLayer<Dtype>*>& accuracyLayers,
        		std::vector<Layer<Dtype>*>& layers,
        		std::vector<LearnableLayer<Dtype>*>& learnableLayers,
        		std::map<uint32_t, Layer<Dtype>*>& idLayerMap);

		void buildNameLayerMap(std::vector<Layer<Dtype>*>& layers,
				std::map<std::string, Layer<Dtype>*>& nameLayerMap);

		void buildLayerData(std::vector<Layer<Dtype>*>& layers,
				std::map<std::string, Data<Dtype>*>& layerDataMap);

		void printFinalLayerConfiguration(std::vector<Layer<Dtype>*>& olayers);

		void insertSplitLayers(std::vector<Layer<Dtype>*>& layers);

		const std::string getSplitLayerName(const std::string& layerName,
				const std::string& dataName, const int dataIdx);

		const std::string getSplitDataName(const std::string& layerName,
				const std::string& dataName, const int dataIdx, const int splitIdx);

		void _updateLayerData(std::map<std::string, Data<Dtype>*>& dataMap,
				std::vector<std::string>& dataNameVec,
                std::vector<Data<Dtype>*>& layerDataVec);

		void _orderLayers(std::vector<Layer<Dtype>*>& tempLayers,
            std::vector<Layer<Dtype>*>& layers);

		bool _isSetContainsAll(std::set<std::string>& dataSet,
            std::vector<std::string>& inputs);
	};

	std::map<std::string, Data<Dtype>*> _layerDataMap;
    std::vector<Layer<Dtype>*> _firstLayers;
    std::vector<Layer<Dtype>*> _lastLayers;
    std::vector<AccuracyLayer<Dtype>*> _accuracyLayers;
    std::vector<Layer<Dtype>*> _layers;
    std::vector<LearnableLayer<Dtype>*> _learnableLayers;
    std::vector<LossLayer<Dtype>*> _lossLayers;
    InputLayer<Dtype>* _inputLayer;

    std::map<std::string, Layer<Dtype>*> _nameLayerMap;
    std::map<std::string, uint32_t> _nameLayerIdxMap;
	Builder* _builder;

	LayersConfig(Builder* builder);
	LayersConfig<Dtype>* firstLayers(std::vector<Layer<Dtype>*>& firstLayers);
	LayersConfig<Dtype>* lastLayers(std::vector<Layer<Dtype>*>& lastLayers);
	LayersConfig<Dtype>* accuracyLayers(std::vector<AccuracyLayer<Dtype>*>& accuracyLayers);
	LayersConfig<Dtype>* layers(std::vector<Layer<Dtype>*>& layers);
	LayersConfig<Dtype>* learnableLayers(std::vector<LearnableLayer<Dtype>*>& learnableLayers);
	LayersConfig<Dtype>* nameLayerMap(std::map<std::string, Layer<Dtype>*>& nameLayerMap);
	LayersConfig<Dtype>* layerDataMap(std::map<std::string, Data<Dtype>*>& layerDataMap);
	LayersConfig<Dtype>* nameLayerIdxMap(std::map<std::string, uint32_t>& nameLayerIdxMap);
};


enum NetworkStatus {
	Train = 0,
	Test = 1
};

enum NetworkPhase {
	TrainPhase = 0,
	TestPhase = 1
};

enum LRPolicy {
	Fixed = 0,
	Step,
	Exp,
	Inv,
	Multistep,
	Poly
};

enum Optimizer {
    Momentum = 0,
    Vanilla,
    Nesterov,
    Adagrad,
    RMSprop,
    Adam
};


template <typename Dtype>
class NetworkConfig {
public:

	class Builder {
	public:
		//DataSet<Dtype>* _dataSet;
        //std::vector<Evaluation<Dtype>*> _evaluations;
        std::vector<NetworkListener*> _networkListeners;
        std::vector<LayersConfig<Dtype>*> layersConfigs;

		uint32_t _batchSize;
		uint32_t _dop;	/* degree of parallel */
		uint32_t _epochs;
		uint32_t _testInterval;
		uint32_t _saveInterval;
		uint32_t _stepSize;					// update _baseLearningRate
		float _baseLearningRate;
		float _momentum;
		float _weightDecay;
		float _clipGradientsLevel;
		float _gamma;
        float _epsilon;     // for Adam, Adagrad, Rmsprop learning policy
        float _beta1;       // for Adam learning policy
        float _beta2;       // for Adam learning policy
        float _decayRate;   // for RMSprop learning policy

        std::string _savePathPrefix;
        std::string _weightsPath;
        std::vector<WeightsArg> _weightsArgs;

        Optimizer _optimizer;   // optimizer
		LRPolicy _lrPolicy;     // learning rate policy
		NetworkPhase _phase;

		std::vector<std::string> _lossLayers;
		std::vector<std::string> _accuracyLayers;

		Builder() {
			//this->_dataSet = NULL;
			this->_batchSize = 1;
			this->_dop = 1;
			this->_epochs = 1;
			this->_clipGradientsLevel = 35.0f;
			this->_phase = NetworkPhase::TrainPhase;
            this->_optimizer = Optimizer::Momentum;
            this->_epsilon = 0.000000001;
            this->_beta1 = 0.9;
            this->_beta2 = 0.999;
            this->_decayRate = 0.9;
		}
		Builder* networkListeners(const std::vector<NetworkListener*> networkListeners) {
			this->_networkListeners = networkListeners;
			return this;
		}
		Builder* batchSize(uint32_t batchSize) {
			this->_batchSize = batchSize;
			return this;
		}
		Builder* dop(uint32_t dop) {
			this->_dop = dop;
			return this;
		}
		Builder* epochs(uint32_t epochs) {
			this->_epochs = epochs;
			return this;
		}
		Builder* testInterval(uint32_t testInterval) {
			this->_testInterval = testInterval;
			return this;
		}
		Builder* saveInterval(uint32_t saveInterval) {
			this->_saveInterval = saveInterval;
			return this;
		}
		Builder* stepSize(uint32_t stepSize) {
			this->_stepSize = stepSize;
			return this;
		}
		Builder* savePathPrefix(std::string savePathPrefix) {
			this->_savePathPrefix = savePathPrefix;
			return this;
		}
		Builder* weightsPath(std::string weightsPath) {
			this->_weightsPath = weightsPath;
			return this;
		}
		Builder* weightsArgs(std::vector<WeightsArg> weightsArgs) {
			this->_weightsArgs = weightsArgs;
			return this;
		}
		Builder* clipGradientsLevel(float clipGradientsLevel) {
			this->_clipGradientsLevel = clipGradientsLevel;
			return this;
		}
		Builder* gamma(float gamma) {
			this->_gamma = gamma;
			return this;
		}
        Builder* epsilon(float epsilon) {
            this->_epsilon = epsilon;
            return this;
        }
        Builder* beta(float beta1, float beta2) {
            this->_beta1 = beta1;
            this->_beta2 = beta2;
            return this;
        }
        Builder* decayRate(float decayRate) {
            this->_decayRate = decayRate;
            return this;
        }
		Builder* baseLearningRate(float baseLearningRate) {
			this->_baseLearningRate = baseLearningRate;
			return this;
		}
		Builder* momentum(float momentum) {
			this->_momentum = momentum;
			return this;
		}
		Builder* weightDecay(float weightDecay) {
			this->_weightDecay = weightDecay;
			return this;
		}
		Builder* lrPolicy(LRPolicy lrPolicy) {
			this->_lrPolicy = lrPolicy;
			return this;
		}
		Builder* optimizer(Optimizer opt) {
			this->_optimizer = opt;
			return this;
		}
		Builder* networkPhase(NetworkPhase phase) {
			this->_phase = phase;
			return this;
		}
		Builder* lossLayers(const std::vector<std::string>& lossLayers) {
			this->_lossLayers = lossLayers;
			return this;
		}
		Builder* accuracyLayers(const std::vector<std::string>& accuracyLayers) {
			this->_accuracyLayers = accuracyLayers;
			return this;
		}
		NetworkConfig<Dtype>* build();
		void print();
	};

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
	float _momentum;
	float _weightDecay;
	float _clipGradientsLevel;
	float _gamma;
    float _epsilon;     // for Adam, Adagrad, Rmsprop learning policy
    float _beta1;       // for Adam learning policy
    float _beta2;       // for Adam learning policy
    float _decayRate;   // for RMSprop learning policy

    std::string _savePathPrefix;
    std::string _weightsPath;
    std::vector<WeightsArg> _weightsArgs;

    std::vector<std::string> _lossLayers;
    std::vector<std::string> _accuracyLayers;

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
	NetworkConfig* accuracyLayers(const std::vector<std::string>& accuracyLayers) {
		this->_accuracyLayers = accuracyLayers;
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
	NetworkConfig* weightsPath(const std::string weightsPath) {
		this->_weightsPath = weightsPath;
		return this;
	}
	NetworkConfig* weightsArgs(const std::vector<WeightsArg>& weightsArgs) {
		this->_weightsArgs = weightsArgs;
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

private:
	float _rate;
};



#endif /* NETWORKPARAM_H_ */
