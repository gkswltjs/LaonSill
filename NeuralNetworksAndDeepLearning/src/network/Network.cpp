/*
 * Network.cpp
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#include <vector>
#include <map>
#include <cfloat>

#include "DataSet.h"
#include "Layer.h"
#include "SoftmaxWithLossLayer.h"
#include "LossLayer.h"
#include "Util.h"
#include "Worker.h"
#include "Perf.h"
#include "StdOutLog.h"
#include "Network.h"
#include "SysLog.h"
#include "DebugUtil.h"

using namespace std;


template<typename Dtype>
atomic<int>         Network<Dtype>::networkIDGen;
template<typename Dtype>
map<int, Network<Dtype>*> Network<Dtype>::networkIDMap;
template<typename Dtype>
mutex Network<Dtype>::networkIDMapMutex;

template <typename Dtype>
Network<Dtype>::Network() {
    this->networkID = atomic_fetch_add(&Network<Dtype>::networkIDGen, 1);
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    Network<Dtype>::networkIDMap[this->networkID] = this;
}

template<typename Dtype>
void Network<Dtype>::init() {
    atomic_store(&Network<Dtype>::networkIDGen, 0);
}

template<typename Dtype>
Network<Dtype>* Network<Dtype>::getNetworkFromID(int networkID) {
    Network<Dtype>* network;
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    network = Network<Dtype>::networkIDMap[networkID];
    lock.unlock();
    return network;
}

template <typename Dtype>
Network<Dtype>::~Network() {
    unique_lock<mutex> lock(Network<Dtype>::networkIDMapMutex);
    Network<Dtype>::networkIDMap.erase(this->networkID);
}

template <typename Dtype>
void Network<Dtype>::sgd_with_timer(int epochs) {
    struct timespec startTime;
    SPERF_START(NETWORK_TRAINING_TESTTIME, &startTime);
    // XXX: 임시
    //epochs = 1000000;
    //epochs = 1;
	sgd(epochs);

    SPERF_END(NETWORK_TRAINING_TESTTIME, startTime, epochs);
    STDOUT_BLOCK(cout << "Total Training Time : " << SPERF_TIME(NETWORK_TRAINING_TESTTIME)
                    << endl;);
}

template<typename Dtype>
Dtype Network<Dtype>::sgd(int epochs) {
    return 0.0;
}

template <typename Dtype>
void Network<Dtype>::save() {
    // TODO : 반드시 구현 필요
#if 0
	config->save();
#endif
}

template <typename Dtype>
void Network<Dtype>::load() {
    // TODO : 반드시 구현 필요
#if 0
	// load data list from model file
	map<std::string, Data<float>*> dataMap;

    ifstream ifs(config->_loadPath, std::ios::in | std::ios::binary);

    uint32_t numData;
    ifs.read((char*)&numData, sizeof(uint32_t));

    Data<float>::printConfig = true;
    cout << "Load Pretrained Weights ... ----------" << endl;
    for (uint32_t j = 0; j < numData; j++) {
        Data<float>* data = new Data<float>("", true);
        data->load(ifs);

        if (data)
            data->print();

        string dataName;
        dataName = data->_name;

        map<string, Data<float>*>::iterator it;
        it = dataMap.find(dataName);
        if (it != dataMap.end()) {
            cout << dataName << " overwrites ... " << endl;
            delete it->second;
        }

        dataMap[dataName] = data;
        cout << data->_name << " is set to " << dataName << endl;
    }
    cout << "--------------------------------------" << endl;
    Data<float>::printConfig = false;
    ifs.close();

	LayersConfig<Dtype>* layersConfig = getLayersConfig();
	vector<LearnableLayer<Dtype>*> learnableLayers = layersConfig->_learnableLayers;
	const uint32_t numLearnableLayers = learnableLayers.size();

	for (uint32_t i = 0; i < numLearnableLayers; i++) {
		learnableLayers[i]->loadParams(dataMap);
	}

	map<std::string, Data<float>*>::iterator it;
	for (it = dataMap.begin(); it != dataMap.end(); it++)
		delete it->second;
	dataMap.clear();
#endif
}

template <typename Dtype>
Layer<Dtype>* Network<Dtype>::findLayer(const string name) {
    // TODO: 구현 필요
#if 0
	//return getLayersConfig()->_inputLayer->find(0, name);
	map<string, Layer<Dtype>*>& nameLayerMap = getLayersConfig()->_nameLayerMap;
	typename map<string, Layer<Dtype>*>::iterator it = nameLayerMap.find(name);
	if(it != nameLayerMap.end()) {
		return it->second;
	} else {
		return 0;
	}
#endif
}

template class Network<float>;
