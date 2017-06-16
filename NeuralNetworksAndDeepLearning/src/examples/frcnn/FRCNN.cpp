/**
 * @file FRCNN.cpp
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>

#include "common.h"
#include "Debug.h"
#include "FRCNN.h"
#include "StdOutLog.h"
#include "Network.h"
#include "NetworkMonitor.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"

using namespace std;

template<typename Dtype>
void FRCNN<Dtype>::setLayerTrain(Network<Dtype>* network, bool train) {
    vector<Layer<Dtype>*> layers = network->findLayersByType((int)Layer<Dtype>::BatchNorm);

    for (int i = 0; i < layers.size(); i++) {
        BatchNormLayer<Dtype>* bnLayer = dynamic_cast<BatchNormLayer<Dtype>*>(layers[i]);
        SASSUME0(bnLayer != NULL);

        bnLayer->setTrain(train);
    }
}

#define EXAMPLE_FRCNN_TRAIN_NETWORK_FILEPATH	("../src/examples/frcnn/frcnn_train.json")

template<typename Dtype>
void FRCNN<Dtype>::run() {
    int networkID = PlanParser::loadNetwork(string(EXAMPLE_FRCNN_TRAIN_NETWORK_FILEPATH));
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
    network->build(1);

    for (int i = 0; i < 10000; i++) {
        cout << "epoch : " << i << endl;
        network->run(false);
    }
}


template class FRCNN<float>;
