/**
 * @file YOLO.cpp
 * @date 2017-12-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>
#include <string>

#include "common.h"
#include "Debug.h"
#include "YOLO.h"
#include "StdOutLog.h"
#include "Network.h"
#include "NetworkMonitor.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"

using namespace std;

//#define YOLO_INFERENCE 1

template<typename Dtype>
void YOLO<Dtype>::setLayerTrain(Network<Dtype>* network, bool train) {
    vector<Layer<Dtype>*> layers = network->findLayersByType((int)Layer<Dtype>::BatchNorm);

    for (int i = 0; i < layers.size(); i++) {
        BatchNormLayer<Dtype>* bnLayer = dynamic_cast<BatchNormLayer<Dtype>*>(layers[i]);
        SASSUME0(bnLayer != NULL);

        bnLayer->setTrain(train);
    }
}

#if !YOLO_INFERENCE
#define EXAMPLE_YOLO_NETWORK_FILEPATH	    SPATH("examples/YOLO/yolo_train.json")
#else
#define EXAMPLE_YOLO_NETWORK_FILEPATH	    SPATH("examples/YOLO/yolo_test.json")
#endif

template<typename Dtype>
void YOLO<Dtype>::run() {
    string networkID = PlanParser::loadNetwork(string(EXAMPLE_YOLO_NETWORK_FILEPATH));
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);

#if !INFERENCE
    network->build(4);
    network->run(false);
#else
    network->build(1);
    network->run(true);
#endif

}


template class YOLO<float>;
