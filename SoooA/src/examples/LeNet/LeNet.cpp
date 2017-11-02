/**
 * @file LeNet.cpp
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>

#include "common.h"
#include "Debug.h"
#include "LeNet.h"
#include "StdOutLog.h"
#include "Network.h"
#include "NetworkMonitor.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"

using namespace std;

#define EXAMPLE_LENET_TRAIN_NETWORK_FILEPATH	SPATH("examples/LeNet/lenet_train.json")

template<typename Dtype>
void LeNet<Dtype>::run() {
    int networkID = PlanParser::loadNetwork(string(EXAMPLE_LENET_TRAIN_NETWORK_FILEPATH));
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
    network->build(1000);
    network->run(false);
#if 0
    for (int i = 0; i < 100; i++) {
        cout << "epoch : " << i << endl;
        network->run(false);
        network->reset();
    }
#endif
}


template class LeNet<float>;
