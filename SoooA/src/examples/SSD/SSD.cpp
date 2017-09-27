/**
 * @file SSD.cpp
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>

#include "common.h"
#include "Debug.h"
#include "SSD.h"
#include "StdOutLog.h"
#include "Network.h"
#include "NetworkMonitor.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"

using namespace std;

#define EXAMPLE_SSD_TRAIN_NETWORK_FILEPATH	SPATH("examples/SSD/ssd_300_train.json")

template<typename Dtype>
void SSD<Dtype>::run() {
    int networkID = PlanParser::loadNetwork(string(EXAMPLE_SSD_TRAIN_NETWORK_FILEPATH));
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
    network->build(10000);

    for (int i = 0; i < 10000; i++) {
        cout << "epoch : " << i << endl;
        network->run(false);
    }
}


template class SSD<float>;
