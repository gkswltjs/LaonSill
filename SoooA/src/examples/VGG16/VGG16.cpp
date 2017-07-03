/**
 * @file VGG16.cpp
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <vector>

#include "common.h"
#include "VGG16.h"
#include "StdOutLog.h"
#include "Network.h"
#include "NetworkMonitor.h"
#include "ImageUtil.h"
#include "PlanParser.h"
#include "PropMgmt.h"

using namespace std;

#define EXAMPLE_VGG16_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/git/soooa/SoooA/src/examples/VGG16/vgg16_train.json")

template<typename Dtype>
void VGG16<Dtype>::run() {
    int networkID = PlanParser::loadNetwork(string(EXAMPLE_VGG16_TRAIN_NETWORK_FILEPATH));
    Network<Dtype>* network = Network<Dtype>::getNetworkFromID(networkID);
    network->build(10000);

	network->run(false);
	//network->reset();
	//cout << "rpn_loss_cls: " << network->findLayer("rpn_loss_cls")->_outputData[0]->host_data()[0] << endl;
	//cout << "rpn_loss_bbox: " << network->findLayer("rpn_loss_bbox")->_outputData[0]->host_data()[0] << endl;
	//cout << "loss_cls: " << network->findLayer("loss_cls")->_outputData[0]->host_data()[0] << endl;
	//cout << "loss_bbox: " << network->findLayer("loss_bbox")->_outputData[0]->host_data()[0] << endl;
}


template class VGG16<float>;
