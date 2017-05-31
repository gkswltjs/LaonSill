#define NETWORK_TEST 0
#if NETWORK_TEST


#include <cstdlib>
#include <fstream>
#include <map>

#include "Network.h"
#include "Util.h"
#include "Cuda.h"
#include "Debug.h"



using namespace std;



Network<float>* network;

void setup();
void cleanup();

void network_save_test();
void loadPretrainedWeights_test();
void load_proposal_target_layer_test();



int main() {
	//setup();

	//network_save_test();
	//loadPretrainedWeights_test();
	load_proposal_target_layer_test();
	//cleanup();
}



void setup() {
	LayersConfig<float>* layersConfig = createCNNSimpleLayersConfig<float>();

	const string loadPath = "/home/jkim/Dev/SOOOA_HOME/network/network.param";
	const string savePathPrefix = "/home/jkim/Dev/SOOOA_HOME/network";
	Worker<float>::consumerCount = 1;
	NetworkConfig<float>* config =
			(new NetworkConfig<float>::Builder())
			->savePathPrefix(savePathPrefix)
			->loadPath(weightsPath)
			->build();

	config->layersConfigs[0] = layersConfig;

	network = new Network<float>(config);
	network->setLayersConfig(layersConfig);
}

void cleanup() {
	delete network;
}

void network_save_test() {
	network->save();
}

void loadPretrainedWeights_test() {
	network->loadPretrainedWeights();
}


/*
//"rois", "labels", "bbox_targets", "bbox_inside_weights", "bbox_outside_weights"
template <typename Dtype>
struct ptl {
	Data<Dtype>* rois;
	Data<Dtype>* labels;
	Data<Dtype>* bboxTargets;
	Data<Dtype>* bboxInsideWeights;
	Data<Dtype>* bboxOutsideWeights;
};
*/

void load_proposal_target_layer_test() {
	const string path = "/home/jkim/Dev/SOOOA_HOME/network/proposal_target_layer.ptl";
	ifstream ifs(path, std::ios::in | std::ios::binary);

	uint32_t numData;
	ifs.read((char*)&numData, sizeof(uint32_t));

	cout << "numData: " << numData << endl;
	numData /= 5;
	vector<vector<Data<float>*>> dataList(numData);

	Data<float>::printConfig = true;
	for (uint32_t i = 0; i < numData; i++) {
		dataList[i].resize(5);
		for (uint32_t j = 0; j < 5; j++) {
			Data<float>* data = new Data<float>("", true);
			data->load(ifs);
			dataList[i][j] = data;
			dataList[i][j]->print_data({}, false);
		}
	}
	Data<float>::printConfig = false;
	ifs.close();
}

#endif
