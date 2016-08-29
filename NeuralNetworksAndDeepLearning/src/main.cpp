#include <CImg.h>
#include <stddef.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "cuda/Cuda.h"
#include "dataset/DataSet.h"
#include "dataset/MockDataSet.h"
#include "debug/Debug.h"
#include "evaluation/Top1Evaluation.h"
#include "evaluation/Top5Evaluation.h"
#include "monitor/NetworkMonitor.h"
#include "network/Network.h"
#include "Util.h"


#include "layer/Layer.h"

using namespace std;


void network_test();


int main(int argc, char** argv) {
	cout << "main" << endl;
	cout.precision(11);
	cout.setf(ios::fixed);
	Util::setOutstream(&cout);
	//Util::setOutstream("./log");

	network_test();

	cout << "end" << endl;
	return 0;
}


void network_test() {
	Cuda::create(0);
	cout << "Cuda creation done ... " << endl;
	Util::setPrint(false);

	const uint32_t maxEpoch = 1000;
	const uint32_t batchSize = 50;
	const float baseLearningRate = 0.01f;
	const float weightDecay = 0.0002f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 500.0f;

	//DataSet* dataSet = new MockDataSet(28, 28, 1, 10, 10, 10);
	//DataSet* dataSet = new MockDataSet(56, 56, 3, 10, 10, 10);
	//DataSet* dataSet = new MnistDataSet(0.8);
	//DataSet* dataSet = new MockDataSet(224, 224, 3, 100, 100, 100);
	//DataSet* dataSet = createImageNet10CatDataSet();
	//DataSet* dataSet = createImageNet100CatDataSet();
	DataSet<float>* dataSet = createMnistDataSet<float>();
	dataSet->load();
	dataSet->zeroMean(true);

	Evaluation* top1Evaluation = new Top1Evaluation();
	Evaluation* top5Evaluation = new Top5Evaluation();
	NetworkListener* top1Listener = new NetworkMonitor(maxEpoch);
	NetworkListener* top5Listener = new NetworkMonitor(maxEpoch);

	//LayersConfig* layersConfig = createCNNSimpleLayersConfig();
	LayersConfig<float>* layersConfig = createCNNDoubleLayersConfig<float>();
	//LayersConfig* layersConfig = createGoogLeNetLayersConfig();
	//LayersConfig* layersConfig = createInceptionLayersConfig();
	//LayersConfig* layersConfig = createGoogLeNetInception3ALayersConfig();
	//LayersConfig* layersConfig = createGoogLeNetInception3ASimpleLayersConfig();

	NetworkConfig<float>* networkConfig =
			(new NetworkConfig<float>::Builder())
			->batchSize(batchSize)
			->baseLearningRate(baseLearningRate)
			->weightDecay(weightDecay)
			->momentum(momentum)
			->clipGradientsLevel(clipGradientsLevel)
			->dataSet(dataSet)
			->evaluations({top1Evaluation, top5Evaluation})
			->networkListeners({top1Listener, top5Listener})
			->layersConfig(layersConfig)
			->build();

	Network<float>* network = new Network<float>(networkConfig);
	network->shape();
	network->sgd(maxEpoch);

	Cuda::destroy();
}















