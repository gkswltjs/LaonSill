#include <cstdint>
#include <iostream>
#include <vector>

#include "cuda/Cuda.h"
#include "dataset/DataSet.h"
#include "dataset/MockDataSet.h"
#include "debug/Debug.h"
#include "evaluation/Evaluation.h"
#include "evaluation/Top1Evaluation.h"
#include "evaluation/Top5Evaluation.h"
#include "monitor/NetworkMonitor.h"
#include "network/Network.h"
#include "network/NetworkConfig.h"
#include "Util.h"

using namespace std;


void network_test();


int main(int argc, char** argv) {
	cout << "main 10000 samples " << endl;
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

	const uint32_t maxEpoch = 10000;
	const uint32_t batchSize = 20;
	const float baseLearningRate = 0.01f;
	const float weightDecay = 0.0002f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;


	cout << "maxEpoch: " << maxEpoch << endl;
	cout << "batchSize: " << batchSize << endl;
	cout << "baseLearningRate: " << baseLearningRate << endl;
	cout << "weightDecay: " << weightDecay << endl;
	cout << "momentum: " << momentum << endl;
	cout << "clipGradientsLevel: " << clipGradientsLevel << endl;

	//SyncMem<float>::setOutstream("./mem");

	//DataSet<float>* dataSet = new MockDataSet<float>(4, 4, 2, 20, 20, 10, MockDataSet<float>::NOTABLE_IMAGE);
	//DataSet<float>* dataSet = new MockDataSet<float>(28, 28, 1, 100, 100, 10);
	//DataSet<float>* dataSet = new MockDataSet<float>(56, 56, 3, 10, 10, 10);
	//DataSet<float>* dataSet = new MnistDataSet<float>(0.8);
	//DataSet<float>* dataSet = new MockDataSet<float>(224, 224, 3, 100, 100, 100);
	//DataSet<float>* dataSet = createImageNet10CatDataSet<float>();
	//DataSet<float>* dataSet = createImageNet100CatDataSet<float>();
	//DataSet<float>* dataSet = createImageNet1000DataSet<float>();
	//DataSet<float>* dataSet = createImageNet10000DataSet<float>();
	DataSet<float>* dataSet = createMnistDataSet<float>();
	dataSet->load();
	//dataSet->zeroMean(true);

	Evaluation<float>* top1Evaluation = new Top1Evaluation<float>();
	Evaluation<float>* top5Evaluation = new Top5Evaluation<float>();
	NetworkListener* networkListener = new NetworkMonitor(NetworkMonitor::PLOT_ONLY);

	LayersConfig<float>* layersConfig = createCNNSimpleLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createCNNDoubleLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createInceptionLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInception3ALayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInception3ALayersConfigTest<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInception3ASimpleLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInception5ALayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInceptionAuxLayersConfig<float>();


	NetworkConfig<float>* networkConfig =
			(new NetworkConfig<float>::Builder())
			->batchSize(batchSize)
			->baseLearningRate(baseLearningRate)
			->weightDecay(weightDecay)
			->momentum(momentum)
			->clipGradientsLevel(clipGradientsLevel)
			->dataSet(dataSet)
			->evaluations({top1Evaluation, top5Evaluation})
			->networkListeners({networkListener})
			->layersConfig(layersConfig)
			->build();


	Network<float>* network = new Network<float>(networkConfig);
	network->shape();
	network->sgd(maxEpoch);

	Cuda::destroy();
}















