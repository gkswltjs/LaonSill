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
#include "application/ArtisticStyle.h"

using namespace std;


void network_test();
void network_load();
void artistic_style();


int main(int argc, char** argv) {
	cout << "main 10000 samples " << endl;
	cout.precision(11);
	cout.setf(ios::fixed);
	Util::setOutstream(&cout);
	//Util::setOutstream("./log");

	//network_test();
	//network_load();
	artistic_style();

	cout << "end" << endl;
	return 0;
}


void network_test() {
	Cuda::create(0);
	cout << "Cuda creation done ... " << endl;
	Util::setPrint(false);

	const uint32_t maxEpoch = 10000;
	const uint32_t batchSize = 20;
	const uint32_t testInterval = 500;			// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 5000;			// 1000000 / batchSize
	const uint32_t stepSize = 100000;
	const float baseLearningRate = 0.001f;
	const float weightDecay = 0.00001f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
	const float gamma = 0.1;
	const LRPolicy lrPolicy = LRPolicy::Step;
	const string savePathPrefix = "/home/jhkim/network_save/current/network";

	//SyncMem<float>::setOutstream("./mem");

	//DataSet<float>* dataSet = new MockDataSet<float>(4, 4, 2, 20, 20, 10, MockDataSet<float>::NOTABLE_IMAGE);
	//DataSet<float>* dataSet = new MockDataSet<float>(28, 28, 1, 100, 100, 10);
	//DataSet<float>* dataSet = new MockDataSet<float>(56, 56, 3, 10, 10, 10);
	//DataSet<float>* dataSet = new MnistDataSet<float>(0.8);
	//DataSet<float>* dataSet = new MockDataSet<float>(224, 224, 3, 100, 100, 100);
	//DataSet<float>* dataSet = createImageNet10CatDataSet<float>();
	//DataSet<float>* dataSet = createImageNet100CatDataSet<float>();
	//DataSet<float>* dataSet = createImageNet1000DataSet<float>();
	DataSet<float>* dataSet = createImageNet1000DataSet<float>();
	//DataSet<float>* dataSet = createImageNet50000DataSet<float>();
	//DataSet<float>* dataSet = createMnistDataSet<float>();
	//DataSet<float>* dataSet = createSampleDataSet<float>();
	dataSet->load();
	//dataSet->zeroMean(true);

	Evaluation<float>* top1Evaluation = new Top1Evaluation<float>();
	Evaluation<float>* top5Evaluation = new Top5Evaluation<float>();
	NetworkListener* networkListener = new NetworkMonitor(NetworkMonitor::PLOT_AND_WRITE);

	//LayersConfig<float>* layersConfig = createCNNSimpleLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createCNNDoubleLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInception3ALayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInception3ALayersConfigTest<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInception3ASimpleLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInception5BLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createVGG19_2_NetLayersConfig<float>();
	LayersConfig<float>* layersConfig = createVGG19NetLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createGoogLeNetInceptionAuxLayersConfig<float>();

	NetworkConfig<float>::Builder* networkBuilder =
			(new NetworkConfig<float>::Builder())
			->batchSize(batchSize)
			->baseLearningRate(baseLearningRate)
			->weightDecay(weightDecay)
			->momentum(momentum)
			->testInterval(testInterval)
			->saveInterval(saveInterval)
			->stepSize(stepSize)
			->clipGradientsLevel(clipGradientsLevel)
			->lrPolicy(lrPolicy)
			->gamma(gamma)
			->dataSet(dataSet)
			->evaluations({top1Evaluation, top5Evaluation})
			->layersConfig(layersConfig)
			->savePathPrefix(savePathPrefix)
			->networkListeners({networkListener});
	networkBuilder->print();
	NetworkConfig<float>* networkConfig = networkBuilder->build();


	Util::printVramInfo();

	Network<float>* network = new Network<float>(networkConfig);
	network->sgd(maxEpoch);
	network->save();

	Cuda::destroy();
}


void network_load() {
	Cuda::create(0);
	cout << "Cuda creation done ... " << endl;

	//DataSet<float>* dataSet = createMnistDataSet<float>();
	DataSet<float>* dataSet = createImageNet10000DataSet<float>();
	dataSet->load();

	Evaluation<float>* top1Evaluation = new Top1Evaluation<float>();
	Evaluation<float>* top5Evaluation = new Top5Evaluation<float>();
	NetworkListener* networkListener = new NetworkMonitor(NetworkMonitor::PLOT_AND_WRITE);

	// save file 경로로 builder 생성,
	NetworkConfig<float>::Builder* networkBuilder = new NetworkConfig<float>::Builder();
	networkBuilder->load("/home/jhkim/network_save/current/network");
	networkBuilder->dataSet(dataSet);
	networkBuilder->evaluations({top1Evaluation, top5Evaluation});
	networkBuilder->networkListeners({networkListener});

	networkBuilder->print();

	NetworkConfig<float>* networkConfig = networkBuilder->build();
	networkConfig->load("");

	Util::printVramInfo();

	Network<float>* network = new Network<float>(networkConfig);
	//network->sgd(10000);
	//network->save();
	network->test();

	Cuda::destroy();
}

void artistic_style() {
	Cuda::create(0);
	cout << "Cuda creation done ... " << endl;
	Util::setPrint(false);

	//ArtisticStyle<float> artisticStyle;
	//artisticStyle.test();

	const uint32_t maxEpoch = 1000;
	const uint32_t batchSize = 1;
	const uint32_t testInterval = 100;			// 10000(목표 샘플수) / batchSize
	const uint32_t saveInterval = 100;			// 1000000 / batchSize
	const uint32_t stepSize = 100000;
	const float baseLearningRate = 0.001f;
	const float weightDecay = 0.0002f;
	const float momentum = 0.9f;
	const float clipGradientsLevel = 0.0f;
	const float gamma = 0.96;
	const LRPolicy lrPolicy = LRPolicy::Step;
	const string savePathPrefix = "/home/jhkim/network_save/current/network";


	/*
	LayersConfig<float>* layersConfig = createCNNDoubleLayersConfig<float>();
	//LayersConfig<float>* layersConfig = createVGG19NetLayersArtisticConfig<float>();
	NetworkConfig<float>::Builder* networkBuilder =
			(new NetworkConfig<float>::Builder())
			->batchSize(batchSize)
			->baseLearningRate(baseLearningRate)
			->weightDecay(weightDecay)
			->momentum(momentum)
			->testInterval(testInterval)
			->saveInterval(saveInterval)
			->stepSize(stepSize)
			->clipGradientsLevel(clipGradientsLevel)
			->lrPolicy(lrPolicy)
			->gamma(gamma)
			->inputShape({8, 8, 3})
			//->inputShape({320, 320, 3})
			->layersConfig(layersConfig)
			->savePathPrefix(savePathPrefix);
	networkBuilder->print();
	NetworkConfig<float>* networkConfig = networkBuilder->build();
	*/

	LayersConfig<float>* layersConfig = createVGG19NetLayersArtisticConfig<float>();
	NetworkConfig<float>::Builder* networkBuilder =
			(new NetworkConfig<float>::Builder())
			->batchSize(batchSize)
			->baseLearningRate(baseLearningRate)
			->weightDecay(weightDecay)
			->momentum(momentum)
			->testInterval(testInterval)
			->saveInterval(saveInterval)
			->stepSize(stepSize)
			->clipGradientsLevel(clipGradientsLevel)
			->lrPolicy(lrPolicy)
			->gamma(gamma)
			//->inputShape({8, 8, 3})
			//->inputShape({320, 320, 3})
			->inputShape({448, 448, 3})
			->layersConfig(layersConfig)
			->savePathPrefix(savePathPrefix);
			//->dataSet(dataSet)
			//->evaluations({top1Evaluation, top5Evaluation})
			//->networkListeners({networkListener});
	networkBuilder->print();
	NetworkConfig<float>* networkConfig = networkBuilder->build();
	networkConfig->load("conv5_1");

	/*
	NetworkConfig<float>::Builder* networkBuilder = new NetworkConfig<float>::Builder();
	networkBuilder->load("/home/jhkim/network_save/current/network");
	networkBuilder->batchSize(1);
	networkBuilder->inputShape({320, 320, 3});
	networkBuilder->print();
	NetworkConfig<float>* networkConfig = networkBuilder->build();
	networkConfig->load();
	*/

	Util::printVramInfo();

	Network<float>* network = new Network<float>(networkConfig);
	//network->test();


	ArtisticStyle<float> artisticStyle(
			network,
			//"/home/jhkim/image/artistic/tubingen_8.jpg",
			//"/home/jhkim/image/artistic/starry_night_8.jpg",
			"/home/jhkim/image/artistic/tubingen_448.jpg",
			"/home/jhkim/image/artistic/eh2_448.jpg",
			//"/home/jhkim/image/artistic/tubingen_448.jpg",
			//"/home/jhkim/image/artistic/starry_night_448.jpg",
			//"/home/jhkim/image/artistic/composition_320.jpg",
			//"/home/jhkim/image/artistic/monk_320.jpg",
			//"/home/jhkim/image/artistic/picasso_320.jpg",
			//"/home/jhkim/image/artistic/simpson_320.jpg",
			//"/home/jhkim/image/artistic/donelli_320.jpg",
			{"conv3_2"},
			{"conv4_1", "conv3_1", "conv2_1", "conv1_1"},
			0.05,					// weight for content grad
			0.00001,					// weight for style grad
			-0.001,					// learning rate
			"conv4_1",				// last layer name
			true,
			true
			);
	artisticStyle.style();
	Cuda::destroy();
}











