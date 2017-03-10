#if 0

#include "LayerTestInterface.h"
#include "LayerTest.h"
#include "LearnableLayerTest.h"

#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "ReluLayer.h"
#include "PoolingLayer.h"
#include "SoftmaxWithLossLayer.h"

#include "NetworkTestInterface.h"
#include "NetworkTest.h"

#include "TestUtil.h"
#include "Debug.h"


void layerTest();
void networkTest();

int main(void) {
	cout.precision(10);
	cout.setf(ios::fixed);

	//layerTest();
	networkTest();
}

void layerTest() {
	const int gpuid = 0;
	vector<LayerTestInterface<float>*> layerTestList;

#if 1
	ConvLayer<float>::Builder* convBuilder = new typename ConvLayer<float>::Builder();
	convBuilder->id(1)
			->name("conv2")
			->filterDim(5, 5, 20, 50, 0, 1)
			->inputs({"pool1"})
			->outputs({"conv2"});
	layerTestList.push_back(new LearnableLayerTest<float>(convBuilder));
#endif

#if 0
	FullyConnectedLayer<float>::Builder* fcBuilder =
			new typename FullyConnectedLayer<float>::Builder();
	fcBuilder->id(4)
			->name("ip1")
			->nOut(500)
			->inputs({"pool2"})
			->outputs({"ip1"});
	layerTestList.push_back(new LearnableLayerTest<float>(fcBuilder));
#endif

#if 0
	Layer<float>::Builder* reluBuilder =
			new typename ReluLayer<float>::Builder();
	reluBuilder->id(42)
			->name("relu1")
			->inputs({"ip1"})
			->outputs({"relu1"});
	layerTestList.push_back(new LayerTest<float>(reluBuilder));
#endif

#if 0
	PoolingLayer<float>::Builder* poolBuilder =
			new typename PoolingLayer<float>::Builder();
	poolBuilder->id(42)
			->name("pool1")
			->poolDim(2, 2, 0, 2)
			->poolingType(Pooling<float>::Max)
			->inputs({"conv1"})
			->outputs({"pool1"});
	layerTestList.push_back(new LayerTest<float>(poolBuilder));
#endif

#if 0
	SoftmaxWithLossLayer<float>::Builder* softmaxWithLossBuilder =
			new typename SoftmaxWithLossLayer<float>::Builder();
	softmaxWithLossBuilder->id(42)
			->name("loss")
			->inputs({"ip2", "label"})
			->outputs({"loss"});
	layerTestList.push_back(new LayerTest<float>(softmaxWithLossBuilder));
#endif


	LayerTestInterface<float>::globalSetUp(gpuid);
	for (uint32_t i = 0; i < layerTestList.size(); i++) {
		LayerTestInterface<float>* layerTest = layerTestList[i];
		layerTest->setUp();
		layerTest->forwardTest();
		layerTest->backwardTest();
		layerTest->cleanUp();
	}
	LayerTestInterface<float>::globalCleanUp();
}


#define NETWORK_LENET	0
#define NETWORK_VGG19	1
#define NETWORK			NETWORK_VGG19



void networkTest() {
	const int gpuid 				= 0;

#if NETWORK == NETWORK_LENET
	// LENET
	LayersConfig<float>* layersConfig = createLeNetLayersConfig<float>();
	const string networkName		= "lenet";
	const int batchSize 			= 64;
	const float baseLearningRate 	= 0.01;
	const float weightDecay 		= 0.0005;
	const float momentum 			= 0.0;
	const LRPolicy lrPolicy 		= LRPolicy::Fixed;
	const int stepSize 				= 20000;
	const NetworkPhase networkPhase	= NetworkPhase::TrainPhase;
	const float gamma 				= 0.0001;
#elif NETWORK == NETWORK_VGG19
	// VGG19
	LayersConfig<float>* layersConfig = createVGG19NetLayersConfig<float>();
	const string networkName		= "vgg19";
	const int batchSize 			= 10;
	const float baseLearningRate 	= 0.1;
	const float weightDecay 		= 0.05;
	const float momentum 			= 0.0;
	const LRPolicy lrPolicy 		= LRPolicy::Step;
	const int stepSize 				= 20000;
	const NetworkPhase networkPhase	= NetworkPhase::TrainPhase;
	const float gamma 				= 0.1;
#else
	cout << "invalid network ... " << endl;
	exit(1);
#endif

	NetworkConfig<float>* networkConfig =
		(new typename NetworkConfig<float>::Builder())
		->batchSize(batchSize)
		->baseLearningRate(baseLearningRate)
		->weightDecay(weightDecay)
		->momentum(momentum)
		->lrPolicy(lrPolicy)
		->stepSize(stepSize)
		->networkPhase(networkPhase)
		->gamma(gamma)
		->build();

	for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->setNetworkConfig(networkConfig);
	}

	NetworkTest<float>* networkTest = new NetworkTest<float>(layersConfig, networkName);

	NetworkTestInterface<float>::globalSetUp(gpuid);
	networkTest->setUp();
	networkTest->updateTest();
	networkTest->cleanUp();
	NetworkTestInterface<float>::globalCleanUp();
}

#endif
