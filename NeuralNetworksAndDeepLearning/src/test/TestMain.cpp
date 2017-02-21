#include "LayerTestInterface.h"
#include "ConvLayerTest.h"
#include "FullyConnectedLayerTest.h"
#include "LayerTest.h"

#include "ReluLayer.h"
#include "PoolingLayer.h"
#include "SoftmaxWithLossLayer.h"

#include "TestUtil.h"

int main(void) {
	const int gpuid = 0;
	vector<LayerTestInterface<float>*> layerTestList;

#if 0
	ConvLayer<float>::Builder* convBuilder = new typename ConvLayer<float>::Builder();
	convBuilder->id(1)
			->name("conv2")
			->filterDim(5, 5, 20, 50, 0, 1)
			->inputs({"pool1"})
			->outputs({"conv2"});
	layerTestList.push_back(new ConvLayerTest<float>(convBuilder));
#endif

#if 0
	FullyConnectedLayer<float>::Builder* fcBuilder =
			new typename FullyConnectedLayer<float>::Builder();
	fcBuilder->id(4)
			->name("ip1")
			->nOut(500)
			->inputs({"pool2"})
			->outputs({"ip1"});
	layerTestList.push_back(new FullyConnectedLayerTest<float>(fcBuilder));
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
