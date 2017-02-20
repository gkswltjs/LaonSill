#include "LayerTestInterface.h"
#include "ConvLayerTest.h"
#include "FullyConnectedLayerTest.h"

#include "TestUtil.h"

int main(void) {
	const int gpuid = 0;
	vector<LayerTestInterface<float>*> layerTestList;

#if 1
	ConvLayer<float>::Builder* convBuilder = new typename ConvLayer<float>::Builder();
	convBuilder->id(1)
			->name("conv2")
			->filterDim(5, 5, 20, 50, 0, 1)
			->inputs({"pool1"})
			->outputs({"conv2"});
	layerTestList.push_back(new ConvLayerTest<float>(gpuid, convBuilder));
#endif

#if 1
	FullyConnectedLayer<float>::Builder* fcBuilder =
			new typename FullyConnectedLayer<float>::Builder();
	fcBuilder->id(4)
			->name("ip1")
			->nOut(500)
			->inputs({"pool2"})
			->outputs({"ip1"});
	layerTestList.push_back(new FullyConnectedLayerTest<float>(gpuid, fcBuilder));
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
