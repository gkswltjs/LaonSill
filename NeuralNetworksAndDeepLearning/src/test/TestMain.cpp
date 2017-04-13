#if 0

#include "LayerTestInterface.h"
#include "LayerTest.h"
#include "LayerInputTest.h"
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


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "gpu_nms.hpp"
#include "cnpy.h"


void layerTest();
void networkTest();

int main(void) {
	cout.precision(10);
	cout.setf(ios::fixed);

	//layerTest();
	networkTest();


	/*
	RoIInputLayer<float>::Builder* roiInputBuilder =
				new typename RoIInputLayer<float>::Builder();
	roiInputBuilder->id(0)
			->name("input-data")
			->numClasses(21)
			->pixelMeans({102.9801f, 115.9465f, 122.7717f})	// BGR
			->outputs({"data", "im_info", "gt_boxes"});

	RoIInputLayer<float>* layer = dynamic_cast<RoIInputLayer<float>*>(roiInputBuilder->build());

	for (int i = 0; i < 1000000; i++) {
		layer->feedforward();
	}

	map<string, RoIInputLayer<float>::InputStat*>::iterator itr;
	for (itr = layer->inputStatMap.begin(); itr != layer->inputStatMap.end(); itr++) {
		cout << itr->first << ": " << endl;
		cout << "\tnfcnt: " << itr->second->nfcnt << endl;
		cout << "\tnfcnt: " << itr->second->fcnt << endl;
		for (int i = 0; i < 4; i++) {
			cout << "\tscaleCnt[" << i << "]: " << itr->second->scaleCnt[i] << endl;
		}
	}
	*/


	return 0;
}

void layerTest() {
	const int gpuid = 0;
	vector<LayerTestInterface<float>*> layerTestList;

#if 0
	ConvLayer<float>::Builder* convBuilder = new typename ConvLayer<float>::Builder();
	convBuilder->id(10)
		->name("conv2")
		->filterDim(5, 5, 96, 256, 1, 2)
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
	poolBuilder->id(3)
			->name("pool1/3x3_s2")
			->poolDim(3, 3, 0, 2)
			->poolingType(Pooling<float>::Max)
			->inputs({"conv1/7x7_s2"})
			->outputs({"pool1/3x3_s2"});
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

#if 0
	LRNLayer<float>::Builder* lrnBuilder =
			new typename LRNLayer<float>::Builder();
	lrnBuilder->id(42)
			->name("pool1/norm1")
			->lrnDim(5, 0.0001, 0.75, 1.0)
			->inputs({"pool1/3x3_s2"})
			->outputs({"pool1/norm1"});
	layerTestList.push_back(new LayerTest<float>(lrnBuilder));
#endif

#if 0
	DepthConcatLayer<float>::Builder* depthConcatBuilder =
			new typename DepthConcatLayer<float>::Builder();
	depthConcatBuilder->id(24)
			->name("inception_3a/output")
			->propDown({true, true, true, true})
			->inputs({
				"inception_3a/1x1",
				"inception_3a/3x3",
				"inception_3a/5x5",
				"inception_3a/pool_proj"})
			->outputs({"inception_3a/output"});

	layerTestList.push_back(new LayerTest<float>(depthConcatBuilder));
#endif

#if 0
	SplitLayer<float>::Builder* splitBuilder =
			new typename SplitLayer<float>::Builder();
	splitBuilder->id(24)
			->name("pool2/3x3_s2_pool2/3x3_s2_0_split")
			->inputs({"pool2/3x3_s2"})
			->outputs({"pool2/3x3_s2_pool2/3x3_s2_0_split_0",
				"pool2/3x3_s2_pool2/3x3_s2_0_split_1",
				"pool2/3x3_s2_pool2/3x3_s2_0_split_2",
				"pool2/3x3_s2_pool2/3x3_s2_0_split_3"});

	layerTestList.push_back(new LayerTest<float>(splitBuilder));
#endif

#if 0
	RoIInputLayer<float>::Builder* roiInputBuilder =
			new typename RoIInputLayer<float>::Builder();
	roiInputBuilder->id(0)
			->name("input-data")
			->numClasses(21)
			->pixelMeans({102.9801f, 115.9465f, 122.7717f})	// BGR
			->outputs({"data", "im_info", "gt_boxes"});

	layerTestList.push_back(new LayerInputTest<float>(roiInputBuilder));
#endif

#if 0
	RoITestInputLayer<float>::Builder* roiTestInputBuilder =
			new typename RoITestInputLayer<float>::Builder();
	roiTestInputBuilder->id(0)
			->name("input-data")
			->numClasses(21)
			->pixelMeans({102.9801f, 115.9465f, 122.7717f})	// BGR
			->outputs({"data", "im_info"});

	layerTestList.push_back(new LayerInputTest<float>(roiTestInputBuilder));
#endif

#if 0
	AnchorTargetLayer<float>::Builder* anchorTargetBuilder =
			new typename AnchorTargetLayer<float>::Builder();
	anchorTargetBuilder->id(14)
			->name("rpn-data")
			->featStride(16)
			->inputs({
				"rpn_cls_score_rpn_cls_score_0_split_1",
				"gt_boxes_input-data_2_split_0",
				"im_info_input-data_1_split_0",
				"data_input-data_0_split_1"})
			->propDown({false, false, false, false})
			->outputs({
				"rpn_labels",
				"rpn_bbox_targets",
				"rpn_bbox_inside_weights",
				"rpn_bbox_outside_weights"});

	layerTestList.push_back(new LayerTest<float>(anchorTargetBuilder));
#endif

#if 0
	ProposalLayer<float>::Builder* proposalBuilder =
			new typename ProposalLayer<float>::Builder();
	proposalBuilder->id(19)
			->name("proposal")
			->featStride(16)
			->inputs({
				"rpn_cls_prob_reshape",
				"rpn_bbox_pred_rpn_bbox_pred_0_split_1",
				"im_info_input-data_1_split_1"})
			->propDown({false, false, false})
			->outputs({"rpn_rois"});

	NetworkConfig<float>* networkConfig = (new typename NetworkConfig<float>::Builder())
			->networkPhase(NetworkPhase::TrainPhase)
			->build();

	layerTestList.push_back(new LayerTest<float>(proposalBuilder, networkConfig));
#endif

#if 0
	ProposalTargetLayer<float>::Builder* proposalTargetBuilder =
			new typename ProposalTargetLayer<float>::Builder();
	proposalTargetBuilder->id(20)
			->name("roi-data")
			->numClasses(21)
			->inputs({
				"rpn_rois",
				"gt_boxes_input-data_2_split_1"})
			->propDown({false, false})
			->outputs({
				"rois",
				"labels",
				"bbox_targets",
				"bbox_inside_weights",
				"bbox_outside_weights"});

	layerTestList.push_back(new LayerTest<float>(proposalTargetBuilder));
#endif

#if 0
	ReshapeLayer<float>::Builder* reshapeBuilder =
			new typename ReshapeLayer<float>::Builder();
	reshapeBuilder->id(13)
			->name("rpn_cls_score_reshape")
			->shape({0, 2, -1, 0})
			->inputs({"rpn_cls_score_rpn_cls_score_0_split_0"})
			->propDown({false})
			->outputs({"rpn_cls_score_reshape"});

	layerTestList.push_back(new LayerTest<float>(reshapeBuilder));
#endif

#if 0
	SmoothL1LossLayer<float>::Builder* smoothL1LossBuilder =
			new typename SmoothL1LossLayer<float>::Builder();
	smoothL1LossBuilder->id(16)
			->name("rpn_loss_bbox")
			->lossWeight(1.0f)
			->sigma(3.0f)
			->inputs({
				"rpn_bbox_pred_rpn_bbox_pred_0_split_0",
				"rpn_bbox_targets",
				"rpn_bbox_inside_weights",
				"rpn_bbox_outside_weights"})
			->propDown({false, false, false, false})
			->outputs({"rpn_loss_bbox"});

	layerTestList.push_back(new LayerTest<float>(smoothL1LossBuilder));
#endif

#if 0
	RoIPoolingLayer<float>::Builder* roiPoolingBuilder =
			new typename RoIPoolingLayer<float>::Builder();
	roiPoolingBuilder->id(31)
			->name("roi_pool5")
			->pooledW(6)
			->pooledH(6)
			->spatialScale(0.0625f)
			->inputs({
				"conv5_relu5_0_split_1",
				"rois"})
			->outputs({"pool5"});

	layerTestList.push_back(new LayerTest<float>(roiPoolingBuilder));
#endif

#if 1
	FrcnnTestOutputLayer<float>::Builder* frcnnTestOutputBuilder =
			new typename FrcnnTestOutputLayer<float>::Builder();
	frcnnTestOutputBuilder->id(350)
			->name("test_output")
			//->maxPerImage(5)
			->thresh(0.5)
			->inputs({"rois", "im_info", "cls_prob", "bbox_pred"});

	layerTestList.push_back(new LayerTest<float>(frcnnTestOutputBuilder));
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


#define NETWORK_LENET		0
#define NETWORK_VGG19		1
#define NETWORK_FRCNN		2
#define NETWORK_FRCNN_TEST	3
#define NETWORK				NETWORK_FRCNN



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
	//LayersConfig<float>* layersConfig = createVGG19NetLayersConfig<float>();
	LayersConfig<float>* layersConfig = createInceptionLayersConfig<float>();
	const string networkName		= "inception";
	const int batchSize 			= 8;
	const float baseLearningRate 	= 0.01;
	const float weightDecay 		= 0.0002;
	const float momentum 			= 0.0;
	const LRPolicy lrPolicy 		= LRPolicy::Step;
	const int stepSize 				= 320000;
	const NetworkPhase networkPhase	= NetworkPhase::TrainPhase;
	const float gamma 				= 0.96;
#elif NETWORK == NETWORK_FRCNN
	const int numSteps = 2;

	// FRCNN
	LayersConfig<float>* layersConfig = createFrcnnTrainOneShotLayersConfig<float>();
	const string networkName		= "frcnn";
	const int batchSize 			= 1;
	const float baseLearningRate 	= 0.01;
	const float weightDecay 		= 0.0005;
	//const float baseLearningRate 	= 1;
	//const float weightDecay 		= 0.000;
	const float momentum 			= 0.0;
	const LRPolicy lrPolicy 		= LRPolicy::Step;
	const int stepSize 				= 50000;
	const NetworkPhase networkPhase	= NetworkPhase::TrainPhase;
	const float gamma 				= 0.1;
#elif NETWORK == NETWORK_FRCNN_TEST
	// FRCNN_TEST
	LayersConfig<float>* layersConfig = createFrcnnTestOneShotLayersConfig<float>();
	const string networkName		= "frcnn";
	const NetworkPhase networkPhase	= NetworkPhase::TestPhase;
#else
	cout << "invalid network ... " << endl;
	exit(1);
#endif
	NetworkConfig<float>* networkConfig =
		(new typename NetworkConfig<float>::Builder())
		//->networkPhase(networkPhase)
		//->build();
		->batchSize(batchSize)
		->baseLearningRate(baseLearningRate)
		->weightDecay(weightDecay)
		->momentum(momentum)
		->stepSize(stepSize)
		->lrPolicy(lrPolicy)
		->networkPhase(networkPhase)
		->gamma(gamma)
		->build();

	for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->setNetworkConfig(networkConfig);
	}

	NetworkTest<float>* networkTest = new NetworkTest<float>(layersConfig, networkName, numSteps);
	NetworkTestInterface<float>::globalSetUp(gpuid);
	networkTest->setUp();

	/*
	const string savePathPrefix = "/home/jkim/Dev/SOOOA_HOME/network";
	ofstream paramOfs(
			(savePathPrefix+"/VGG_CNN_M_1024_FRCNN_CAFFE.param").c_str(),
			ios::out | ios::binary);

	uint32_t numLearnableLayers = layersConfig->_learnableLayers.size();
	uint32_t numParams = 0;
	for (uint32_t i = 0; i < numLearnableLayers; i++) {
		numParams += layersConfig->_learnableLayers[i]->numParams();
	}

	paramOfs.write((char*)&numParams, sizeof(uint32_t));
	for (uint32_t i = 0; i < numLearnableLayers; i++) {
		layersConfig->_learnableLayers[i]->saveParams(paramOfs);
	}
	paramOfs.close();
	*/
	networkTest->updateTest();
	networkTest->cleanUp();
	NetworkTestInterface<float>::globalCleanUp();
}

#endif
