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
#include "AccuracyLayer.h"

#include "NetworkTestInterface.h"
#include "NetworkTest.h"

#include "TestUtil.h"
#include "Debug.h"

#include "AnnotationDataLayer.h"
#include "NormalizeLayer.h"
#include "PermuteLayer.h"
#include "FlattenLayer.h"
#include "PriorBoxLayer.h"
#include "ConcatLayer.h"
#include "MultiBoxLossLayer.h"
#include "DetectionOutputLayer.h"
#include "DetectionEvaluateLayer.h"


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "gpu_nms.hpp"
#include "cnpy.h"

void plainTest();
void layerTest();
void networkTest();

int main(void) {
	cout.precision(10);
	cout.setf(ios::fixed);

	//plainTest();
	//layerTest();
	networkTest();

	return 0;
}


void plainTest() {
	cout << "plainTest()" << endl;

#if 1
	AnnotationDataLayer<float>::Builder* builder =
			new typename AnnotationDataLayer<float>::Builder();
	builder->id(0)
			->name("data")
			->flip(true)
			->imageHeight(300)
			->imageWidth(300)
			->imageSetPath("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/trainval.txt")
			->baseDataPath("/home/jkim/Dev/git/caffe_ssd/data/VOCdevkit/")
			->labelMapPath("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/labelmap_voc.prototxt")
			//->pixelMeans({102.9801f, 115.9465f, 122.7717f})	// BGR
			//->pixelMeans({104.f, 117.f, 123.f})	// BGR
			->pixelMeans({0.f, 0.f, 0.f})	// BGR
			->outputs({"data", "label"});

	Layer<float>* layer = builder->build();
	layer->_outputData.push_back(new Data<float>("data"));
	layer->_outputData.push_back(new Data<float>("label"));

	NetworkConfig<float>* networkConfig = (new typename NetworkConfig<float>::Builder())
			->batchSize(1)
			->build();
	layer->setNetworkConfig(networkConfig);
	layer->feedforward();

	Data<float>::printConfig = true;
	SyncMem<float>::printConfig = true;

	layer->_outputData[0]->print_data({}, false);
	layer->_outputData[1]->print_data({}, false, -1);

	SyncMem<float>::printConfig = false;
	Data<float>::printConfig = false;

	delete layer;
#endif

#if 0
	NormalizeLayer<float>::Builder* builder =
			new typename NormalizeLayer<float>::Builder();
	builder->id(0)
			->name("conv4_3_norm")
			->acrossSpatial(false)
			->scaleFiller(ParamFillerType::Constant, 20.0f)
			->channelShared(false)
			->inputs({"conv4_3"})
			->outputs({"conv4_3_norm"});

	Layer<float>* layer = builder->build();
	layer->_inputData.push_back(new Data<float>("conv4_3"));
	layer->_outputData.push_back(new Data<float>("conv4_3_norm"));
#endif

#if 0
	PermuteLayer<float>::Builder* builder =
			new typename PermuteLayer<float>::Builder();
	builder->id(0)
			->name("conv4_3_norm_mbox_loc_perm")
			->orders({0, 2, 3, 1})
			->inputs({"conv4_3_norm_mbox_loc"})
			->outputs({"conv4_3_norm_mbox_loc_perm"});

	Layer<float>* layer = builder->build();
	layer->_inputData.push_back(new Data<float>("conv4_3_norm_mbox_loc"));
	layer->_outputData.push_back(new Data<float>("conv4_3_norm_mbox_loc_perm"));
#endif

#if 0
	FlattenLayer<float>::Builder* builder =
			new typename FlattenLayer<float>::Builder();
	builder->id(0)
			->name("conv4_3_norm_mbox_loc_flat")
			->axis(1)
			->inputs({"conv4_3_norm_mbox_loc_perm"})
			->outputs({"conv4_3_norm_mbox_loc_flat"});

	Layer<float>* layer = builder->build();
	layer->_inputData.push_back(new Data<float>("conv4_3_norm_mbox_loc_perm"));
	layer->_outputData.push_back(new Data<float>("conv4_3_norm_mbox_loc_flat"));
#endif

#if 0
	PriorBoxLayer<float>::Builder* builder =
			new typename PriorBoxLayer<float>::Builder();
	builder->id(0)
			->name("conv4_3_norm_mbox_priorbox")
			->minSizes({30.0})
			->maxSizes({60.0})
			->aspectRatios({2.0})
			->flip(true)
			->clip(false)
			->variances({0.1, 0.1, 0.2, 0.2})
			->step(8.0)
			->offset(0.5)
			->inputs({"conv4_3_norm", "data"})
			->outputs({"conv4_3_norm_mbox_priorbox"});

	Layer<float>* layer = builder->build();
	layer->_inputData.push_back(new Data<float>("conv4_3_norm"));
	layer->_inputData.push_back(new Data<float>("data"));
	layer->_outputData.push_back(new Data<float>("conv4_3_norm_mbox_priorbox"));

	NetworkConfig<float>* networkConfig = (new typename NetworkConfig<float>::Builder())
			->batchSize(2)
			->build();
	layer->setNetworkConfig(networkConfig);
	layer->feedforward();

	delete layer;
#endif

#if 0
	AnnotationDataLayer<float>::Builder* builder =
			new typename AnnotationDataLayer<float>::Builder();
	builder->id(0)
			->name("data")
			->flip(true)
			->imageHeight(300)
			->imageWidth(300)
			->imageSetPath("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/trainval.txt")
			->baseDataPath("/home/jkim/Dev/git/caffe_ssd/data/VOCdevkit/")
			->labelMapPath("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/labelmap_voc.prototxt")
			//->pixelMeans({102.9801f, 115.9465f, 122.7717f})	// BGR
			->pixelMeans({104.f, 117.f, 123.f})	// BGR
			->outputs({"data", "label"});

	AnnotationDataLayer<float>* layer = dynamic_cast<AnnotationDataLayer<float>*>(builder->build());
	layer->_inputData.push_back(new Data<float>("data"));
	layer->_inputData.push_back(new Data<float>("label"));

	NetworkConfig<float>* networkConfig = (new typename NetworkConfig<float>::Builder())
			->batchSize(32)
			->build();
	layer->setNetworkConfig(networkConfig);

	for (int i = 0; i < 100000; i++) {
		layer->feedforward();
	}


	map<string, int>& refCount = layer->refCount;
	for (map<string, int>::iterator it = refCount.begin(); it != refCount.end(); it++) {
		cout << it->first << "\t\t" << it->second << endl;
	}

	delete layer;
#endif

#if 0
	cv::VideoCapture cap("/home/jkim/Downloads/frcnn_ok.mp4");
	if (!cap.isOpened()) {
		return;
	}
	cv::VideoCapture cap1("/home/jkim/Downloads/frcnn_ok.mp4");
	if (!cap1.isOpened()) {
		return;
	}
	cv::Mat im;
	if (!cap.read(im)) {
		cout << "error at cap" << endl;
	}
	if (cap1.read(im)) {
		cout << "error at cap1" << endl;
	}
	exit(1);


	//cap.set(CV_CAP_PROP_FOURCC, CV_FOURCC('A', 'V', 'C', '1'));
	//int codec = int(cap.get(CV_CAP_PROP_FOURCC));
	//cout << codec << endl;
	double fps = cap.get(CV_CAP_PROP_FPS);
	cout << "fps: " << fps << endl;
	size_t frameCount = size_t(cap.get(CV_CAP_PROP_FRAME_COUNT));
	cv::namedWindow("Video", 1);
	while (1) {
		size_t posFrames = size_t(cap.get(CV_CAP_PROP_POS_FRAMES));

		cv::Mat frame;
		for (int i = 0; i < 3; i++) {
			if (!cap.grab()) {
				cout << "end of video ... " << endl;
				cap.release();
				return;
			}
		}
		cap.retrieve(frame);
		imshow("Video", frame);
		if (cv::waitKey(30) == 'c') break;
	}

	cap.release();


#endif

	/*
	//checkCudaErrors(cudaSetDevice(0));
	//checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));

	cout << "cudnn version: " << CUDNN_VERSION << endl;


	int pad = 6;
	int stride = 1;
	int dilation = 6;

	cudnnConvolutionDescriptor_t convDesc;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
	cudnnStatus_t status = cudnnSetConvolution2dDescriptor(convDesc,
			pad, pad, stride, stride, dilation, dilation,
			CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

	std::stringstream _error;
	if (status != CUDNN_STATUS_SUCCESS) {
	  _error << "CUDNN failure: " << cudnnGetErrorString(status);
	  FatalError(_error.str());
	}

	cout << "dilation test done ... " << endl;
	//checkCUDNN(cudnnDestroy(Cuda::cudnnHandle));
	 */
}

void layerTest() {
	const int gpuid = 0;
	vector<LayerTestInterface<float>*> layerTestList;

#if 0
	ConvLayer<float>::Builder* convBuilder = new typename ConvLayer<float>::Builder();
	convBuilder->id(10)
		->name("fc6")
		->filterDim(3, 3, 512, 1024, 6, 1)
		->dilation(6)
		->inputs({"pool5"})
		->outputs({"fc6"});
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

#if 0
	FrcnnTestOutputLayer<float>::Builder* frcnnTestOutputBuilder =
			new typename FrcnnTestOutputLayer<float>::Builder();
	frcnnTestOutputBuilder->id(350)
			->name("test_output")
			//->maxPerImage(5)
			->thresh(0.5)
			->inputs({"rois", "im_info", "cls_prob", "bbox_pred"});

	layerTestList.push_back(new LayerTest<float>(frcnnTestOutputBuilder));
#endif

#if 1
	AnnotationDataLayer<float>::Builder* annotationDataBuilder =
			new typename AnnotationDataLayer<float>::Builder();
	annotationDataBuilder->id(0)
			->name("data")
			->flip(true)
			->imageHeight(300)
			->imageWidth(300)
			->imageSetPath("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/trainval.txt")
			->baseDataPath("/home/jkim/Dev/git/caffe_ssd/data/VOCdevkit/")
			->labelMapPath("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/labelmap_voc.prototxt")
			//->pixelMeans({102.9801f, 115.9465f, 122.7717f})	// BGR
			->pixelMeans({104.f, 117.f, 123.f})	// BGR
			->outputs({"data", "label"});
	layerTestList.push_back(new LayerTest<float>(annotationDataBuilder));
#endif

#if 0
	AccuracyLayer<float>::Builder* accuracyBuilder =
			new typename AccuracyLayer<float>::Builder();
	accuracyBuilder->id(390)
			->name("accuracy")
			->topK(5)
			->axis(2)
			->inputs({"fc8_fc8_0_split_1", "label_data_1_split_1"})
			->outputs({"accuracy"});
	layerTestList.push_back(new LayerTest<float>(accuracyBuilder));
#endif

#if 0
	NormalizeLayer<float>::Builder* normalizeBuilder =
			new typename NormalizeLayer<float>::Builder();
	normalizeBuilder->id(0)
			->name("conv4_3_norm")
			->acrossSpatial(false)
			->scaleFiller(ParamFillerType::Constant, 20.0f)
			->channelShared(false)
			->inputs({"conv4_3_relu4_3_0_split_1"})
			->outputs({"conv4_3_norm"});

	NetworkConfig<float>* networkConfig = (new typename NetworkConfig<float>::Builder())
			->networkPhase(NetworkPhase::TrainPhase)
			->batchSize(4)
			->build();

	layerTestList.push_back(new LearnableLayerTest<float>(normalizeBuilder, networkConfig));
#endif

#if 0
	PermuteLayer<float>::Builder* builder =
			new typename PermuteLayer<float>::Builder();
	builder->id(0)
			->name("conv4_3_norm_mbox_loc_perm")
			->orders({0, 2, 3, 1})
			->inputs({"conv4_3_norm_mbox_loc"})
			->outputs({"conv4_3_norm_mbox_loc_perm"});

	layerTestList.push_back(new LayerTest<float>(builder));
#endif

#if 0
	FlattenLayer<float>::Builder* builder =
			new typename FlattenLayer<float>::Builder();
	builder->id(0)
			->name("conv4_3_norm_mbox_loc_flat")
			->axis(1)
			->endAxis(3)
			->inputs({"conv4_3_norm_mbox_loc_perm"})
			->outputs({"conv4_3_norm_mbox_loc_flat"});

	layerTestList.push_back(new LayerTest<float>(builder));
#endif

#if 0
	PriorBoxLayer<float>::Builder* builder =
			new typename PriorBoxLayer<float>::Builder();
	builder->id(0)
			->name("conv4_3_norm_mbox_priorbox")
			->minSizes({30.0})
			->maxSizes({60.0})
			->aspectRatios({2.0})
			->flip(true)
			->clip(false)
			->variances({0.1, 0.1, 0.2, 0.2})
			->step(8.0)
			->offset(0.5)
			->inputs({"conv4_3_norm_conv4_3_norm_0_split_2", "data_data_0_split_1"})
			->outputs({"conv4_3_norm_mbox_priorbox"});

	layerTestList.push_back(new LayerTest<float>(builder));
#endif

#if 0
	ConcatLayer<float>::Builder* builder =
			new typename ConcatLayer<float>::Builder();
	builder->id(0)
			->name("mbox_loc")
			->axis(1)
			->inputs({
				"conv4_3_norm_mbox_loc_flat",
				"fc7_mbox_loc_flat",
				"conv6_2_mbox_loc_flat",
				"conv7_2_mbox_loc_flat",
				"conv8_2_mbox_loc_flat",
				"conv9_2_mbox_loc_flat"})
			->outputs({"mbox_loc"});

	layerTestList.push_back(new LayerTest<float>(builder));
#endif

#if 0
	MultiBoxLossLayer<float>::Builder* builder =
			new typename MultiBoxLossLayer<float>::Builder();
	builder->id(0)
			->name("mbox_loss")
			->locLossType("SMOOTH_L1")
			->confLossType("SOFTMAX")
			->locWeight(1.0)
			->numClasses(21)
			->shareLocation(true)
			->matchType("PER_PREDICTION")
			->overlapThreshold(0.5)
			->usePriorForMatching(true)
			->backgroundLabelId(0)
			->useDifficultGt(true)
			->negPosRatio(3.0)
			->negOverlap(0.5)
			->codeType("CENTER_SIZE")
			->ignoreCrossBoundaryBbox(false)
			->miningType("MAX_NEGATIVE")
			->propDown({true, true, false, false})
			->inputs({"mbox_loc", "mbox_conf", "mbox_priorbox", "label"})
			->outputs({"mbox_loss"});

	layerTestList.push_back(new LayerTest<float>(builder));
#endif

#if 0
	DetectionOutputLayer<float>::Builder* builder =
			new typename DetectionOutputLayer<float>::Builder();
	builder->id(0)
			->name("detection_out")
			->numClasses(21)
			->shareLocation(true)
			->backgroundLabelId(0)
			->nmsThreshold(0.449999988079)
			->topK(400)
			->outputDirectory("/home/jkim/Dev/data/ssd/data/VOCdevkit/results/VOC2007/SSD_300x300/Main")
			->outputNamePrefix("comp4_det_test_")
			->outputFormat("VOC")
			->labelMapFile("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/labelmap_voc.prototxt")
			->nameSizeFile("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/test_name_size.txt")
			->numTestImage(4952)
			->codeType("CENTER_SIZE")
			->keepTopK(200)
			->confidenceThreshold(0.00999999977648)
			->visualize(true)
			->propDown({false, false, false, false})
			->inputs({"mbox_loc", "mbox_conf_flatten", "mbox_priorbox", "data_data_0_split_7"})
			->outputs({"detection_out"});

	layerTestList.push_back(new LayerTest<float>(builder));
#endif

#if 0
	DetectionEvaluateLayer<float>::Builder* builder =
			new typename DetectionEvaluateLayer<float>::Builder();
	builder->id(0)
			->name("detection_eval")
			->numClasses(21)
			->backgroundLabelId(0)
			->overlapThreshold(0.5)
			->evaluateDifficultGt(false)
			->nameSizeFile("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/test_name_size.txt")
			->propDown({false, false})
			->inputs({"detection_out", "label"})
			->outputs({"detection_eval"});

	layerTestList.push_back(new LayerTest<float>(builder));
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
#define NETWORK_SSD			4
#define NETWORK_SSD_TEST	5
#define NETWORK				NETWORK_SSD



void saveNetworkParams(LayersConfig<float>* layersConfig);


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
#elif NETWORK == NETWORK_SSD
	const int numSteps = 3;

	LayersConfig<float>* layersConfig = createSSDNetLayersConfig<float>();
	const string networkName		= "ssd";
	const int batchSize 			= 1;
	const float baseLearningRate 	= 0.001;
	const float weightDecay 		= 0.0005;
	//const float baseLearningRate 	= 1;
	//const float weightDecay 		= 0.000;
	//const float momentum 			= 0.0;
	const float momentum 			= 0.0;
	const LRPolicy lrPolicy 		= LRPolicy::Fixed;
	const int stepSize 				= 50000;
	const NetworkPhase networkPhase	= NetworkPhase::TrainPhase;
	const float gamma 				= 0.1;
#elif NETWORK == NETWORK_SSD_TEST
	const int numSteps = 1;

	LayersConfig<float>* layersConfig = createSSDNetTestLayersConfig<float>();
	const string networkName		= "ssd";
	const int batchSize 			= 2;
	const float baseLearningRate 	= 0.01;
	const float weightDecay 		= 0.0005;
	//const float baseLearningRate 	= 1;
	//const float weightDecay 		= 0.000;
	//const float momentum 			= 0.0;
	const float momentum 			= 0.0;
	const LRPolicy lrPolicy 		= LRPolicy::Fixed;
	const int stepSize 				= 50000;
	const NetworkPhase networkPhase	= NetworkPhase::TrainPhase;
	const float gamma 				= 0.1;
#else
	cout << "invalid network ... " << endl;
	exit(1);
#endif



#if 0
	vector<WeightsArg> weightsArgs(1);
	//weightsArgs[0].weightsPath = "/home/jkim/Dev/SOOOA_HOME/network/SSD_PRETRAINED.param";
	weightsArgs[0].weightsPath = "/home/jkim/Dev/SOOOA_HOME/network/SSD_CAFFE_TRAINED.param";
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
#if 0
		->weightsArgs(weightsArgs)
#endif
		->build();

	for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->setNetworkConfig(networkConfig);
	}


#if 0
	Network<float>* network = new Network<float>(networkConfig);
	// (2) network config 정보를 layer들에게 전달한다.
	for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->setNetworkConfig(network->config);
	}
	network->setLayersConfig(layersConfig);
	network->loadPretrainedWeights();
#endif

#if 0
	Data<float>::printConfig = true;
	SyncMem<float>::printConfig = true;
	for (int i = 0; i < layersConfig->_learnableLayers.size(); i++) {
		for (int j = 0; j < layersConfig->_learnableLayers[i]->_params.size(); j++) {
			layersConfig->_learnableLayers[i]->_params[j]->print_data({}, false);
		}
	}
	Data<float>::printConfig = false;
	SyncMem<float>::printConfig = false;
	exit(1);
#endif

	NetworkTest<float>* networkTest = new NetworkTest<float>(layersConfig, networkName, numSteps);
	NetworkTestInterface<float>::globalSetUp(gpuid);
	networkTest->setUp();
	//saveNetworkParams(layersConfig);
	//exit(1);
	networkTest->updateTest();
	networkTest->cleanUp();
	NetworkTestInterface<float>::globalCleanUp();
}

void saveNetworkParams(LayersConfig<float>* layersConfig) {
	const string savePathPrefix = "/home/jkim/Dev/SOOOA_HOME/network";
	ofstream paramOfs(
			(savePathPrefix+"/SSD_CAFFE_TRAINED.param").c_str(),
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
}

#endif
