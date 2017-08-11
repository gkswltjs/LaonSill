#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/variant.hpp>



#include "LayerTestInterface.h"
#include "LayerTest.h"
#include "LearnableLayerTest.h"

#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "ReluLayer.h"
#include "PoolingLayer.h"
#include "SoftmaxWithLossLayer.h"
#include "DataInputLayer.h"
#include "MultiLabelDataInputLayer.h"

#include "NetworkTestInterface.h"
#include "NetworkTest.h"

#include "TestUtil.h"
#include "Debug.h"

#include "PlanParser.h"
#include "WorkContext.h"
#include "PhysicalPlan.h"
#include "LearnableLayer.h"
#include "PropMgmt.h"
#include "InitParam.h"
#include "Perf.h"
#include "ColdLog.h"
#include "Broker.h"
#include "ResourceManager.h"
#include "PlanOptimizer.h"
#include "LayerFunc.h"
#include "gpu_nms.hpp"
#include "cnpy.h"
#include "ParamManipulator.h"
#include "Datum.h"
#include "SDF.h"
#include "IO.h"
#include "DataReader.h"
#include "ssd_common.h"
#include "Tools.h"
#include "StdOutLog.h"



using namespace std;
namespace fs = ::boost::filesystem;






void plainTest(int argc, char** argv);

void dataReaderTest(int argc, char** argv);
void dataReaderMemoryLeakTest();
void runNetwork();

void layerTest(int argc, char** argv);
void networkTest(int argc, char** argv);
void saveNetwork();


#if 0
int main(int argc, char** argv) {
	cout << "begin test ... " << endl;
	cout.precision(2);
	cout.setf(ios::fixed);

	plainTest(argc, argv);
	//layerTest(argc, argv);
	//networkTest(argc, argv);
	//saveNetwork();

	cout << "end test ... " << endl;
	return 0;
}
#endif



void initializeNetwork() {
	WorkContext::curBootMode = BootMode::DeveloperMode;

	InitParam::init();
	Perf::init();
	SysLog::init();
	ColdLog::init();
	Job::init();
	Task::init();
	Broker::init();
	Network<float>::init();

	ResourceManager::init();
	PlanOptimizer::init();
	LayerFunc::init();
	LayerPropList::init();

	Util::setOutstream(&cout);
	Util::setPrint(false);
}


void plainTest(int argc, char** argv) {
	//denormalizeTest(argc, argv);
	//convertMnistDataTest(argc, argv);
	//convertImageSetTest(argc, argv);
	//convertAnnoSetTest(argc, argv);
	//dataReaderTest(argc, argv);
	//computeImageMean(argc, argv);
	//runNetwork();
	//dataReaderMemoryLeakTest();
}

void dataReaderTest(int argc, char** argv) {
	DataReader<Datum> dr("/home/jkim/Dev/SOOOA_HOME/data/sdf/plantynet_train_0.25/");
	//DataReader<Datum> dr("/home/jkim/Dev/SOOOA_HOME/data/sdf/test_train/");
	int numData = dr.getNumData();
	cout << "numData: " << numData << endl;

	const string windowName = "result";
	cv::namedWindow(windowName);

	for (int i = 0; i < std::min(numData, 100); i++) {
		Datum* datum = dr.getNextData();
		cout << i << " label: " << datum->label;
		if (datum->float_data.size() > 0) {
			for (int j = 0; j < datum->float_data.size(); j++) {
				cout << "," << (int)datum->float_data[j];
			}
		}
		cout << endl;
		datum->print();
		//PrintDatumData(datum, false);

		cv::Mat cv_img = DecodeDatumToCVMat(datum, true, true);
		//PrintCVMatData(cv_img);

		cv::imshow(windowName, cv_img);
		cv::waitKey(0);
	}
	cv::destroyWindow(windowName);
}

void dataReaderMemoryLeakTest() {
	const string source = "/home/jkim/Dev/SOOOA_HOME/data/sdf/plantynet_train_0.25/";
	int cnt = 0;

	/*
	DataReader<Datum> dr(source);
	int numData = dr.getNumData();
	cout << "numData: " << numData << endl;

	while (true) {
		Datum* datum = dr.getNextData();
		delete datum;
		if (++cnt % 1000 == 0) {
			STDOUT_LOG("Processed %d images.", cnt);
		}
	}
	*/

	SDF db(source, Mode::READ);
	db.open();

	string value = db.getNextValue();
	int numData = atoi(value.c_str());

	while (true) {
		string value = db.getNextValue();
		Datum* datum = new Datum();
		//T::deserializeFromString(value, datum);
		deserializeFromString(value, datum);
		delete datum;
		if (++cnt % 1000 == 0) {
			STDOUT_LOG("Processed %d images.", cnt);
		}
	}
}

bool readKeywords(const string& keywordPath, vector<string>& keywordList) {
	if (keywordPath.empty()) {
		return false;
	}

	ifstream infile(keywordPath);
	string line;
	keywordList.clear();

	while (std::getline(infile, line)) {
		keywordList.push_back(line);
	}

	if (keywordList.size() < 1) {
		return false;
	} else {
		return true;
	}
}

void runNetwork() {
	cout << "runNetwork ... " << endl;

	int gpuid = 0;
	initializeNetwork();
	NetworkTestInterface<float>::globalSetUp(gpuid);

	int networkID = PlanParser::loadNetwork("/home/jkim/Dev/SOOOA_HOME/network_def/data_input_network.json");
	const string keywordPath = "/home/jkim/Dev/data/image/ESP-ImageSet/keywordList.txt";
	vector<string> keywordList;
	bool hasKeyword = readKeywords(keywordPath, keywordList);


	Network<float>* network = Network<float>::getNetworkFromID(networkID);
	network->build(100);

	WorkContext::updateNetwork(networkID);
	WorkContext::updatePlan(0, true);

	PhysicalPlan* pp = WorkContext::curPhysicalPlan;


	Layer<float>* inputLayer = network->findLayer("data");
	//SASSERT0(dynamic_cast<DataInputLayer<float>*>(inputLayer));
	SASSERT0(dynamic_cast<MultiLabelDataInputLayer<float>*>(inputLayer));


	const string windowName = "result";
	cv::namedWindow(windowName);

	for (int i = 0; i < 100; i++) {
		cout << i << "th iteration" << endl;
		network->runPlanType(PlanType::PLANTYPE_FORWARD, true);
		network->reset();



		// has label ...
		if (inputLayer->_outputData.size() > 1) {
			Data<float>* label = inputLayer->_outputData[1];

			/*
			Data<float>::printConfig = true;
			SyncMem<float>::printConfig = true;
			label->print_data_flatten();
			Data<float>::printConfig = false;
			SyncMem<float>::printConfig = false;
			*/
			const int numLabels = label->getShape(1);
			// single label
			if (numLabels == 1) {
				int labelIdx = (int)label->host_data()[i];
				if (hasKeyword) {
					cout << "label: " << labelIdx << " (" << keywordList[labelIdx] << ")" << endl;
				} else {
					cout << "label: " << labelIdx << endl;
				}
			}
			else if (numLabels > 1) {
				cout << "---------" << endl;
				for (int i = 0; i < numLabels; i++) {
					if ((int)label->host_data()[i] > 0) {
						if (hasKeyword) {
							cout << "label: " << i << " (" << keywordList[i] << ")" << endl;
						} else {
							cout << "label: " << i << endl;
						}
					}
				}
				cout << "---------" << endl;
			}
		}

		Data<float>* data = inputLayer->_outputData[0];
		data->print_shape();

		int height = data->getShape(2);
		int width = data->getShape(3);
		int channels = data->getShape(1);

		if (channels == 1) {
			cv::Mat cv_img(height, width, CV_32F, data->mutable_host_data());
			cv_img.convertTo(cv_img, CV_8U);
			cv::imshow(windowName, cv_img);
			cv::waitKey(0);
		} else if (channels = 3) {
			data->transpose({0, 2, 3, 1});
			cv::Mat cv_img(height, width, CV_32FC3, data->mutable_host_data());
			cv_img.convertTo(cv_img, CV_8UC3);
			cv::imshow(windowName, cv_img);
			data->transpose({0, 3, 1, 2});
			cv::waitKey(0);
		}
	}

	cv::destroyWindow(windowName);
	NetworkTestInterface<float>::globalCleanUp();

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

#if 0
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
#define NETWORK_VGG16		6
#define NETWORK				NETWORK_VGG16

#define EXAMPLE_LENET_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/git/soooa/SoooA/src/examples/LeNet/lenet_train_test.json")
#define EXAMPLE_FRCNN_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/git/soooa/SoooA/src/examples/frcnn/frcnn_train_test.json")
#define EXAMPLE_VGG16_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/git/soooa/SoooA/src/examples/VGG16/vgg16_train_test.json")
#define EXAMPLE_DUMMY_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/SOOOA_HOME/network_def/data_input_network.json")


//void saveNetworkParams(LayersConfig<float>* layersConfig);







void networkTest() {
	const int gpuid = 0;
	initializeNetwork();


#if NETWORK == NETWORK_LENET
	const string networkFilePath = string(EXAMPLE_LENET_TRAIN_NETWORK_FILEPATH);
	const string networkName = "lenet";
	const int numSteps = 3;
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
	const string networkFilePath = string(EXAMPLE_FRCNN_TRAIN_NETWORK_FILEPATH);
	const string networkName = "frcnn";
	const int numSteps = 2;

	/*
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
	*/
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
#elif NETWORK == NETWORK_VGG16
	const string networkFilePath = string(EXAMPLE_VGG16_TRAIN_NETWORK_FILEPATH);
	const string networkName = "vgg16";
	const int numSteps = 1;
#else
	cout << "invalid network ... " << endl;
	exit(1);
#endif



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

	NetworkTestInterface<float>::globalSetUp(gpuid);

	NetworkTest<float>* networkTest =
			new NetworkTest<float>(networkFilePath, networkName, numSteps);


	//networkTest->setUp();
	//networkTest->updateTest();
	//networkTest->cleanUp();

	NetworkTestInterface<float>::globalCleanUp();
}



void saveNetwork() {
	const int gpuid = 0;
	initializeNetwork();

	const string networkFilePath = string("/home/jkim/Dev/git/soooa/SoooA/src/examples/VGG16/vgg16_train.json");
	const string networkName = "vgg16";
	const int numSteps = 0;

#if 0
	Network<float>* network = new Network<float>(networkConfig);
	// (2) network config 정보를 layer들에게 전달한다.
	for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->setNetworkConfig(network->config);
	}
	network->setLayersConfig(layersConfig);
	network->loadPretrainedWeights();
#endif

	NetworkTestInterface<float>::globalSetUp(gpuid);

	NetworkTest<float>* networkTest =
			new NetworkTest<float>(networkFilePath, networkName, numSteps);

	networkTest->setUp();

	Network<float>* network = networkTest->network;
	//network->load("/home/jkim/Dev/SOOOA_HOME/VGG16_CAFFE_TRAINED.param");
	network->save("/home/jkim/Dev/SOOOA_HOME/VGG16_CAFFE_TRAINED.param");



	NetworkTestInterface<float>::globalCleanUp();
}


#if 0
void saveNetworkParams(LayersConfig<float>* layersConfig) {
	const string savePathPrefix = "/home/jkim/Dev/SOOOA_HOME/param";
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
