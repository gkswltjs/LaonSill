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
#include "DataTransformer.h"
#include "LayerPropParam.h"
//#include "jsoncpp/json/json.h"
#include "MathFunctions.h"
#include "ImTransforms.h"
#include "AnnotatedDataLayer.h"

#include <execinfo.h>



using namespace std;
namespace fs = ::boost::filesystem;






void plainTest(int argc, char** argv);

void dataReaderTest(int argc, char** argv);
void annoDataReaderTest(int argc, char** argv);
void dataReaderMemoryLeakTest();
void runNetwork();
void jsonTest();
void dataTransformerTest();
void imTransformTest();
void readAnnoDataSetTest();
void randTest(int argc, char** argv);
void signalTest();

void layerTest(int argc, char** argv);
void layerTest2(int argc, char** argv);
void networkTest(int argc, char** argv);
void saveNetwork();




void testJsonType(Json::Value& value) {
	cout << value.type() << endl;
}


#if 0
int main(int argc, char** argv) {
	cout << "begin test ... " << endl;
	cout.precision(10);
	cout.setf(ios::fixed);

	//plainTest(argc, argv);
	//layerTest(argc, argv);
	networkTest(argc, argv);
	//saveNetwork();
	//layerTest2(argc, argv);

	cout << "end test ... " << endl;
	return 0;
}
#endif




void jsonTest() {
	const string filePath = "/home/jkim/Dev/git/soooa/SoooA/src/examples/SSD/ssd_multiboxloss_test.json";

	filebuf fb;
	if (fb.open(filePath.c_str(), ios::in) == NULL) {
		SASSERT(false, "cannot open cluster configuration file. file path=%s",
			filePath.c_str());
	}

	Json::Value rootValue;
	istream is(&fb);
	Json::Reader reader;
	bool parse = reader.parse(is, rootValue);

	if (!parse) {
		SASSERT(false, "invalid json-format file. file path=%s. error message=%s",
			filePath.c_str(), reader.getFormattedErrorMessages().c_str());
	}



	stringstream softmaxDef;
	softmaxDef << "{\n";
	softmaxDef << "\t\"name\" : \"inner_softmax\",\n";
	softmaxDef << "\t\"id\" : 7001,\n";
	softmaxDef << "\t\"layer\" : \"Softmax\",\n";
	softmaxDef << "\t\"input\" : [\"inner_softmax_7001_input\"],\n";
	softmaxDef << "\t\"output\" : [\"inner_softmax_7001_output\"],\n";
	softmaxDef << "\t\"softmaxAxis\" : 2\n";
	softmaxDef << "}\n";

	cout << softmaxDef.str() << endl;
	Json::Value tempValue;
	reader.parse(softmaxDef, tempValue);

	cout << tempValue["output"] << endl;
	testJsonType(tempValue);


	/*
	string stmt1 = "";
	stmt1 += "{";
	stmt1 += "    'name' : 'data'";
	stmt1 += "}";

	Json::Value v1(stmt1);
	cout << v1 << endl;
	testJsonType(v1);

	string stmt2 = "";
	stmt2 += "'data'";

	Json::Value v2(stmt2);
	cout << v2 << endl;
	testJsonType(v2);


	string stmt3 = "";
	stmt3 += "    'name' : 'data'";

	Json::Value v3(stmt3);
	cout << v3 << endl;
	testJsonType(v3);

	Json::Value v4(4);
	testJsonType(v4);




	Json::Value tempValue;
	istringstream iss(stmt1);
	reader.parse(iss, tempValue);
	testJsonType(tempValue);
	*/



	/*
	Json::Value layerList = rootValue["layers"];
	for (int i = 0; i < layerList.size(); i++) {
		Json::Value layer = layerList[i];
		vector<string> keys = layer.getMemberNames();

		for (int j = 0; j < keys.size(); j++) {
			string key = keys[j];
			Json::Value val = layer[key.c_str()];

			cout << "val=" << val << endl;

		}
		cout << "--------------------------------------------" << endl;
	}
	*/

}


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
	//dataTransformerTest();
	//imTransformTest();
	//annoDataReaderTest(argc, argv);
	//randTest(argc, argv);
	//signalTest();
}


void signalTest() {
	void *trace[16];
	char **messages = (char **) NULL;
	int i, trace_size = 0;

	trace_size = backtrace(trace, 16);
	/* overwrite sigaction with caller's address */
	messages = backtrace_symbols(trace, trace_size);
	/* skip first stack frame (points here) */
	printf("[bt] Execution path:\n");
	for (i = 1; i < trace_size; ++i) {
		printf("[bt] #%d %s\n", i, messages[i]);

		/* find first occurence of '(' or ' ' in message[i] and assume
		 * everything before that is the file name. (Don't go beyond 0 though
		 * (string terminator)*/
		size_t p = 0;
		while (messages[i][p] != '(' && messages[i][p] != ' '
				&& messages[i][p] != 0)
			++p;

		char syscom[256];
		sprintf(syscom, "addr2line %p -e %.*s", trace[i], p, messages[i]);
		//last parameter is the file name of the symbol
		system(syscom);
	}
}



void randTest(int argc, char** argv) {

	for (int i = 0; i < 100; i++) {
		soooa_rng_rand();
	}

	//float r;
	//soooa_rng_uniform(1, 0.f, 1.f, &r);
}


void imTransformTest() {
	const string windowName = "result";

	DistortionParam distortParam;
	//distortParam.brightnessProb = 1.f;
	//distortParam.brightnessDelta = 100.f;
	//distortParam.contrastProb = 1.f;
	//distortParam.contrastLower = 0.5f;
	//distortParam.contrastUpper = 2.0f;
	//distortParam.saturationProb = 1.f;
	//distortParam.saturationLower = 0.5f;
	//distortParam.saturationUpper = 2.0f;
	//distortParam.hueProb = 1.0f;
	//distortParam.hueDelta = 100.0f;
	distortParam.randomOrderProb = 1.0f;


	for (int i  = 0; i < 100; i++) {
		cv::Mat im = cv::imread("/home/jkim/Downloads/sample.jpg");
		const int height = im.rows;
		const int width = im.cols;
		const int channels = im.channels();

		im = ApplyDistort(im, distortParam);

		cv::imshow("result", im);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
}


void dataTransformerTest() {
	const string windowName = "result";
	DataTransformParam dtp;
	//dtp.mirror = flase;

	DistortionParam distortParam;
	distortParam.brightnessProb = 1.f;
	distortParam.brightnessDelta = 100.f;

	dtp.distortParam = distortParam;


	DataTransformer<float> dt(&dtp);

	cv::Mat im = cv::imread("/home/jkim/Downloads/sample.jpg");
	const int height = im.rows;
	const int width = im.cols;
	const int channels = im.channels();

	//im.convertTo(im, CV_32FC3);
	//cv::resize(im, im, cv::Size(300, 300), 0, 0, CV_INTER_LINEAR);

	Data<float> data("data", true);
	data.reshape({1, (uint32_t)channels, (uint32_t)height, (uint32_t)width});

	dt.transform(im, &data, 0);


	Data<float> temp("temp");
	temp.reshapeLike(&data);
	const int singleImageSize = data.getCount();
	transformInv(1, singleImageSize, height, width, height, width, {0.f, 0.f, 0.f},
			data.host_data(), temp);
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


void annoDataReaderTest(int argc, char** argv) {
	DataReader<AnnotatedDatum> dr("/home/jkim/Dev/SOOOA_HOME/data/sdf/voc2007_train_sdf/");
	int numData = dr.getNumData();
	cout << "numData: " << numData << endl;

	const string windowName = "result";
	cv::namedWindow(windowName);

	for (int i = 0; i < std::min(numData, 100); i++) {
		AnnotatedDatum* datum = dr.getNextData();
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



void layerTest(int argc, char** argv) {
	initializeNetwork();

	const char* soooaHome = std::getenv("SOOOA_DEV_HOME");

	const int gpuid = 0;
	const string networkName = "ssd";
	const string networkFilePath = string(soooaHome) +
			string("/src/examples/SSD/ssd_annotateddata_test.json");
	const string targetLayerName = "data";

	const int numAfterSteps = 10;
	const int numSteps = 10;
	const NetworkStatus status = NetworkStatus::Test;

	LayerTestInterface<float>::globalSetUp(gpuid);
	LayerTestInterface<float>* layerTest = new LayerTest<float>(networkFilePath,
			networkName, targetLayerName, numSteps, numAfterSteps, status);
	cout << "LayerTest initialized ... " << endl;

	layerTest->setUp();
	cout << "setUp completed ... " << endl;

	layerTest->forwardTest();

	cout << "forwardTest completed ... " << endl;
	if (status == NetworkStatus::Train) {
		layerTest->backwardTest();
	}

	layerTest->cleanUp();
	LayerTestInterface<float>::globalCleanUp();
}

void layerTest2(int argc, char** argv) {
	const char* soooaHome = std::getenv("SOOOA_DEV_HOME");
	std::ifstream layerDefJsonStream(string(soooaHome) + string("/src/test/annotated_data.json"));
	SASSERT0(layerDefJsonStream);
	std::stringstream annotateddataDef;
	annotateddataDef << layerDefJsonStream.rdbuf();
	layerDefJsonStream.close();
	cout << annotateddataDef.str() << endl;

	_AnnotatedDataPropLayer* prop = new _AnnotatedDataPropLayer();

	Json::Reader reader;
	Json::Value layer;
	reader.parse(annotateddataDef, layer);

	vector<string> keys = layer.getMemberNames();
	string layerType = layer["layer"].asCString();

	for (int j = 0; j < keys.size(); j++) {
		string key = keys[j];
		Json::Value val = layer[key.c_str()];
		if (strcmp(key.c_str(), "layer") == 0) continue;
		if (strcmp(key.c_str(), "innerLayer") == 0) continue;

		PlanParser::setPropValue(val, true, layerType, key,  (void*)prop);
	}

	AnnotatedDataLayer<float>* l = new AnnotatedDataLayer<float>(prop);
	Data<float> data("data");
	Data<float> label("label");
	l->_outputData.push_back(&data);
	l->_outputData.push_back(&label);

	l->reshape();
	l->feedforward();
}



#define NETWORK_LENET		0
#define NETWORK_VGG19		1
#define NETWORK_FRCNN		2
#define NETWORK_FRCNN_TEST	3
#define NETWORK_SSD			4
#define NETWORK_SSD_TEST	5
#define NETWORK_VGG16		6
#define NETWORK				NETWORK_SSD

#define EXAMPLE_LENET_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/git/soooa/SoooA/src/examples/LeNet/lenet_train_test.json")
#define EXAMPLE_FRCNN_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/git/soooa/SoooA/src/examples/frcnn/frcnn_train_test.json")
#define EXAMPLE_VGG16_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/git/soooa/SoooA/src/examples/VGG16/vgg16_train_test.json")
#define EXAMPLE_DUMMY_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/SOOOA_HOME/network_def/data_input_network.json")


//void saveNetworkParams(LayersConfig<float>* layersConfig);







void networkTest(int argc, char** argv) {
	const char* soooaHome = std::getenv("SOOOA_DEV_HOME");
	const int gpuid = 0;
	initializeNetwork();

#if NETWORK == NETWORK_SSD
	const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_300_train_test.json");
	//const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_300_train.json");
	//const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_300_infer_test.json");
	//const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_512_train_test.json");
	//const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_512_infer_test.json");
	const string networkName = "ssd";
	const int numSteps = 1;
	const NetworkStatus status = NetworkStatus::Train;
#elif NETWORK == NETWORK_VGG16
#else
	cout << "invalid network ... " << endl;
	exit(1);
#endif
	NetworkTestInterface<float>::globalSetUp(gpuid);

	NetworkTest<float>* networkTest =
			new NetworkTest<float>(networkFilePath, networkName, numSteps, status);

	networkTest->setUp();
	networkTest->updateTest();
	//Network<float>* network = networkTest->network;
	//network->save("/home/jkim/Dev/SOOOA_HOME/param/VGG_ILSVRC_16_layers_fc_reduced_SSD_300x300.param");
	networkTest->cleanUp();

	NetworkTestInterface<float>::globalCleanUp();
	cout << "networkTest() Done ... " << endl;
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
