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
//#include "jsoncpp/json/json.h"



using namespace std;
namespace fs = ::boost::filesystem;






void plainTest(int argc, char** argv);

void dataReaderTest(int argc, char** argv);
void dataReaderMemoryLeakTest();
void runNetwork();
void jsonTest();

void layerTest(int argc, char** argv);
void networkTest(int argc, char** argv);
void saveNetwork();




void testJsonType(Json::Value& value) {
	cout << value.type() << endl;
}


#if 0
int main(int argc, char** argv) {
	cout << "begin test ... " << endl;
	cout.precision(11);
	cout.setf(ios::fixed);

	//plainTest(argc, argv);
	//layerTest(argc, argv);
	networkTest(argc, argv);
	//saveNetwork();

	cout << "end test ... " << endl;
	return 0;
}
#endif




void jsonTest() {
	const string filePath = "/home/jkim/Dev/git/soooa/SoooA/src/examples/SSD/ssd_multiboxloss_test.json";

	filebuf fb;
	if (fb.open(filePath.c_str(), ios::in) == NULL) {
		SASSERT(false, "cannot open cluster confifuration file. file path=%s",
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
	cout << "SOOOA_DEV_HOME=" << soooaHome << endl;

	const int gpuid = 0;
	const string networkName = "ssd";
	//const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_permute_test.json");
	//const string targetLayerName = "conv4_3_norm_mbox_loc_perm";
	//const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_flatten_test.json");
	//const string targetLayerName = "conv4_3_norm_mbox_loc_flat";
	//const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_priorbox_test.json");
	//const string targetLayerName = "conv4_3_norm_mbox_priorbox";
	//const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_concat_test.json");
	//const string targetLayerName = "mbox_loc";
	//const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_normalize_test.json");
	//const string targetLayerName = "conv4_3_norm";
	//const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_multiboxloss_test.json");
	//const string targetLayerName = "mbox_loss";
	const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_detectionoutput_test.json");
	const string targetLayerName = "detection_out";
	const int numAfterSteps = 10;
	const NetworkStatus status = NetworkStatus::Test;

	LayerTestInterface<float>::globalSetUp(gpuid);
	LayerTestInterface<float>* layerTest = new LayerTest<float>(networkFilePath,
			networkName, targetLayerName, numAfterSteps, status);
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
	//const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_300_train_test.json");
	//const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_300_infer_test.json");
	//const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_512_train_test.json");
	const string networkFilePath = string(soooaHome) + string("/src/examples/SSD/ssd_512_infer_test.json");
	const string networkName = "ssd";
	const int numSteps = 1;
	const NetworkStatus status = NetworkStatus::Test;
#elif NETWORK == NETWORK_VGG16
#else
	cout << "invalid network ... " << endl;
	exit(1);
#endif
	NetworkTestInterface<float>::globalSetUp(gpuid);

	NetworkTest<float>* networkTest =
			new NetworkTest<float>(networkFilePath, networkName, numSteps, status);

	networkTest->setUp();
	//networkTest->updateTest();
	Network<float>* network = networkTest->network;
	network->save("/home/jkim/Dev/SOOOA_HOME/param/VGG_VOC0712_SSD_512x512_iter_120000.param");
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
