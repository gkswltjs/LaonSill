#if 1

#include "cnpy.h"
#include "Data.h"
#include "Cuda.h"

#include "ConvLayer.h"

using namespace std;
using namespace cnpy;



const string TYPE_DATA = "data";
const string TYPE_DIFF = "diff";
const string SIG_BOTTOM = "_bottom_";
const string SIG_TOP = "_top_";
const string SIG_PARAMS = "_params_";


void buildNameDataMapFromNpzFile(const string& npz_path, const string& layer_name,
		map<string, Data<float>*>& nameDataMap);
const vector<uint32_t> getShape(NpyArray& npyArray);
const string getDataKeyFromDataName(const string& data_name);
const string getDataTypeFromDataName(const string& data_name);
template <typename T, typename S>
bool hasKey(map<T, S*>& dict, const T& key);
template <typename T, typename S>
S* retrieveValueFromMap(map<T, S*>& dict, const T& key);
void printNpzFiles(npz_t& cnpy_npz);
void printNameDataMap(map<string, Data<float>*>& nameDataMap, bool printData);
ConvLayer<float>* buildConvLayer(const string& layer_name);
void fillLayerDataVec(const vector<string>& dataNameVec, vector<Data<float>*>& dataVec);
void testForward(ConvLayer<float>* layer, map<string, Data<float>*>& nameDataMap);
void testBackward(ConvLayer<float>* layer, map<string, Data<float>*>& nameDataMap);
void fillData(map<string, Data<float>*>& nameDataMap, const string& data_prefix,
		vector<Data<float>*>& dataVec);
void fillParam(map<string, Data<float>*>& nameDataMap, const string& param_prefix,
		vector<Data<float>*>& paramVec);
void compareData(map<string, Data<float>*>& nameDataMap, const string& data_prefix,
		vector<Data<float>*>& dataVec, uint32_t compareType);
void setupCuda();
void cleanupCuda();
void printData(vector<Data<float>*>& dataVec);
void printConfigOn();
void printConfigOff();




int main(void) {
	cout.setf(ios::fixed);
	cout.precision(9);

	const string npz_path = "/home/jkim/Dev/data/numpy_array/";
	const string layer_name = "conv2";
	map<string, Data<float>*> nameDataMap;

	buildNameDataMapFromNpzFile(npz_path, layer_name, nameDataMap);
	printNameDataMap(nameDataMap, false);

	ConvLayer<float>* convLayer = buildConvLayer(layer_name);

	setupCuda();

	testForward(convLayer, nameDataMap);
	//testBackward(convLayer, nameDataMap);

	cleanupCuda();

	return 0;
}




void buildNameDataMapFromNpzFile(const string& npz_path, const string& layer_name,
		map<string, Data<float>*>& nameDataMap) {

	const string npz_file = npz_path + layer_name + ".npz";
	npz_t cnpy_npz = npz_load(npz_file);
	printNpzFiles(cnpy_npz);

	for (npz_t::iterator itr = cnpy_npz.begin(); itr != cnpy_npz.end(); itr++) {
		string data_name = itr->first;
		NpyArray npyArray = itr->second;

		const string data_key = getDataKeyFromDataName(data_name);
		const string data_type = getDataTypeFromDataName(data_name);

		cout << "for " << data_name << ": data_key-" << data_key << ", data_type-" <<
				data_type << endl;

		Data<float>* data = retrieveValueFromMap(nameDataMap, data_key);
		if (!data) {
			data = new Data<float>(data_key);
			const vector<uint32_t> shape = getShape(npyArray);
			data->reshape(shape);
			nameDataMap[data_key] = data;
		}

		if (data_type == TYPE_DATA) {
			data->set_host_data((float*)npyArray.data);
		} else if (data_type == TYPE_DIFF) {
			data->set_host_grad((float*)npyArray.data);
		}
	}
}


const vector<uint32_t> getShape(NpyArray& npyArray) {
	vector<uint32_t> shape(4);
	const uint32_t shapeSize = npyArray.shape.size();

	for (uint32_t i = 0; i < 4 - shapeSize; i++) {
		shape[i] = 1;
	}
	for (uint32_t i = 4 - shapeSize; i < 4; i++) {
		shape[i] = npyArray.shape[i - (4 - shapeSize)];
		if (shape[i] == 0)
			shape[i] = 1;
	}
	return shape;
}

const string getDataKeyFromDataName(const string& data_name) {
	assert(data_name.length() > 5);
	return data_name.substr(0, data_name.length() - 5);
}

const string getDataTypeFromDataName(const string& data_name) {
	assert(data_name.length() > 5);
	const string data_type = data_name.substr(data_name.length() - 4);
	assert(data_type == TYPE_DATA || data_type == TYPE_DIFF);
	return data_type;
}



template <typename T, typename S>
bool hasKey(map<T, S*>& dict, const T& key) {
	typename map<T, S*>::iterator itr = dict.find(key);
	return (itr != dict.end());
}

template <typename T, typename S>
S* retrieveValueFromMap(map<T, S*>& dict, const T& key) {
	typename map<T, S*>::iterator itr = dict.find(key);
	if (itr == dict.end()) {
		return 0;
	} else
		return itr->second;
}


void printNpzFiles(npz_t& cnpy_npz) {
	cout << "<npz_t array list>----------------" << endl;
	for (npz_t::iterator itr = cnpy_npz.begin(); itr != cnpy_npz.end(); itr++) {
		std::cout << itr->first << std::endl;
	}
	cout << "----------------------------------" << endl;
}

void printNameDataMap(map<string, Data<float>*>& nameDataMap, bool printData) {
	printConfigOn();

	for (typename map<string, Data<float>*>::iterator itr = nameDataMap.begin();
			itr != nameDataMap.end(); itr++) {

		if (printData) {
			itr->second->print_data({}, false);
			itr->second->print_grad({}, false);
		} else
			cout << itr->first << endl;
	}

	printConfigOff();
}



ConvLayer<float>* buildConvLayer(const string& layer_name) {
	ConvLayer<float>::Builder* builder = new typename ConvLayer<float>::Builder();
	builder
	->id(1)
	->name(layer_name)
	->filterDim(5, 5, 3, 5, 0, 1)
	->inputs({"pool1"})
	->outputs({"conv2"});

	ConvLayer<float>* layer = dynamic_cast<ConvLayer<float>*>(builder->build());

	fillLayerDataVec(layer->_inputs, layer->_inputData);
	fillLayerDataVec(layer->_outputs, layer->_outputData);

	return layer;
}

void fillLayerDataVec(const vector<string>& dataNameVec, vector<Data<float>*>& dataVec) {
	dataVec.clear();
	for (uint32_t i = 0; i < dataNameVec.size(); i++) {
		Data<float>* data = new Data<float>(dataNameVec[i]);
		dataVec.push_back(data);
	}
}


void testForward(ConvLayer<float>* layer, map<string, Data<float>*>& nameDataMap) {
	fillData(nameDataMap, layer->name + SIG_BOTTOM, layer->_inputData);
	fillParam(nameDataMap, layer->name + SIG_PARAMS, layer->_params);

	//printData(layer->_inputData);
	//printData(layer->_params);

	layer->feedforward();

	//printData(layer->_outputData);

	compareData(nameDataMap, layer->name + SIG_TOP, layer->_outputData, 0);
}


void testBackward(ConvLayer<float>* layer, map<string, Data<float>*>& nameDataMap) {
	fillData(nameDataMap, layer->name + SIG_BOTTOM, layer->_inputData);
	fillData(nameDataMap, layer->name + SIG_TOP, layer->_outputData);
	fillParam(nameDataMap, layer->name + SIG_PARAMS, layer->_params);

	printData(layer->_inputData);
	printData(layer->_outputData);
	printData(layer->_params);

	layer->backpropagation();

	printData(layer->_inputData);

	compareData(nameDataMap, layer->name + SIG_BOTTOM, layer->_inputData, 1);
}



void fillData(map<string, Data<float>*>& nameDataMap, const string& data_prefix,
		vector<Data<float>*>& dataVec) {

	for (uint32_t i = 0; i < dataVec.size(); i++) {
		const string dataName = dataVec[i]->_name;
		const string key = data_prefix + dataName;

		Data<float>* data = retrieveValueFromMap(nameDataMap, key);
		assert(data != 0);

		Data<float>* targetData = dataVec[i];
		targetData->set(data, true);
	}
}

void fillParam(map<string, Data<float>*>& nameDataMap, const string& param_prefix,
		vector<Data<float>*>& paramVec) {

	for (uint32_t i = 0; i < paramVec.size(); i++) {
		const string key = param_prefix + to_string(i);

		Data<float>* param = retrieveValueFromMap(nameDataMap, key);
		assert(param != 0);

		Data<float>* targetParam = paramVec[i];
		targetParam->set(param, true);
	}
}

void compareData(map<string, Data<float>*>& nameDataMap, const string& data_prefix,
		vector<Data<float>*>& dataVec, uint32_t compareType) {

	for (uint32_t i = 0; i < dataVec.size(); i++) {
		const string dataName = dataVec[i]->_name;
		const string key = data_prefix + dataName;

		Data<float>* data = retrieveValueFromMap(nameDataMap, key);
		assert(data != 0);

		Data<float>* targetData = dataVec[i];


		printConfigOn();
		data->print_data({}, false);
		targetData->print_data({}, false);
		printConfigOff();

		if (compareType == 0)
			assert(targetData->compareData(data));
		else
			assert(targetData->compareGrad(data));
	}
}

void setupCuda() {
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));
}

void cleanupCuda() {
	checkCudaErrors(cublasDestroy(Cuda::cublasHandle));
	checkCUDNN(cudnnDestroy(Cuda::cudnnHandle));
}

void printData(vector<Data<float>*>& dataVec) {
	printConfigOn();

	for (uint32_t i = 0; i < dataVec.size(); i++) {
		dataVec[i]->print_data({}, false);
		dataVec[i]->print_grad({}, false);
	}

	printConfigOff();
}

void printConfigOn() {
	Data<float>::printConfig = true;
	SyncMem<float>::printConfig = true;
}

void printConfigOff() {
	Data<float>::printConfig = false;
	SyncMem<float>::printConfig = false;
}


#endif
