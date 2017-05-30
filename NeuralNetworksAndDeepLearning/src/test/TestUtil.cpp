#include "TestUtil.h"
#include "Cuda.h"
#include "Layer.h"
#include "LearnableLayer.h"
#include "NetworkConfig.h"

#if 0

using namespace std;
using namespace cnpy;


const vector<uint32_t> getShape(const string& data_key, NpyArray& npyArray);
const string getDataKeyFromDataName(const string& data_name);
const string getDataTypeFromDataName(const string& data_name);
void Tokenize(const string& str, vector<string>& tokens, const string& delimiters = " ", bool print = false);

template <typename T, typename S>
bool hasKey(map<T, S*>& dict, const T& key);






void setUpCuda(int gpuid) {
	assert(gpuid >= 0);

	checkCudaErrors(cudaSetDevice(gpuid));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));
}

void cleanUpCuda() {
	checkCudaErrors(cublasDestroy(Cuda::cublasHandle));
	checkCUDNN(cudnnDestroy(Cuda::cudnnHandle));
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
			data = new Data<float>(data_key, false);
			const vector<uint32_t> shape = getShape(data_key, npyArray);
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




const vector<uint32_t> getShape(const string& data_key, NpyArray& npyArray) {
	vector<uint32_t> shape(4);
	const uint32_t shapeSize = npyArray.shape.size();

	/*
	if (shapeSize == 4)
		shape = npyArray.shape;
	else if (shapeSize == 2) {

	}
	*/


	vector<string> tokens;
	Tokenize(data_key, tokens, "*", true);
	assert(tokens.size() == 3);

	if (tokens[1] == "params") {
		for (uint32_t i = 0; i < 4 - shapeSize; i++) {
			shape[i] = 1;
		}
		for (uint32_t i = 4 - shapeSize; i < 4; i++) {
			shape[i] = npyArray.shape[i - (4 - shapeSize)];
			if (shape[i] == 0)
				shape[i] = 1;
		}
	} else if (tokens[1] == "bottom" || tokens[1] == "top" || tokens[1] == "blobs") {
		assert(shapeSize == 1 || shapeSize == 2 || shapeSize == 4);
		if (shapeSize == 1) {
			shape[0] = npyArray.shape[0];
			shape[1] = 1;
			shape[2] = 1;
			shape[3] = 1;
		} else if (shapeSize == 2) {
			shape[0] = npyArray.shape[0];
			shape[1] = 1;
			shape[2] = npyArray.shape[1];
			shape[3] = 1;
		} else if (shapeSize == 4) {
			shape = npyArray.shape;
		}
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
template bool hasKey(map<string, Data<float>*>& dict, const string& key);

template <typename T, typename S>
S* retrieveValueFromMap(map<T, S*>& dict, const T& key) {
	typename map<T, S*>::iterator itr = dict.find(key);
	if (itr == dict.end()) {
		return 0;
	} else
		return itr->second;
}
template Data<float>* retrieveValueFromMap(map<string, Data<float>*>& dict,
		const string& key);


template <typename T, typename S>
void cleanUpMap(map<T, S*>& dict) {
	typename map<T, S*>::iterator itr;
	for (itr = dict.begin(); itr != dict.end(); itr++) {
		if (itr->second)
			delete itr->second;
	}
}
template void cleanUpMap(map<string, Data<float>*>& dict);

template <typename T>
void cleanUpObject(T* obj) {
	if (obj)
		delete obj;
}
template void cleanUpObject(Layer<float>* obj);
template void cleanUpObject(Layer<float>::Builder* obj);
template void cleanUpObject(LearnableLayer<float>* obj);
template void cleanUpObject(LearnableLayer<float>::Builder* obj);
template void cleanUpObject(LayersConfig<float>* obj);




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



void fillLayerDataVec(const vector<string>& dataNameVec, vector<Data<float>*>& dataVec) {
	dataVec.clear();
	for (uint32_t i = 0; i < dataNameVec.size(); i++) {
		Data<float>* data = new Data<float>(dataNameVec[i]);
		dataVec.push_back(data);
	}
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

		cout << "fill param key: " << key << endl;
		Data<float>* param = retrieveValueFromMap(nameDataMap, key);
		assert(param != 0);

		Data<float>* targetParam = paramVec[i];
		targetParam->set(param, true);


		/*
		printConfigOn();
		param->print_data({}, false);
		targetParam->print_data({}, false);
		printConfigOff();
		*/
	}
}


bool compareData(map<string, Data<float>*>& nameDataMap, const string& data_prefix,
		vector<Data<float>*>& dataVec, uint32_t compareType) {

	/*
	bool final_result = true;
	for (uint32_t i = 0; i < dataVec.size(); i++) {
		const string dataName = dataVec[i]->_name;
		const string key = data_prefix + dataName;

		Data<float>* data = retrieveValueFromMap(nameDataMap, key);
		assert(data != 0);

		Data<float>* targetData = dataVec[i];

		bool partial_result = false;
		if (compareType == 0)
			partial_result = targetData->compareData(data, COMPARE_ERROR);
		else
			partial_result = targetData->compareGrad(data, COMPARE_ERROR);

		if (!partial_result) {
			printConfigOn();
			if (compareType == 0) {
				data->print_data({}, false);
				targetData->print_data({}, false);
			} else {
				data->print_grad({}, false);
				targetData->print_grad({}, false);
			}
			printConfigOff();
		}
		final_result = final_result && partial_result;
	}
	return final_result;
	*/

	bool final_result = true;
	for (uint32_t i = 0; i < dataVec.size(); i++) {
		Data<float>* targetData = dataVec[i];
		bool partial_result = compareData(nameDataMap, data_prefix, targetData, compareType);

		final_result = final_result && partial_result;
	}
	return final_result;
}

bool compareData(map<string, Data<float>*>& nameDataMap, const string& data_prefix,
		Data<float>* targetData, uint32_t compareType) {
	const string dataName = targetData->_name;
	const string key = data_prefix + dataName;

	Data<float>* data = retrieveValueFromMap(nameDataMap, key);
	assert(data != 0);

	bool result = false;
	if (compareType == 0)
		result = targetData->compareData(data, COMPARE_ERROR);
	else
		result = targetData->compareGrad(data, COMPARE_ERROR);

	if (!result) {
		//printConfigOn();
		if (compareType == 0) {
			data->print_data({}, false);
			targetData->print_data({}, false);
		} else {
			data->print_grad({}, false);
			targetData->print_grad({}, false);
		}
		printConfigOff();
	}
	return result;
}






bool compareParam(map<string, Data<float>*>& nameDataMap, const string& param_prefix,
		vector<Data<float>*>& paramVec, uint32_t compareType) {

	for (uint32_t i = 0; i < paramVec.size(); i++) {
		const string key = param_prefix + to_string(i);
		Data<float>* data = retrieveValueFromMap(nameDataMap, key);
		assert(data != 0);

		Data<float>* targetData = paramVec[i];

		/*
		printConfigOn();
		if (compareType == 0) {
			data->print_data({}, false);
			targetData->print_data({}, false);
		} else {
			data->print_grad({}, false);
			targetData->print_grad({}, false);
		}
		printConfigOff();
		*/

		if (compareType == 0)
			//assert(targetData->compareData(data, COMPARE_ERROR));
			return targetData->compareData(data, COMPARE_ERROR);
		else
			//assert(targetData->compareGrad(data, COMPARE_ERROR));
			return targetData->compareGrad(data, COMPARE_ERROR);
	}
}


void Tokenize(const string& str, vector<string>& tokens, const string& delimiters, bool print) {
	// 맨 첫 글자가 구분자인 경우 무시
	string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	// 구분자가 아닌 첫 글자를 찾는다
	string::size_type pos = str.find_first_of(delimiters, lastPos);

	while (string::npos != pos || string::npos != lastPos) {
		// token을 찾았으니 vector에 추가한다
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		// 구분자를 뛰어넘는다.  "not_of"에 주의하라
		lastPos = str.find_first_not_of(delimiters, pos);
		// 다음 구분자가 아닌 글자를 찾는다
		pos = str.find_first_of(delimiters, lastPos);
	}

	if (print) {
		cout << "Tokenize Result------------------------" << endl;
		for (uint32_t i = 0; i < tokens.size(); i++) {
			cout << "'" << tokens[i] << "'" << endl;
		}
		cout << "---------------------------------" << endl;
	}
}















#endif


