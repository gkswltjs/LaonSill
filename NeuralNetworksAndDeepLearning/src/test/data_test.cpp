
#include <cstdlib>
#include <fstream>
#include <map>

#include "Data.h"
#include "Util.h"
#include "Cuda.h"


#define DATA_TEST 0


void reshape_test();
void range_test();
void transpose_test();
void save_test();

#if DATA_TEST
int main() {
	//reshape_test();
	//range_test();
	//transpose_test();
	save_test();
}
#endif

void save_test() {
	Data<float>::printConfig = true;
	checkCudaErrors(cudaSetDevice(0));

	srand((uint32_t)time(NULL));

	// create random data list
	const uint32_t numData = 3;
	std::vector<Data<float>*> dataList;
	for (uint32_t i = 0; i < numData; i++) {
		Data<float>* data = new Data<float>("data"+std::to_string(i));
		data->reshape({2, 3, 2, 2});
		float* dataPtr = data->mutable_host_data();
		const uint32_t dataSize = data->getCount();
		for (uint32_t j = 0; j < dataSize; j++)
			dataPtr[j] = rand() % 1000;
		data->print_data({}, false);
		dataList.push_back(data);
	}

	// save data list to model file
	const std::string modelName = "/tmp/test.soooamodel";
	std::ofstream ofs(modelName.c_str(), std::ios::out | std::ios::binary);
	for (uint32_t i = 0; i < numData; i++) {
		dataList[i]->save(ofs);
		delete dataList[i];
	}
	dataList.clear();
	ofs.close();

	// load data list from model file
	std::map<std::string, Data<float>*> dataMap;
	std::ifstream ifs(modelName, std::ios::in | std::ios::binary);
	for (uint32_t i = 0; i < numData; i++) {
		Data<float>* data = new Data<float>("");
		data->load(ifs);
		dataMap[data->_name] = data;
	}

	// print loaded data list
	std::map<std::string, Data<float>*>::iterator it;
	for (it = dataMap.begin(); it != dataMap.end(); it++) {
		it->second->print_data({}, false);
		delete it->second;
	}
	dataMap.clear();
	Data<float>::printConfig = false;
}


void reshape_test() {
	const uint32_t batches = 1;
	const uint32_t channels = 5;
	const uint32_t height = 3;
	const uint32_t width = 3;

	Data<float>* data = new Data<float>("data");
	data->reshape({batches, channels, height, width});
	const uint32_t _0Size = data->getCountByAxis(1);
	const uint32_t _1Size = data->getCountByAxis(2);
	const uint32_t _2Size = data->getCountByAxis(3);

	float* dataPtr = data->mutable_host_data();
	for (uint32_t i = 0; i < batches; i++) {
		for (uint32_t j = 0; j < channels; j++) {
			for (uint32_t k = 0; k < height; k++) {
				for (uint32_t l = 0; l < height; l++) {
					dataPtr[i*_0Size + j*_1Size + k*_2Size + l] =
                        i*_0Size + j*_1Size + k*_2Size + l;
				}
			}
		}
	}

	Data<float>::printConfig = true;
	data->print_data({}, false);
	Data<float>::printConfig = false;

	data->reshape({1, 1, 5, 9});

	Data<float>::printConfig = true;
	data->print_data({}, false);
	Data<float>::printConfig = false;


}


void range_test() {
	const uint32_t batches = 1;
	const uint32_t channels = 5;
	const uint32_t height = 3;
	const uint32_t width = 3;

	Data<float>* data = new Data<float>("data");
	data->reshape({batches, channels, height, width});
	const uint32_t _0Size = data->getCountByAxis(1);
	const uint32_t _1Size = data->getCountByAxis(2);
	const uint32_t _2Size = data->getCountByAxis(3);

	float* dataPtr = data->mutable_host_data();
	for (uint32_t i = 0; i < batches; i++) {
		for (uint32_t j = 0; j < channels; j++) {
			for (uint32_t k = 0; k < height; k++) {
				for (uint32_t l = 0; l < height; l++) {
					dataPtr[i*_0Size + j*_1Size + k*_2Size + l] = float(j);
				}
			}
		}
	}

	Data<float>::printConfig = true;
	data->print_data();
	Data<float>::printConfig = false;

	Data<float>* rangedData = data->range({0, 3, 0, 0}, {-1, -1, -1, -1});
	Data<float>::printConfig = true;
	rangedData->print_data();
	Data<float>::printConfig = false;
}


void transpose_test() {
	const uint32_t batches = 1;
	const uint32_t channels = 5;
	const uint32_t height = 3;
	const uint32_t width = 3;

	Data<float>* data = new Data<float>("data");
	data->reshape({batches, channels, height, width});
	const uint32_t _0Size = data->getCountByAxis(1);
	const uint32_t _1Size = data->getCountByAxis(2);
	const uint32_t _2Size = data->getCountByAxis(3);

	float* dataPtr = data->mutable_host_data();
	for (uint32_t i = 0; i < batches; i++) {
		for (uint32_t j = 0; j < channels; j++) {
			for (uint32_t k = 0; k < height; k++) {
				for (uint32_t l = 0; l < height; l++) {
					dataPtr[i*_0Size + j*_1Size + k*_2Size + l] =
                        i*_0Size + j*_1Size + k*_2Size + l;
				}
			}
		}
	}

	Data<float>::printConfig = true;
	data->print_data({}, false);
	Data<float>::printConfig = false;

	data->transpose({0, 3, 1, 2});
	Data<float>::printConfig = true;
	data->print_data({}, false);
	Data<float>::printConfig = false;

}


















