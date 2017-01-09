/*
 * test_common.h
 *
 *  Created on: Dec 2, 2016
 *      Author: jkim
 */

#ifndef TEST_COMMON_H_
#define TEST_COMMON_H_


#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <cnpy.h>

#include "Data.h"

void create_cuda_handle() {
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cublasCreate(&Cuda::cublasHandle));
	checkCUDNN(cudnnCreate(&Cuda::cudnnHandle));
}

void destroy_cuda_handle() {
	checkCudaErrors(cublasDestroy(Cuda::cublasHandle));
	checkCUDNN(cudnnDestroy(Cuda::cudnnHandle));
}


template <typename Dtype>
void load_npz(
		const std::string& filename,
		const std::vector<std::string>& datanames,
		std::map<std::string, Data<Dtype>*>& datamap) {

	datamap.clear();
	cnpy::npz_t cnpy_npz = cnpy::npz_load(filename);

	std::cout << "data in npz file ---------------------" << std::endl;
	cnpy::npz_t::iterator tempItr;
	for (tempItr = cnpy_npz.begin(); tempItr != cnpy_npz.end(); tempItr++)
		std::cout << tempItr->first << std::endl;
	std::cout << "--------------------------------------" << std::endl;

	const uint32_t npzSize = cnpy_npz.size();
	assert(datanames.size() == npzSize/2);

	cnpy::npz_t::iterator it;
	for (uint32_t i = 0; i < npzSize; i++) {
		const std::string npz_dataname = "arr_" + std::to_string(i);

		it = cnpy_npz.find(npz_dataname);
		assert(it != cnpy_npz.end());

		cnpy::NpyArray npyArray = it->second;

		const std::string dataname = datanames[i/2];
		if (i % 2 == 0) {
			Data<float>* data = new Data<float>(dataname);
			std::vector<uint32_t> dataShape(4);
			for (uint32_t i = 0; i < 4 - npyArray.shape.size(); i++)
				dataShape[i] = 1;
			for (uint32_t i = 4-npyArray.shape.size(); i < 4; i++) {
				dataShape[i] = npyArray.shape[i-(4-npyArray.shape.size())];
				if (dataShape[i] == 0)
					dataShape[i] = 1;
			}

			data->reshape(dataShape);
			data->set_host_data((float*)npyArray.data);
			datamap[dataname] = data;
		} else {
			typename std::map<std::string, Data<Dtype>*>::iterator datamapItr;
			datamapItr = datamap.find(dataname);
			assert(datamapItr != datamap.end());
			Data<float>* data = datamapItr->second;
			data->set_host_grad((float*)npyArray.data);
		}

		Data<float>::printConfig = true;
		//data->print_data({}, false);
		Data<float>::printConfig = false;
	}

	/*
	for (uint32_t i = 0; i < datanames.size(); i++) {
		std::cout << "for data: " << datanames[i] << std::endl;

		cnpy::npz_t::iterator it = cnpy_npz.find(datanames[i]);
		assert(it != cnpy_npz.end());

		cnpy::NpyArray npyArray = it->second;

		Data<float>* data = new Data<float>(datanames[i]);
		std::vector<uint32_t> dataShape(4);
		for (uint32_t i = 0; i < 4 - npyArray.shape.size(); i++)
			dataShape[i] = 1;
		for (uint32_t i = 4-npyArray.shape.size(); i < 4; i++) {
			dataShape[i] = npyArray.shape[i-(4-npyArray.shape.size())];
		}

		data->reshape(dataShape);
		data->set_host_data((float*)npyArray.data);

		datamap[datanames[i]] = data;

		Data<float>::printConfig = true;
		//data->print_data({}, false);
		Data<float>::printConfig = false;
	}
	*/
}


template <typename Dtype>
void clean_data_list(std::vector<Data<Dtype>*>& dataList) {
	for (uint32_t i = 0; i < dataList.size(); i++)
		delete dataList[i];
	dataList.clear();
}

template <typename Dtype>
void set_layer_data(
		const std::vector<std::string>& dataNames,
		std::vector<Data<Dtype>*>& data) {
	clean_data_list(data);

	for (uint32_t i = 0; i < dataNames.size(); i++)
		data.push_back(new Data<Dtype>(dataNames[i]));
}


template <typename Dtype>
void set_layer_data(
		const std::vector<std::string>& dataNames,
		std::map<std::string, Data<Dtype>*>& dataMap,
		std::vector<Data<Dtype>*>& data) {
	clean_data_list(data);

	typename std::map<std::string, Data<Dtype>*>::iterator it;
	for (uint32_t i = 0; i < dataNames.size(); i++) {
		it = dataMap.find(dataNames[i]);
		assert(it != dataMap.end());

		Data<Dtype>* tempData = new Data<Dtype>(it->second);
		data.push_back(tempData);
	}
}





template <typename Dtype>
void compare_data(
		std::vector<Data<Dtype>*>& dataVector,
		std::map<std::string, Data<Dtype>*>& dataMap,
		const Dtype error = Dtype(0.001)) {

	typename std::map<std::string, Data<Dtype>*>::iterator it;
	for (uint32_t i = 0; i < dataVector.size(); i++) {
		it = dataMap.find(dataVector[i]->_name);
		assert(it != dataMap.end());

		Data<Dtype>::compareData(dataVector[i], it->second, error);
	}
}


template <typename Dtype>
void compare_grad(
		std::vector<Data<Dtype>*>& dataVector,
		std::map<std::string, Data<Dtype>*>& dataMap,
		const Dtype error = Dtype(0.001)) {

	typename std::map<std::string, Data<Dtype>*>::iterator it;
	for (uint32_t i = 0; i < dataVector.size(); i++) {
		it = dataMap.find(dataVector[i]->_name);
		assert(it != dataMap.end());
		Data<Dtype>::compareGrad(dataVector[i], it->second, error);
	}
}



template <typename Dtype>
void print_datamap(
		std::map<std::string, Data<Dtype>*>& dataMap,
		const uint32_t opt=0) {

	assert(opt >= 0 && opt < 3);

	std::cout << "print_datamap-------------------------" << std::endl;
	typename std::map<std::string, Data<Dtype>*>::iterator it;
	for (it = dataMap.begin(); it != dataMap.end(); it++) {
		Data<Dtype>* data = it->second;
		Data<Dtype>::printConfig = true;

		switch(opt) {
		case 0:
			data->print_data({}, false);
			data->print_grad({}, false);
			break;
		case 1:
			data->print_data({}, false);
			break;
		case 2:
			data->print_grad({}, false);
			break;
		}

		Data<Dtype>::printConfig = false;
	}
	std::cout << "--------------------------------------" << std::endl;
}


template <typename Dtype>
void print_data(
		std::vector<Data<Dtype>*>& data) {

	std::cout << "print_data----------------------------" << std::endl;
	for (uint32_t i = 0; i < data.size(); i++) {
		Data<Dtype>::printConfig = true;
		data[i]->print_data({}, false);
		data[i]->print_grad({}, false);
		Data<Dtype>::printConfig = false;
	}
	std::cout << "--------------------------------------" << std::endl;
}


#endif /* TEST_COMMON_H_ */












































