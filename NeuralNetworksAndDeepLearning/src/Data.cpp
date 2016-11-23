/*
 * Data.cpp
 *
 *  Created on: 2016. 8. 19.
 *      Author: jhkim
 */

#include <string.h>
#include <cstdlib>
#include <string>

#include "Cuda.h"
#include "Data.h"

//#define DATA_LOG
using namespace std;


template <typename Dtype>
uint32_t Data<Dtype>::printConfig = 0;


template <typename Dtype>
Data<Dtype>::Data(const std::string& name) {
	this->_name = name;
	this->_shape.resize(SHAPE_SIZE);
	this->_count = 0;

	shared_ptr<SyncMem<Dtype>> temp_data(new SyncMem<Dtype>());
	shared_ptr<SyncMem<Dtype>> temp_grad(new SyncMem<Dtype>());

	this->_data = temp_data;
	this->_grad = temp_grad;
}

template <typename Dtype>
Data<Dtype>::Data(const std::string& name, Data<Dtype>* data, uint32_t type) {
	this->_name = name;
	this->_shape.resize(SHAPE_SIZE);
	this->_count = 0;

	if (type == 0) {
		this->_data = data->_data;
		shared_ptr<SyncMem<Dtype>> temp_grad(new SyncMem<Dtype>());
		this->_grad = temp_grad;
	} else if(type == 1) {
		shared_ptr<SyncMem<Dtype>> temp_data(new SyncMem<Dtype>());
		this->_data = temp_data;
		this->_grad = data->_grad;
	}
}

template <typename Dtype>
Data<Dtype>::Data(const std::string& name, const vector<uint32_t>& _shape) : Data(name) {
	shape(_shape);
}


template <typename Dtype>
Data<Dtype>::~Data() {}


bool isValidShapeSize(const vector<uint32_t>& shape, const uint32_t shapeSize) {
	return shape.size() == shapeSize;
}

bool isValidShapeValue(const vector<uint32_t>& shape) {
	for (uint32_t i = 0; i < shape.size(); i++) {
		if(shape[i] <= 0) {
			return false;
		}
	}
	return true;
}



template <typename Dtype>
void Data<Dtype>::shape(const vector<uint32_t>& shape) {
	if (!isValidShapeSize(shape, SHAPE_SIZE)) {
		cout << "invalid data shape ... " << endl;
		exit(1);
	}
	if (!isValidShapeValue(shape)) {
		cout << "invalid data shape ... " << endl;
		exit(1);
	}
	_shape = shape;

	_count = 1;
	for(uint32_t i = 0; i < _shape.size(); i++) _count *= _shape[i];

	_data->shape(_count);
	_grad->shape(_count);
}

template <typename Dtype>
void Data<Dtype>::reshape(const vector<uint32_t>& shape) {
	if (!isValidShapeSize(shape, SHAPE_SIZE)) {
		cout << "invalid data shape ... " << endl;
		exit(1);
	}
	if (!isValidShapeValue(shape)) {
		cout << "invalid data shape ... " << endl;
		exit(1);
	}
	_shape = shape;

	_count = 1;
	for(uint32_t i = 0; i < _shape.size(); i++) _count *= _shape[i];

	_data->reshape(_count);
	_grad->reshape(_count);
}


template <typename Dtype>
const Dtype* Data<Dtype>::host_data() {
	return _data->host_mem();
}

template <typename Dtype>
const Dtype* Data<Dtype>::device_data() {
	return _data->device_mem();
}

template <typename Dtype>
const Dtype* Data<Dtype>::host_grad() {
	return _grad->host_mem();
}

template <typename Dtype>
const Dtype* Data<Dtype>::device_grad() {
	return _grad->device_mem();
}

template <typename Dtype>
Dtype* Data<Dtype>::mutable_host_data() {
	return _data->mutable_host_mem();
}

template <typename Dtype>
Dtype* Data<Dtype>::mutable_device_data() {
	return _data->mutable_device_mem();
}

template <typename Dtype>
Dtype* Data<Dtype>::mutable_host_grad() {
	return _grad->mutable_host_mem();
}

template <typename Dtype>
Dtype* Data<Dtype>::mutable_device_grad() {
	return _grad->mutable_device_mem();
}




template <typename Dtype>
void Data<Dtype>::reset_host_data() {
	_data->reset_host_mem();
}

template <typename Dtype>
void Data<Dtype>::reset_device_data() {
	_data->reset_device_mem();
}

template <typename Dtype>
void Data<Dtype>::reset_host_grad() {
	_grad->reset_host_mem();
}

template <typename Dtype>
void Data<Dtype>::reset_device_grad() {
	_grad->reset_device_mem();
}










template <typename Dtype>
void Data<Dtype>::set_host_data(const Dtype* data) {
	_data->set_mem(data, SyncMemCopyType::HostToHost);
}

template <typename Dtype>
void Data<Dtype>::set_host_with_device_data(const Dtype* data) {
	_data->set_mem(data, SyncMemCopyType::DeviceToHost);
}

template <typename Dtype>
void Data<Dtype>::set_device_with_host_data(const Dtype* data, const size_t offset, const size_t size) {
	_data->set_mem(data, SyncMemCopyType::HostToDevice, offset, size);
}

template <typename Dtype>
void Data<Dtype>::set_device_data(const Dtype* data) {
	_data->set_mem(data, SyncMemCopyType::DeviceToDevice);
}


template <typename Dtype>
void Data<Dtype>::set_host_grad(const Dtype* grad) {
	_grad->set_mem(grad, SyncMemCopyType::HostToHost);
}

template <typename Dtype>
void Data<Dtype>::set_device_grad(const Dtype* grad) {
	_grad->set_mem(grad, SyncMemCopyType::DeviceToDevice);
}





template <typename Dtype>
void Data<Dtype>::add_host_data(const Dtype* data) {
	_data->add_host_mem(data);
}

template <typename Dtype>
void Data<Dtype>::add_device_data(const Dtype* data) {
	_data->add_device_mem(data);
}

template <typename Dtype>
void Data<Dtype>::add_host_grad(const Dtype* grad) {
	_grad->add_host_mem(grad);
}

template <typename Dtype>
void Data<Dtype>::add_device_grad(const Dtype* grad) {
	_grad->add_device_mem(grad);
}



template <typename Dtype>
void Data<Dtype>::scale_host_data(const float scale) {
	_data->scale_host_mem(scale);
}

template <typename Dtype>
void Data<Dtype>::scale_device_data(const float scale) {
	_data->scale_device_mem(scale);
}

template <typename Dtype>
void Data<Dtype>::scale_host_grad(const float scale) {
	_grad->scale_host_mem(scale);
}

template <typename Dtype>
void Data<Dtype>::scale_device_grad(const float scale) {
	_grad->scale_device_mem(scale);
}



template <typename Dtype>
double Data<Dtype>::sumsq_device_data() {
	return _data->sumsq_device_mem();
}

template <typename Dtype>
double Data<Dtype>::sumsq_device_grad() {
	return _grad->sumsq_device_mem();
}


template <typename Dtype>
double Data<Dtype>::asum_device_data() {
	return _data->asum_device_mem();
}

template <typename Dtype>
double Data<Dtype>::asum_device_grad() {
	return _grad->asum_device_mem();
}


template <typename Dtype>
void Data<Dtype>::save(ofstream& ofs) {
	// _shape
	for(uint32_t i = 0; i < SHAPE_SIZE; i++) {
		ofs.write((char*)&_shape[i], sizeof(uint32_t));
	}
	_data->save(ofs);
	_grad->save(ofs);
}

template <typename Dtype>
void Data<Dtype>::load(ifstream& ifs) {
	// _shape
	for(uint32_t i = 0; i < SHAPE_SIZE; i++) {
		ifs.read((char*)&_shape[i], sizeof(uint32_t));
	}
	_data->load(ifs);
	_grad->load(ifs);
}












template <typename Dtype>
void Data<Dtype>::print_data(const string& head, const bool cmo) {
	if(printConfig) {
		_data->print(head, _shape, cmo);
	}
}

template <typename Dtype>
void Data<Dtype>::print_data() {
	print_data(_name+"-data");
}

template <typename Dtype>
void Data<Dtype>::print_data_flatten() {
	if (printConfig) {
		_data->print(_name+"-data");
	}
}


template <typename Dtype>
void Data<Dtype>::print_grad(const string& head) {
	if(printConfig) {
		_grad->print(head, _shape);
	}
}

template <typename Dtype>
void Data<Dtype>::print_grad() {
	print_grad(_name+"-grad");
}



template <typename Dtype>
void Data<Dtype>::fill_host_with_1d_vec(const vector<int>& array,
			const vector<uint32_t>& transpose) {
	assert(array.size() > 0);
	const uint32_t dim1 = array.size();

	Dtype* dataPtr = mutable_host_data();
	const uint32_t batchSize = _shape[1]*_shape[2]*_shape[3];
	const uint32_t heightSize = _shape[2]*_shape[3];
	const uint32_t widthSize = _shape[3];

	const uint32_t tBatchSize = _shape[transpose[1]]*_shape[transpose[2]]*_shape[transpose[3]];
	const uint32_t tHeightSize = _shape[transpose[2]]*_shape[transpose[3]];
	const uint32_t tWidthSize = _shape[transpose[3]];

	uint32_t s[4];
	uint32_t& ts0 = s[transpose[0]];
	uint32_t& ts1 = s[transpose[1]];
	uint32_t& ts2 = s[transpose[2]];
	uint32_t& ts3 = s[transpose[3]];

	// batch
	for (s[0] = 0; s[0] < _shape[0]; s[0]++) {
		// height
		for (s[1] = 0; s[1] < _shape[1]; s[1]++) {
			// width
			for (s[2] = 0; s[2] < _shape[2]; s[2]++) {
				// Anchors
				for (s[3] = 0; s[3] < _shape[3]; s[3]++) {
					dataPtr[ts0*tBatchSize+ts1*tHeightSize+ts2*tWidthSize+ts3]
							= Dtype(array[s[0]*batchSize+s[1]*heightSize+s[2]*widthSize+s[3]]);
				}
			}
		}
	}
}

template <typename Dtype>
void Data<Dtype>::fill_host_with_2d_vec(const vector<vector<float>>& array,
		const vector<uint32_t>& transpose) {
	assert(array.size() > 0);
	const uint32_t dim1 = array.size();
	const uint32_t dim2 = array[0].size();

	assert(_shape[3]%dim2 == 0);

	const uint32_t tBatchSize = _shape[transpose[1]]*_shape[transpose[2]]*_shape[transpose[3]];
	const uint32_t tHeightSize = _shape[transpose[2]]*_shape[transpose[3]];
	const uint32_t tWidthSize = _shape[transpose[3]];

	Dtype* dataPtr = mutable_host_data();
	const uint32_t shape3 = _shape[3] / dim2;
	const uint32_t batchSize = _shape[1]*_shape[2]*shape3;
	const uint32_t heightSize = _shape[2]*shape3;
	const uint32_t widthSize = shape3;

	uint32_t s[4];
	uint32_t& ts0 = s[transpose[0]];
	uint32_t& ts1 = s[transpose[1]];
	uint32_t& ts2 = s[transpose[2]];
	uint32_t& ts3 = s[transpose[3]];

	uint32_t q, r;
	// batch
	for (s[0] = 0; s[0] < _shape[0]; s[0]++) {
		// height
		for (s[1] = 0; s[1] < _shape[1]; s[1]++) {
			// width
			for (s[2] = 0; s[2] < _shape[2]; s[2]++) {
				// Anchors
				for (s[3] = 0; s[3] < _shape[3]; s[3]++) {
					q = s[3] / dim2;
					r = s[3] % dim2;
					dataPtr[ts0*tBatchSize+ts1*tHeightSize+ts2*tWidthSize+ts3] =
							array[s[0]*batchSize+s[1]*heightSize+s[2]*widthSize+q][r];
				}
			}
		}
	}
}







template class Data<float>;
//template class Data<double>;



