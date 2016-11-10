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
Data<Dtype>::Data() {
	this->_shape.resize(SHAPE_SIZE);
}

template <typename Dtype>
Data<Dtype>::Data(const vector<uint32_t>& _shape) : Data() {
	shape(_shape);
}


template <typename Dtype>
Data<Dtype>::~Data() {}


template <typename Dtype>
void Data<Dtype>::shape(const vector<uint32_t>& shape) {
	if(shape.size() != SHAPE_SIZE) {
		cout << "invalid data shape ... " << endl;
		exit(1);
	}
	_shape = shape;

	_count = 1;
	for(uint32_t i = 0; i < _shape.size(); i++) _count *= _shape[i];

	_data.shape(_count);
	_grad.shape(_count);
}

/*
template <typename Dtype>
void Data<Dtype>::reshape(const vector<uint32_t>& shape) {
	if(shape.size() != SHAPE_SIZE) {
		cout << "invalid data shape ... " << endl;
		exit(1);
	}
	_shape = shape;

	_count = 1;
	for(uint32_t i = 0; i < _shape.size(); i++) _count *= _shape[i];

	_data.reshape(_count);
	_grad.reshape(_count);
}
*/

template <typename Dtype>
const Dtype* Data<Dtype>::host_data() {
	return _data.host_mem();
}

template <typename Dtype>
const Dtype* Data<Dtype>::device_data() {
	return _data.device_mem();
}

template <typename Dtype>
const Dtype* Data<Dtype>::host_grad() {
	return _grad.host_mem();
}

template <typename Dtype>
const Dtype* Data<Dtype>::device_grad() {
	return _grad.device_mem();
}

template <typename Dtype>
Dtype* Data<Dtype>::mutable_host_data() {
	return _data.mutable_host_mem();
}

template <typename Dtype>
Dtype* Data<Dtype>::mutable_device_data() {
	return _data.mutable_device_mem();
}

template <typename Dtype>
Dtype* Data<Dtype>::mutable_host_grad() {
	return _grad.mutable_host_mem();
}

template <typename Dtype>
Dtype* Data<Dtype>::mutable_device_grad() {
	return _grad.mutable_device_mem();
}




template <typename Dtype>
void Data<Dtype>::reset_host_data() {
	_data.reset_host_mem();
}

template <typename Dtype>
void Data<Dtype>::reset_device_data() {
	_data.reset_device_mem();
}

template <typename Dtype>
void Data<Dtype>::reset_host_grad() {
	_grad.reset_host_mem();
}

template <typename Dtype>
void Data<Dtype>::reset_device_grad() {
	_grad.reset_device_mem();
}










template <typename Dtype>
void Data<Dtype>::set_host_data(const Dtype* data) {
	_data.set_mem(data, SyncMemCopyType::HostToHost);
}

template <typename Dtype>
void Data<Dtype>::set_host_with_device_data(const Dtype* data) {
	_data.set_mem(data, SyncMemCopyType::DeviceToHost);
}

template <typename Dtype>
void Data<Dtype>::set_device_with_host_data(const Dtype* data, const size_t offset, const size_t size) {
	_data.set_mem(data, SyncMemCopyType::HostToDevice, offset, size);
}

template <typename Dtype>
void Data<Dtype>::set_device_data(const Dtype* data) {
	_data.set_mem(data, SyncMemCopyType::DeviceToDevice);
}


template <typename Dtype>
void Data<Dtype>::set_host_grad(const Dtype* grad) {
	_grad.set_mem(grad, SyncMemCopyType::HostToHost);
}

template <typename Dtype>
void Data<Dtype>::set_device_grad(const Dtype* grad) {
	_grad.set_mem(grad, SyncMemCopyType::DeviceToDevice);
}





template <typename Dtype>
void Data<Dtype>::add_host_data(const Dtype* data) {
	_data.add_host_mem(data);
}

template <typename Dtype>
void Data<Dtype>::add_device_data(const Dtype* data) {
	_data.add_device_mem(data);
}

template <typename Dtype>
void Data<Dtype>::add_host_grad(const Dtype* grad) {
	_grad.add_host_mem(grad);
}

template <typename Dtype>
void Data<Dtype>::add_device_grad(const Dtype* grad) {
	_grad.add_device_mem(grad);
}



template <typename Dtype>
void Data<Dtype>::scale_host_data(const float scale) {
	_data.scale_host_mem(scale);
}

template <typename Dtype>
void Data<Dtype>::scale_device_data(const float scale) {
	_data.scale_device_mem(scale);
}

template <typename Dtype>
void Data<Dtype>::scale_host_grad(const float scale) {
	_grad.scale_host_mem(scale);
}

template <typename Dtype>
void Data<Dtype>::scale_device_grad(const float scale) {
	_grad.scale_device_mem(scale);
}



template <typename Dtype>
double Data<Dtype>::sumsq_device_data() {
	return _data.sumsq_device_mem();
}

template <typename Dtype>
double Data<Dtype>::sumsq_device_grad() {
	return _grad.sumsq_device_mem();
}


template <typename Dtype>
double Data<Dtype>::asum_device_data() {
	return _data.asum_device_mem();
}

template <typename Dtype>
double Data<Dtype>::asum_device_grad() {
	return _grad.asum_device_mem();
}


template <typename Dtype>
void Data<Dtype>::save(ofstream& ofs) {
	// _shape
	for(uint32_t i = 0; i < SHAPE_SIZE; i++) {
		ofs.write((char*)&_shape[i], sizeof(uint32_t));
	}
	_data.save(ofs);
	_grad.save(ofs);
}

template <typename Dtype>
void Data<Dtype>::load(ifstream& ifs) {
	// _shape
	for(uint32_t i = 0; i < SHAPE_SIZE; i++) {
		ifs.read((char*)&_shape[i], sizeof(uint32_t));
	}
	_data.load(ifs);
	_grad.load(ifs);
}












template <typename Dtype>
void Data<Dtype>::print_data(const string& head) {
	if(printConfig) {
		_data.print(head, _shape);
	}
}

template <typename Dtype>
void Data<Dtype>::print_grad(const string& head) {
	if(printConfig) {
		_grad.print(head, _shape);
	}
}





template class Data<float>;
//template class Data<double>;



