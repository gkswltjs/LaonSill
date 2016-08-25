/*
 * Data.cpp
 *
 *  Created on: 2016. 8. 19.
 *      Author: jhkim
 */

#include "Data.h"

#include <string.h>
#include <cstdlib>
#include <iostream>
#include <string>

#include "cuda/Cuda.h"

//#define DATA_LOG



uint32_t Data::printConfig = 0;



Data::Data() {
	this->_shape.resize(SHAPE_SIZE);
}


Data::~Data() {}


void Data::reshape(const vector<uint32_t>& shape) {
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

const DATATYPE* Data::host_data() {
	return _data.host_mem();
}

const DATATYPE* Data::device_data() {
	return _data.device_mem();
}

const DATATYPE* Data::host_grad() {
	return _grad.host_mem();
}

const DATATYPE* Data::device_grad() {
	return _grad.device_mem();
}

DATATYPE* Data::mutable_host_data() {
	return _data.mutable_host_mem();
}

DATATYPE* Data::mutable_device_data() {
	return _data.mutable_device_mem();
}

DATATYPE* Data::mutable_host_grad() {
	return _grad.mutable_host_mem();
}

DATATYPE* Data::mutable_device_grad() {
	return _grad.mutable_device_mem();
}





void Data::reset_host_data() {
	_data.reset_host_mem();
}

void Data::reset_device_data() {
	_data.reset_device_mem();
}

void Data::reset_host_grad() {
	_grad.reset_host_mem();
}

void Data::reset_device_grad() {
	_grad.reset_device_mem();
}











void Data::set_host_data(const DATATYPE* data) {
	_data.set_mem(data, SyncMemCopyType::HostToHost);
}

void Data::set_host_with_device_data(const DATATYPE* data) {
	_data.set_mem(data, SyncMemCopyType::DeviceToHost);
}

void Data::set_device_with_host_data(const DATATYPE* data) {
	_data.set_mem(data, SyncMemCopyType::HostToDevice);
}

void Data::set_device_data(const DATATYPE* data) {
	_data.set_mem(data, SyncMemCopyType::DeviceToDevice);
}


void Data::set_host_grad(const DATATYPE* grad) {
	_grad.set_mem(grad, SyncMemCopyType::HostToHost);
}

void Data::set_device_grad(const DATATYPE* grad) {
	_grad.set_mem(grad, SyncMemCopyType::DeviceToDevice);
}






void Data::add_host_data(const DATATYPE* data) {
	_data.add_host_mem(data);
}

void Data::add_device_data(const DATATYPE* data) {
	_data.add_device_mem(data);
}

void Data::add_host_grad(const DATATYPE* grad) {
	_grad.add_host_mem(grad);
}

void Data::add_device_grad(const DATATYPE* grad) {
	_grad.add_device_mem(grad);
}




void Data::scale_host_data(const float scale) {
	_data.scale_host_mem(scale);
}

void Data::scale_device_data(const float scale) {
	_data.scale_device_mem(scale);
}

void Data::scale_host_grad(const float scale) {
	_grad.scale_host_mem(scale);
}

void Data::scale_device_grad(const float scale) {
	_grad.scale_device_mem(scale);
}




float Data::sumsq_device_data() {
	return _data.sumsq_device_mem();
}

float Data::sumsq_device_grad() {
	return _grad.sumsq_device_mem();
}

















void Data::print_data(const string& head) {
	if(printConfig) {
		print(host_data(), head);
	}
}

void Data::print_grad(const string& head) {
	if(printConfig) {
		print(host_grad(), head);
	}
}

void Data::print(const DATATYPE* data, const string& head) {
	UINT i,j,k,l;

	const uint32_t rows = _shape[2];
	const uint32_t cols = _shape[3];
	const uint32_t channels = _shape[1];
	const uint32_t batches = _shape[0];

	cout << "-------------------------------------" << endl;
	cout << "name: " << head << endl;
	cout << "rows x cols x channels x batches: " << rows << " x " << cols << " x " << channels << " x " << batches << endl;

	UINT batchElem = rows*cols*channels;
	UINT channelElem = rows*cols;
	for(i = 0; i < batches; i++) {
		for(j = 0; j < channels; j++) {
			for(k = 0; k < rows; k++) {
				for(l = 0; l < cols; l++) {
			//for(k = 0; k < std::min(10, (int)rows); k++) {
			//	for(l = 0; l < std::min(10, (int)cols); l++) {
					cout << data[i*batchElem + j*channelElem + l*rows + k] << ", ";
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
	}
	cout << "-------------------------------------" << endl;
}








/*
Data::Data() {
	this->_shape.resize(SHAPE_SIZE);
	_cpu_data = NULL;
	_gpu_data = NULL;
	_cpu_grad = NULL;
	_gpu_grad = NULL;

	_cpu_data_modified = false;
	_gpu_data_modified = false;
	_cpu_grad_modified = false;
	_gpu_grad_modified = false;
}


Data::~Data() {
	if(_cpu_data) delete []_cpu_data;
	if(_gpu_data) checkCudaErrors(cudaFree(_gpu_data));
	if(_cpu_grad) delete [] _cpu_grad;
	if(_gpu_grad) checkCudaErrors(cudaFree(_gpu_grad));
}

void Data::reshape(const vector<uint32_t>& shape) {
	if(shape.size() != SHAPE_SIZE) {
		cout << "invalid data shape ... " << endl;
		exit(1);
	}
	_shape = shape;

	_count = 1;
	for(uint32_t i = 0; i < _shape.size(); i++) _count *= _shape[i];
}

const DATATYPE* Data::cpu_data() {
	checkGpuDataAndUpdateCpuData();
	return (const DATATYPE*)_cpu_data;
}

const DATATYPE* Data::gpu_data() {
	checkCpuDataAndUpdateGpuData();
	return (const DATATYPE*)_gpu_data;
}

const DATATYPE* Data::cpu_grad() {
	checkGpuGradAndUpdateCpuGrad();
	return (const DATATYPE*)_cpu_grad;
}

const DATATYPE* Data::gpu_grad() {
	checkCpuGradAndUpdateGpuGrad();
	return (const DATATYPE*)_gpu_grad;
}

DATATYPE* Data::mutable_cpu_data() {
	checkGpuDataAndUpdateCpuData();
	_cpu_data_modified = true;
	return _cpu_data;
}

DATATYPE* Data::mutable_gpu_data() {
	checkCpuDataAndUpdateGpuData();
	_gpu_data_modified = true;
	return _gpu_data;
}

DATATYPE* Data::mutable_cpu_grad() {
	checkGpuGradAndUpdateCpuGrad();
	_cpu_grad_modified = true;
	return _cpu_grad;
}

DATATYPE* Data::mutable_gpu_grad() {
	checkCpuGradAndUpdateGpuGrad();
	_gpu_grad_modified = true;
	return _gpu_grad;
}

void Data::checkGpuDataAndUpdateCpuData() {
	alloc_cpu_data();

	if(_gpu_data_modified) {
#ifdef DATA_LOG
		cout << "gpu data is modified, updating cpu data ... " << endl;
#endif
		checkCudaErrors(cudaMemcpyAsync(_cpu_data, _gpu_data, sizeof(DATATYPE)*_count, cudaMemcpyDeviceToHost));
		_gpu_data_modified = false;
	}
}

void Data::checkCpuDataAndUpdateGpuData() {
	alloc_gpu_data();

	if(_cpu_data_modified) {
#ifdef DATA_LOG
		cout << "cpu data is modified, updating gpu data ... " << endl;
#endif
		checkCudaErrors(cudaMemcpyAsync(_gpu_data, _cpu_data, sizeof(DATATYPE)*_count, cudaMemcpyHostToDevice));
		_cpu_data_modified = false;
	}
}

void Data::checkGpuGradAndUpdateCpuGrad() {
	alloc_cpu_grad();

	if(_gpu_grad_modified) {
#ifdef DATA_LOG
		cout << "gpu grad is modified, updating cpu grad ... " << endl;
#endif
		checkCudaErrors(cudaMemcpyAsync(_cpu_grad, _gpu_grad, sizeof(DATATYPE)*_count, cudaMemcpyDeviceToHost));
		_gpu_grad_modified = false;
	}
}

void Data::checkCpuGradAndUpdateGpuGrad() {
	alloc_gpu_grad();

	if(_cpu_grad_modified) {
#ifdef DATA_LOG
		cout << "cpu grad is modified, updating gpu grad ... " << endl;
#endif
		checkCudaErrors(cudaMemcpyAsync(_gpu_grad, _cpu_grad, sizeof(DATATYPE)*_count, cudaMemcpyHostToDevice));
		_cpu_grad_modified = false;
	}
}



void Data::reset_cpu_data() {
	if(!alloc_cpu_data()) {
		memset(_cpu_data, 0, sizeof(DATATYPE)*_count);
	}
	_cpu_data_modified = true;
}

void Data::reset_gpu_data() {
	if(!alloc_gpu_data()) {
		checkCudaErrors(cudaMemset(_gpu_data, 0, sizeof(DATATYPE)*_count));
	}
	_gpu_data_modified = true;
}

void Data::reset_cpu_grad() {
	if(!alloc_cpu_grad()) {
		memset(_cpu_grad, 0, sizeof(DATATYPE)*_count);
	}
	_cpu_grad_modified = true;
}

void Data::reset_gpu_grad() {
	if(!alloc_gpu_grad()) {
		checkCudaErrors(cudaMemset(_gpu_grad, 0, sizeof(DATATYPE)*_count));
	}
	_gpu_grad_modified = true;
}











void Data::set_cpu_data(const DATATYPE* data) {
	//DATATYPE* cpu_data = mutable_cpu_data();
	//memcpy(cpu_data, data, sizeof(DATATYPE)*_count);
	set_data(data, HostToHost);
}

void Data::set_gpu_data(const DATATYPE* data) {
	//DATATYPE* gpu_data = mutable_gpu_data();
	//checkCudaErrors(cudaMemcpyAsync(gpu_data, data, sizeof(DATATYPE)*_count, cudaMemcpyDeviceToDevice));
	set_data(data, DeviceToDevice);
}

void Data::set_data(const DATATYPE* data, CopyType copyType) {
	DATATYPE* _data;
	switch(copyType) {
	case CopyType::HostToHost:
		_data = mutable_cpu_data();
		memcpy(_data, data, sizeof(DATATYPE)*_count);
		break;
	case CopyType::HostToDevice:
		_data = mutable_gpu_data();
		checkCudaErrors(cudaMemcpyAsync(_data, data, sizeof(DATATYPE)*_count, cudaMemcpyHostToDevice));
		break;
	case CopyType::DeviceToHost:
		_data = mutable_cpu_data();
		checkCudaErrors(cudaMemcpyAsync(_data, data, sizeof(DATATYPE)*_count, cudaMemcpyDeviceToHost));
		break;
	case CopyType::DeviceToDevice:
		DATATYPE* _data = mutable_gpu_data();
		checkCudaErrors(cudaMemcpyAsync(_data, data, sizeof(DATATYPE)*_count, cudaMemcpyDeviceToDevice));
		break;
	}
}




void Data::set_cpu_grad(const DATATYPE* grad) {
	DATATYPE* cpu_grad = mutable_cpu_grad();
	memcpy(cpu_grad, grad, sizeof(DATATYPE)*_count);
}

void Data::set_gpu_grad(const DATATYPE* grad) {
	DATATYPE* gpu_grad = mutable_gpu_grad();
	checkCudaErrors(cudaMemcpyAsync(gpu_grad, grad, sizeof(DATATYPE)*_count, cudaMemcpyDeviceToDevice));
}






void Data::add_cpu_data(const DATATYPE* data) {
	DATATYPE* cpu_data = mutable_cpu_data();
	for(uint32_t i = 0; i < _count; i++) cpu_data[i] += data[i];
}

void Data::add_gpu_data(const DATATYPE* data) {
	DATATYPE* gpu_data = mutable_gpu_data();
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(_count), &Cuda::alpha, data, 1, gpu_data, 1));
}

void Data::add_cpu_grad(const DATATYPE* grad) {
	DATATYPE* cpu_grad = mutable_cpu_grad();
	for(uint32_t i = 0; i < _count; i++) cpu_grad[i] += grad[i];
}

void Data::add_gpu_grad(const DATATYPE* grad) {
	DATATYPE* gpu_grad = mutable_gpu_grad();
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(_count), &Cuda::alpha, grad, 1, gpu_grad, 1));
}


void Data::scale_cpu_data(const float scale) {
	DATATYPE* cpu_data = mutable_cpu_data();
	for(uint32_t i = 0; i < _count; i++) cpu_data[i] *= scale;
}

void Data::scale_gpu_data(const float scale) {
	DATATYPE* gpu_data = mutable_gpu_data();
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(_count), &scale, gpu_data, 1));
}

void Data::scale_cpu_grad(const float scale) {
	DATATYPE* cpu_grad = mutable_cpu_grad();
	for(uint32_t i = 0; i < _count; i++) cpu_grad[i] *= scale;
}

void Data::scale_gpu_grad(const float scale) {
	DATATYPE* gpu_grad = mutable_gpu_grad();
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(_count), &scale, gpu_grad, 1));
}

DATATYPE Data::sumsq_gpu_data() {
	DATATYPE sumsq;
	const DATATYPE* gpu_data = Data::gpu_data();
	checkCudaErrors(cublasSdot(Cuda::cublasHandle, _count, gpu_data, 1, gpu_data, 1, &sumsq));
	return sumsq;
}

DATATYPE Data::sumsq_gpu_grad() {

	DATATYPE sumsq;
	//alloc_gpu_grad();
	const DATATYPE* gpu_grad = Data::gpu_grad();
	//checkCudaErrors(cublasSdot(Cuda::cublasHandle, _count, gpu_grad, 1, gpu_grad, 1, &sumsq));
	uint32_t status = cublasSdot(Cuda::cublasHandle, _count, gpu_grad, 1, gpu_grad, 1, &sumsq);
	std::stringstream _error;
	if (status != 0) {
		_error << "Cuda failure: " << status;
		FatalError(_error.str());
	}
	return sumsq;
}

















void Data::print_data(const string& head) {
	if(printConfig) {
		print(cpu_data(), head);
	}
}

void Data::print_grad(const string& head) {
	if(printConfig) {
		print(cpu_grad(), head);
	}
}

void Data::print(const DATATYPE* data, const string& head) {
	UINT i,j,k,l;

	const uint32_t rows = _shape[2];
	const uint32_t cols = _shape[3];
	const uint32_t channels = _shape[1];
	const uint32_t batches = _shape[0];

	cout << "-------------------------------------" << endl;
	cout << "name: " << head << endl;
	cout << "rows x cols x channels x batches: " << rows << " x " << cols << " x " << channels << " x " << batches << endl;

	UINT batchElem = rows*cols*channels;
	UINT channelElem = rows*cols;
	for(i = 0; i < batches; i++) {
		for(j = 0; j < channels; j++) {
			for(k = 0; k < rows; k++) {
				for(l = 0; l < cols; l++) {
			//for(k = 0; k < std::min(10, (int)rows); k++) {
			//	for(l = 0; l < std::min(10, (int)cols); l++) {
					cout << data[i*batchElem + j*channelElem + l*rows + k] << ", ";
				}
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
	}
	cout << "-------------------------------------" << endl;
}






bool Data::alloc_cpu_data() {
	if(!_cpu_data) {
		_cpu_data = new DATATYPE[_count];
		memset(_cpu_data, 0, sizeof(DATATYPE)*_count);
		return true;
	}
	return false;
}

bool Data::alloc_gpu_data() {
	if(!_gpu_data) {
		checkCudaErrors(Util::ucudaMalloc(&_gpu_data, sizeof(DATATYPE)*_count));
		checkCudaErrors(cudaMemset(_gpu_data, 0, sizeof(DATATYPE)*_count));
		return true;
	}
	return false;
}

bool Data::alloc_cpu_grad() {
	if(!_cpu_grad) {
		_cpu_grad = new DATATYPE[_count];
		memset(_cpu_grad, 0, sizeof(DATATYPE)*_count);
		return true;
	}
	return false;
}

bool Data::alloc_gpu_grad() {
	if(!_gpu_grad) {
		checkCudaErrors(Util::ucudaMalloc(&_gpu_grad, sizeof(DATATYPE)*_count));
		checkCudaErrors(cudaMemset(_gpu_grad, 0, sizeof(DATATYPE)*_count));
		return true;
	}
	return false;
}
*/





























