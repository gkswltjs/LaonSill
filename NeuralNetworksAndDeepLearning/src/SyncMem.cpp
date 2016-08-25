/*
 * SyncMem.cpp
 *
 *  Created on: 2016. 8. 24.
 *      Author: jhkim
 */

#include "SyncMem.h"
#include "cuda/Cuda.h"
#include <cstring>


//#define SYNCMEM_LOG



SyncMem::SyncMem() {
	_size = 0;

	_host_mem = NULL;
	_device_mem = NULL;

	_host_mem_updated = false;
	_device_mem_updated = false;
}

SyncMem::~SyncMem() {
	if(_host_mem) delete [] _host_mem;
	if(_device_mem) checkCudaErrors(cudaFree(_device_mem));
}


void SyncMem::reshape(size_t size) {
	// reshape가 현 상태의 할당된 메모리보다 더 큰 메모리를 요구하는 경우에만 재할당한다.
	if(size > _size) {
		if(_host_mem) delete [] _host_mem;
		if(_device_mem) checkCudaErrors(cudaFree(_device_mem));

		_host_mem = new DATATYPE[size];
		std::memset(_host_mem, 0, sizeof(DATATYPE)*size);

		checkCudaErrors(Util::ucudaMalloc(&_device_mem, sizeof(DATATYPE)*size));
		checkCudaErrors(cudaMemset(_device_mem, 0, sizeof(DATATYPE)*size));
	}
	_size = size;
}



const DATATYPE* SyncMem::host_mem() {
	checkDeviceMemAndUpdateHostMem();
	return (const DATATYPE*)_host_mem;
}

const DATATYPE* SyncMem::device_mem() {
	checkHostMemAndUpdateDeviceMem();
	return (const DATATYPE*)_device_mem;
}

DATATYPE* SyncMem::mutable_host_mem() {
	checkDeviceMemAndUpdateHostMem();
	_host_mem_updated = true;
	return _host_mem;
}

DATATYPE* SyncMem::mutable_device_mem() {
	checkHostMemAndUpdateDeviceMem();
	_device_mem_updated = true;
	return _device_mem;
}


void SyncMem::set_mem(const DATATYPE* mem, CopyType copyType) {
	checkMemValidity();

	switch(copyType) {
	case CopyType::HostToHost:
		std::memcpy(_host_mem, mem, sizeof(DATATYPE)*_size);
		_host_mem_updated = true;
		break;
	case CopyType::HostToDevice:
		checkCudaErrors(cudaMemcpyAsync(_device_mem, mem, sizeof(DATATYPE)*_size, cudaMemcpyHostToDevice));
		_device_mem_updated = true;
		break;
	case CopyType::DeviceToHost:
		checkCudaErrors(cudaMemcpyAsync(_host_mem, mem, sizeof(DATATYPE)*_size, cudaMemcpyDeviceToHost));
		_host_mem_updated = true;
		break;
	case CopyType::DeviceToDevice:
		checkCudaErrors(cudaMemcpyAsync(_device_mem, mem, sizeof(DATATYPE)*_size, cudaMemcpyDeviceToDevice));
		_device_mem_updated = true;
		break;
	}
}


void SyncMem::reset_host_mem() {
	// reset할 것이므로 device update 여부를 확인, sync과정이 필요없음.
	checkMemValidity();
	std::memset(_host_mem, 0, sizeof(DATATYPE)*_size);
	_host_mem_updated = true;
}

void SyncMem::reset_device_mem() {
	checkMemValidity();
	checkCudaErrors(cudaMemset(_device_mem, 0, sizeof(DATATYPE)*_size));
	_device_mem_updated = true;
}


void SyncMem::add_host_mem(const DATATYPE* mem) {
	DATATYPE* _mem = mutable_host_mem();
	for(uint32_t i = 0; i < _size; i++) _mem[i] += mem[i];
}

void SyncMem::add_device_mem(const DATATYPE* mem) {
	DATATYPE* _mem = mutable_device_mem();
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(_size), &Cuda::alpha, mem, 1, _mem, 1));
}


void SyncMem::scale_host_mem(const float scale) {
	DATATYPE* _mem = mutable_host_mem();
	for(uint32_t i = 0; i < _size; i++) _mem[i] *= scale;
}

void SyncMem::scale_device_mem(const float scale) {
	DATATYPE* _mem = mutable_device_mem();
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(_size), &scale, _mem, 1));
}


//DATATYPE SyncMem::sumsq_host_mem() {}

DATATYPE SyncMem::sumsq_device_mem() {
	DATATYPE sumsq;
	const DATATYPE* _mem = device_mem();
	checkCudaErrors(cublasSdot(Cuda::cublasHandle, _size, _mem, 1, _mem, 1, &sumsq));
	return sumsq;
}



void SyncMem::checkDeviceMemAndUpdateHostMem() {
	checkMemValidity();
	if(_device_mem_updated) {
#ifdef SYNCMEM_LOG
		cout << "device mem is updated, updating host mem ... " << endl;
#endif
		checkCudaErrors(cudaMemcpyAsync(_host_mem, _device_mem, sizeof(DATATYPE)*_size, cudaMemcpyDeviceToHost));
		_device_mem_updated = false;
	}

}

void SyncMem::checkHostMemAndUpdateDeviceMem() {
	checkMemValidity();
	if(_host_mem_updated) {
#ifdef SYNCMEM_LOG
		cout << "host mem is updated, updating device mem ... " << endl;
#endif
		checkCudaErrors(cudaMemcpyAsync(_device_mem, _host_mem, sizeof(DATATYPE)*_size, cudaMemcpyHostToDevice));
		_host_mem_updated = false;
	}
}




void SyncMem::checkMemValidity() {
	if(_size == 0 ||
			_host_mem == NULL ||
			_device_mem == NULL) {

		cout << "assign mem before using ... " << endl;
		exit(1);
		//assert();
	}
}


















