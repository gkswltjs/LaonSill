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


template <typename Dtype>
SyncMem<Dtype>::SyncMem() {
	_size = 0;

	_host_mem = NULL;
	_device_mem = NULL;

	_host_mem_updated = false;
	_device_mem_updated = false;
}

template <typename Dtype>
SyncMem<Dtype>::~SyncMem() {
	if(_host_mem) delete [] _host_mem;
	if(_device_mem) checkCudaErrors(cudaFree(_device_mem));
}

template <typename Dtype>
void SyncMem<Dtype>::reshape(size_t size) {
	// reshape가 현 상태의 할당된 메모리보다 더 큰 메모리를 요구하는 경우에만 재할당한다.
	if(size > _size) {
		if(_host_mem) delete [] _host_mem;
		if(_device_mem) checkCudaErrors(cudaFree(_device_mem));

		_host_mem = new Dtype[size];
		std::memset(_host_mem, 0, sizeof(Dtype)*size);

		checkCudaErrors(Util::ucudaMalloc(&_device_mem, sizeof(Dtype)*size));
		checkCudaErrors(cudaMemset(_device_mem, 0, sizeof(Dtype)*size));
	}
	_size = size;
}


template <typename Dtype>
const Dtype* SyncMem<Dtype>::host_mem() {
	checkDeviceMemAndUpdateHostMem();
	return (const Dtype*)_host_mem;
}

template <typename Dtype>
const Dtype* SyncMem<Dtype>::device_mem() {
	checkHostMemAndUpdateDeviceMem();
	return (const Dtype*)_device_mem;
}

template <typename Dtype>
Dtype* SyncMem<Dtype>::mutable_host_mem() {
	checkDeviceMemAndUpdateHostMem();
	_host_mem_updated = true;
	return _host_mem;
}

template <typename Dtype>
Dtype* SyncMem<Dtype>::mutable_device_mem() {
	checkHostMemAndUpdateDeviceMem();
	_device_mem_updated = true;
	return _device_mem;
}

template <typename Dtype>
void SyncMem<Dtype>::set_mem(const Dtype* mem, SyncMemCopyType copyType) {
	checkMemValidity();

	switch(copyType) {
	case SyncMemCopyType::HostToHost:
		std::memcpy(_host_mem, mem, sizeof(Dtype)*_size);
		_host_mem_updated = true;
		break;
	case SyncMemCopyType::HostToDevice:
		checkCudaErrors(cudaMemcpyAsync(_device_mem, mem, sizeof(Dtype)*_size, cudaMemcpyHostToDevice));
		_device_mem_updated = true;
		break;
	case SyncMemCopyType::DeviceToHost:
		checkCudaErrors(cudaMemcpyAsync(_host_mem, mem, sizeof(Dtype)*_size, cudaMemcpyDeviceToHost));
		_host_mem_updated = true;
		break;
	case SyncMemCopyType::DeviceToDevice:
		checkCudaErrors(cudaMemcpyAsync(_device_mem, mem, sizeof(Dtype)*_size, cudaMemcpyDeviceToDevice));
		_device_mem_updated = true;
		break;
	}
}

template <typename Dtype>
void SyncMem<Dtype>::reset_host_mem() {
	// reset할 것이므로 device update 여부를 확인, sync과정이 필요없음.
	checkMemValidity();
	std::memset(_host_mem, 0, sizeof(Dtype)*_size);
	_host_mem_updated = true;
}

template <typename Dtype>
void SyncMem<Dtype>::reset_device_mem() {
	checkMemValidity();
	checkCudaErrors(cudaMemset(_device_mem, 0, sizeof(Dtype)*_size));
	_device_mem_updated = true;
}

template <typename Dtype>
void SyncMem<Dtype>::add_host_mem(const Dtype* mem) {
	Dtype* _mem = mutable_host_mem();
	for(uint32_t i = 0; i < _size; i++) _mem[i] += mem[i];
}

template <>
void SyncMem<DATATYPE>::add_device_mem(const DATATYPE* mem) {
	DATATYPE* _mem = mutable_device_mem();
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(_size), &Cuda::alpha, mem, 1, _mem, 1));
}

template <typename Dtype>
void SyncMem<Dtype>::scale_host_mem(const float scale) {
	Dtype* _mem = mutable_host_mem();
	for(uint32_t i = 0; i < _size; i++) _mem[i] *= scale;
}

template <>
void SyncMem<DATATYPE>::scale_device_mem(const float scale) {
	DATATYPE* _mem = mutable_device_mem();
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(_size), &scale, _mem, 1));
}


//template <typename Dtype>
//Dtype SyncMem<Dtype>::sumsq_host_mem() {}

template <>
float SyncMem<DATATYPE>::sumsq_device_mem() {
	float sumsq;
	const DATATYPE* _mem = device_mem();
	checkCudaErrors(cublasSdot(Cuda::cublasHandle, _size, _mem, 1, _mem, 1, &sumsq));
	return sumsq;
}


template <typename Dtype>
void SyncMem<Dtype>::checkDeviceMemAndUpdateHostMem() {
	checkMemValidity();
	if(_device_mem_updated) {
#ifdef SYNCMEM_LOG
		cout << "device mem is updated, updating host mem ... " << endl;
#endif
		checkCudaErrors(cudaMemcpyAsync(_host_mem, _device_mem, sizeof(Dtype)*_size, cudaMemcpyDeviceToHost));
		_device_mem_updated = false;
	}

}

template <typename Dtype>
void SyncMem<Dtype>::checkHostMemAndUpdateDeviceMem() {
	checkMemValidity();
	if(_host_mem_updated) {
#ifdef SYNCMEM_LOG
		cout << "host mem is updated, updating device mem ... " << endl;
#endif
		checkCudaErrors(cudaMemcpyAsync(_device_mem, _host_mem, sizeof(Dtype)*_size, cudaMemcpyHostToDevice));
		_host_mem_updated = false;
	}
}



template <typename Dtype>
void SyncMem<Dtype>::checkMemValidity() {
	if(_size == 0 ||
			_host_mem == NULL ||
			_device_mem == NULL) {

		cout << "assign mem before using ... " << endl;
		exit(1);
		//assert();
	}
}



template <typename Dtype>
void SyncMem<Dtype>::print(const string& head) {
	cout << "-------------------------------------" << endl;
	cout << "name: " << head << " of size: " << _size << endl;

	const Dtype* data = host_mem();
	const uint32_t printSize = std::min(10, (int)_size);
	for(uint32_t i = 0; i < printSize; i++) {
		cout << data[i] << ", ";
	}
	cout << endl << "-------------------------------------" << endl;
}



template class SyncMem<DATATYPE>;
template class SyncMem<uint32_t>;













