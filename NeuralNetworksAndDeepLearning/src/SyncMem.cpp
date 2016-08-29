/*
 * SyncMem.cpp
 *
 *  Created on: 2016. 8. 24.
 *      Author: jhkim
 */

#include "SyncMem.h"
#include "cuda/Cuda.h"
#include <cstring>
#include <limits>


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
void SyncMem<float>::add_device_mem(const float* mem) {
	float* _mem = mutable_device_mem();
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(_size), &Cuda::alpha, mem, 1, _mem, 1));
}

template <typename Dtype>
void SyncMem<Dtype>::scale_host_mem(const float scale) {
	Dtype* _mem = mutable_host_mem();
	for(uint32_t i = 0; i < _size; i++) _mem[i] *= scale;
}

template <>
void SyncMem<float>::scale_device_mem(const float scale) {
	float* _mem = mutable_device_mem();
	checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(_size), &scale, _mem, 1));
}


//template <typename Dtype>
//Dtype SyncMem<Dtype>::sumsq_host_mem() {}

template <>
double SyncMem<float>::sumsq_device_mem() {
	float sumsq;
	const float* _mem = device_mem();
	checkCudaErrors(cublasSdot(Cuda::cublasHandle, _size, _mem, 1, _mem, 1, &sumsq));

	// NaN test
	/*
	if(isnan(sumsq)) {
		sumsq = std::numeric_limits<float>::max();
		print("data:");
	}
	*/
	return (double)sumsq;
}


template <typename Dtype>
void SyncMem<Dtype>::checkDeviceMemAndUpdateHostMem(bool reset) {
	checkMemValidity();
	if(_device_mem_updated) {
#ifdef SYNCMEM_LOG
		cout << "device mem is updated, updating host mem ... " << endl;
#endif
		checkCudaErrors(cudaMemcpyAsync(_host_mem, _device_mem, sizeof(Dtype)*_size, cudaMemcpyDeviceToHost));
		if(reset) _device_mem_updated = false;
	}

}

template <typename Dtype>
void SyncMem<Dtype>::checkHostMemAndUpdateDeviceMem(bool reset) {
	checkMemValidity();
	if(_host_mem_updated) {
#ifdef SYNCMEM_LOG
		cout << "host mem is updated, updating device mem ... " << endl;
#endif
		checkCudaErrors(cudaMemcpyAsync(_device_mem, _host_mem, sizeof(Dtype)*_size, cudaMemcpyHostToDevice));
		if(reset) _host_mem_updated = false;
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

	// print()실행시 updated flag를 reset,
	// mutable pointer 조회하여 계속 업데이트할 경우 print() 이후의 update가 반영되지 않음.
	// 강제로 flag를 reset하지 않도록 수정
	checkDeviceMemAndUpdateHostMem(false);
	const Dtype* data = _host_mem;
	const uint32_t printSize = std::min(10, (int)_size);
	for(uint32_t i = 0; i < printSize; i++) {
		cout << data[i] << ", ";
	}
	cout << endl << "-------------------------------------" << endl;
}

template <typename Dtype>
void SyncMem<Dtype>::print(const string& head, const std::vector<uint32_t>& shape) {
	if(shape.size() != 4) {
		cout << "shape size should be 4 ... " << endl;
		exit(1);
	}
	checkDeviceMemAndUpdateHostMem(false);
	const Dtype* data = _host_mem;



	UINT i,j,k,l;

	const uint32_t rows = shape[2];
	const uint32_t cols = shape[3];
	const uint32_t channels = shape[1];
	const uint32_t batches = shape[0];

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



template class SyncMem<float>;
//template class SyncMem<double>;
template class SyncMem<uint32_t>;













