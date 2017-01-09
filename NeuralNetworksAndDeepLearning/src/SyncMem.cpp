/*
 * SyncMem.cpp
 *
 *  Created on: 2016. 8. 24.
 *      Author: jhkim
 */

#include <cstring>
#include <limits>
#include <cmath>
#include <cfloat>

#include "SyncMem.h"
#include "Cuda.h"

//#define SYNCMEM_LOG

#define MEM_MAX (FLT_MAX / 10)

using namespace std;


template <typename Dtype>
ostream *SyncMem<Dtype>::outstream = &cout;





template <typename Dtype>
SyncMem<Dtype>::SyncMem() {
	_size = 0;
	_reserved = 0;

	_host_mem = NULL;
	_device_mem = NULL;

	checkCudaErrors(Util::ucudaMalloc(&_d_int, sizeof(uint32_t)));
	checkCudaErrors(Util::ucudaMalloc(&_d_bool, sizeof(bool)));

	//_host_mem_updated = false;
	//_device_mem_updated = false;
	resetMemUpdated();
}

template <typename Dtype>
SyncMem<Dtype>::SyncMem(SyncMem<Dtype>& syncMem) : SyncMem() {
	reshape(syncMem.getSize());
	set_mem(syncMem.device_mem(), DeviceToDevice, 0, syncMem.getSize());
}



template <typename Dtype>
SyncMem<Dtype>::~SyncMem() {
	if(_host_mem) delete [] _host_mem;
	if(_device_mem) checkCudaErrors(cudaFree(_device_mem));

	if(_d_int) checkCudaErrors(cudaFree(_d_int));
	if(_d_bool) checkCudaErrors(cudaFree(_d_bool));
}

/*
template <typename Dtype>
void SyncMem<Dtype>::shape(size_t size) {
	_reserved = (size_t)(size*1.2f);

	_host_mem = new Dtype[_reserved];
	memset(_host_mem, 0, sizeof(Dtype)*_reserved);

	checkCudaErrors(Util::ucudaMalloc(&_device_mem, sizeof(Dtype)*_reserved));
	checkCudaErrors(cudaMemset(_device_mem, 0, sizeof(Dtype)*_reserved));

	_size = size;
}
*/

template <typename Dtype>
void SyncMem<Dtype>::reshape(size_t size) {
	// reshape가 현 상태의 할당된 메모리보다 더 큰 메모리를 요구하는 경우에만 재할당한다.
	if (size > _reserved) {
		if(_host_mem) delete [] _host_mem;
		if(_device_mem) checkCudaErrors(cudaFree(_device_mem));

		_reserved = (size_t)(size*1.2f);
		_host_mem = new Dtype[_reserved];
		memset(_host_mem, 0, sizeof(Dtype)*_reserved);

		checkCudaErrors(Util::ucudaMalloc(&_device_mem, sizeof(Dtype)*_reserved));
		checkCudaErrors(cudaMemset(_device_mem, 0, sizeof(Dtype)*_reserved));
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
	//_host_mem_updated = true;
	setHostMemUpdated();
	return _host_mem;
}

template <typename Dtype>
Dtype* SyncMem<Dtype>::mutable_device_mem() {
	checkHostMemAndUpdateDeviceMem();
	//_device_mem_updated = true;
	setDeviceMemUpdated();
	return _device_mem;
}

template <typename Dtype>
void SyncMem<Dtype>::set_mem(const Dtype* mem, SyncMemCopyType copyType, const size_t offset, const size_t size) {
	checkMemValidity();

	size_t copySize;
	if(size == 0) copySize = _size;
	else copySize = size;

	switch(copyType) {
	case SyncMemCopyType::HostToHost:
		memcpy(_host_mem+offset, mem, sizeof(Dtype)*copySize);
		//_host_mem_updated = true;
		setHostMemUpdated();
		break;
	case SyncMemCopyType::HostToDevice:
		checkCudaErrors(cudaMemcpyAsync(_device_mem+offset, mem, sizeof(Dtype)*copySize, cudaMemcpyHostToDevice));
		/*
		cudaError_t status = cudaMemcpyAsync(_device_mem+offset, mem, sizeof(Dtype)*copySize, cudaMemcpyHostToDevice);
		if (status != 0) {
		  cout << "Cuda failure: " << status;
		  //FatalError(_error.str());
		}
		*/
		//_device_mem_updated = true;
		setDeviceMemUpdated();
		break;
	case SyncMemCopyType::DeviceToHost:
		checkCudaErrors(cudaMemcpyAsync(_host_mem+offset, mem, sizeof(Dtype)*copySize, cudaMemcpyDeviceToHost));
		//_host_mem_updated = true;
		setHostMemUpdated();
		break;
	case SyncMemCopyType::DeviceToDevice:
		checkCudaErrors(cudaMemcpyAsync(_device_mem+offset, mem, sizeof(Dtype)*copySize, cudaMemcpyDeviceToDevice));
		//_device_mem_updated = true;
		setDeviceMemUpdated();
		break;
	}
}

template <typename Dtype>
void SyncMem<Dtype>::reset_host_mem(const bool setZero, const Dtype value) {
	// reset할 것이므로 device update 여부를 확인, sync과정이 필요없음.
	checkMemValidity();

	if (setZero)
		memset(_host_mem, 0, sizeof(Dtype)*_size);
	else {
		for (size_t i = 0; i < _size; i++) {
			_host_mem[i] = value;
		}
	}

	//_host_mem_updated = true;
	setHostMemUpdated();
}

template <typename Dtype>
void SyncMem<Dtype>::reset_device_mem() {
	checkMemValidity();
	checkCudaErrors(cudaMemset(_device_mem, 0, sizeof(Dtype)*_size));
	//_device_mem_updated = true;
	setDeviceMemUpdated();
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

template <>
void SyncMem<int>::add_device_mem(const int* mem) {
	assert(false &&
			"SyncMem<int>::add_device_mem() is not supported ... ");
}


template <typename Dtype>
void SyncMem<Dtype>::sub_host_mem(const Dtype* mem) {
	Dtype* _mem = mutable_host_mem();
	for(uint32_t i = 0; i < _size; i++) _mem[i] -= mem[i];
}


template<>
void SyncMem<float>::sub_device_mem(const float* mem) {
	float* _mem = mutable_device_mem();
	checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(_size), &Cuda::negativeOne, mem, 1, _mem, 1));
}

template<>
void SyncMem<int>::sub_device_mem(const int* mem) {
	assert(false &&
			"SyncMem<int>::sub_device_mem() is not supported ... ");
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

template <>
void SyncMem<int>::scale_device_mem(const float scale) {
	assert(false &&
			"SyncMem<int>::scale_device_mem() is not supported ... ");
}





//template <typename Dtype>
//Dtype SyncMem<Dtype>::sumsq_host_mem() {}

template <>
double SyncMem<float>::sumsq_device_mem() {
	float sumsq;
	const float* _mem = device_mem();
	checkCudaErrors(cublasSdot(Cuda::cublasHandle, _size, _mem, 1, _mem, 1, &sumsq));
	return (double)sumsq;
}

template <>
double SyncMem<int>::sumsq_device_mem() {
	assert(false &&
			"SyncMem<int>::sumsq_device_mem() is not supported ... ");
}


template <>
double SyncMem<float>::asum_device_mem() {
	float asum;
	const float* _mem = device_mem();
	checkCudaErrors(cublasSasum(Cuda::cublasHandle, _size, _mem, 1, &asum));
	return (double)asum;
}

template <>
double SyncMem<int>::asum_device_mem() {
	assert(false &&
			"SyncMem<int>::asum_device_mem() is not supported ... ");
}




template <typename Dtype>
void SyncMem<Dtype>::checkDeviceMemAndUpdateHostMem(bool reset) {
	checkMemValidity();
	if(_device_mem_updated) {
#ifdef SYNCMEM_LOG
		cout << "device mem is updated, updating host mem ... " << endl;
#endif
		checkCudaErrors(cudaMemcpyAsync(_host_mem, _device_mem, sizeof(Dtype)*_size, cudaMemcpyDeviceToHost));
		if(reset) {
			//_device_mem_updated = false;
			resetMemUpdated();
		}
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
		if(reset) {
			//_host_mem_updated = false;
			resetMemUpdated();
		}
	}
}



template <typename Dtype>
void SyncMem<Dtype>::checkMemValidity() {
	/*
	if(_size == 0 ||
			_host_mem == NULL ||
			_device_mem == NULL) {

		cout << "assign mem before using ... " << endl;
		exit(1);
	}
	*/
	assert(_size > 0 &&
			_host_mem != NULL &&
			_device_mem != NULL &&
			"assign mem before using ... ");
}


template <typename Dtype>
void SyncMem<Dtype>::setHostMemUpdated() {
	_device_mem_updated = false;
	_host_mem_updated = true;
}

template <typename Dtype>
void SyncMem<Dtype>::setDeviceMemUpdated() {
	_host_mem_updated = false;
	_device_mem_updated = true;
}

template <typename Dtype>
void SyncMem<Dtype>::resetMemUpdated() {
	_host_mem_updated = false;
	_device_mem_updated = false;
}




template <typename Dtype>
bool SyncMem<Dtype>::is_nan_mem() {
	checkDeviceMemAndUpdateHostMem(false);
	const Dtype* data = _host_mem;

	for(uint32_t i = 0; i < _size; i++) {
		if(isnan(data[i])) {
			return true;
		}
	}
	return false;
}

template <typename Dtype>
bool SyncMem<Dtype>::is_inf_mem() {
	checkDeviceMemAndUpdateHostMem(false);
	const Dtype* data = _host_mem;

	for(uint32_t i = 0; i < _size; i++) {
		if(isinff(data[i])) {
			return true;
		}
	}
	return false;
}

/*
template <>
bool SyncMem<float>::bound_mem() {
	//const float* d_mem = device_mem();
	//float asum = 0;
	//checkCudaErrors(cublasSasum(Cuda::cublasHandle, static_cast<int>(_size), d_mem, 1, &asum));
	//float bound = 1000*(asum / _size);

	float* h_mem = mutable_host_mem();
	//double average = 0.0;
	int updateCount = 0;
	for(size_t i = 0; i < _size; i++) {
		//if(abs(h_mem[i]) > bound) {
			//h_mem[i] = (h_mem[i]>0)?bound:-bound;
		if(h_mem[i] > 10) {
			h_mem[i] = 10;
			updateCount++;
		}
		//}
	}
	if(updateCount > 0) return true;
	return false;
	//cout << "bounded " << updateCount << " elements ... " << endl;
}
*/



template <typename Dtype>
void SyncMem<Dtype>::save(ofstream& ofs) {
	const Dtype* ptr = host_mem();
	//ofs.write((char*)&_size, sizeof(size_t));
	ofs.write((char*)ptr, sizeof(Dtype)*_size);
}


template <typename Dtype>
void SyncMem<Dtype>::load(ifstream& ifs) {

	// size와 reshape 관련은 Data에서 처리하여 생략------------
	//size_t size;
	//ifs.read((char*)&size, sizeof(size_t));
	//reshape(size);
	// --------------------------------------------

	Dtype* ptr = this->mutable_host_mem();
	ifs.read((char*)ptr, sizeof(Dtype)*this->_size);
}



template <typename Dtype>
void SyncMem<Dtype>::print(const string& head) {
	if(true) {
		(*outstream) << "-------------------------------------" << endl;
		(*outstream) << "name: " << head << " of size: " << _size << endl;

		// print()실행시 updated flag를 reset,
		// mutable pointer 조회하여 계속 업데이트할 경우 print() 이후의 update가 반영되지 않음.
		// 강제로 flag를 reset하지 않도록 수정
		checkDeviceMemAndUpdateHostMem(false);
		const Dtype* data = _host_mem;
		const uint32_t printSize = min(64*3, (int)_size);
		//const uint32_t printSize = (uint32_t)_size;
		for(uint32_t i = 0; i < printSize; i++) {
			(*outstream) << data[i] << ", ";
		}
		(*outstream) << endl << "-------------------------------------" << endl;
	}
}


template <typename Dtype>
void SyncMem<Dtype>::print(const string& head, const vector<uint32_t>& shape, const bool cmo) {
	if(true) {
		if(shape.size() != 4) {
			(*outstream) << "shape size should be 4 ... " << endl;
			exit(1);
		}
		checkDeviceMemAndUpdateHostMem(false);
		const Dtype* data = _host_mem;

		uint32_t i,j,k,l;

		const uint32_t rows = shape[2];
		const uint32_t cols = shape[3];
		const uint32_t channels = shape[1];
		const uint32_t batches = shape[0];

		(*outstream) << "-------------------------------------" << endl;
		(*outstream) << "name: " << head << endl;
		(*outstream) << "batches x channels x rows x cols: " << batches << " x " <<
				channels << " x " << rows << " x " << cols << endl;

		const uint32_t batchElem = rows*cols*channels;
		const uint32_t channelElem = rows*cols;

		if (cmo) {
			for(i = 0; i < batches; i++) {
				for(j = 0; j < channels; j++) {
					for(k = 0; k < rows; k++) {
						for(l = 0; l < cols; l++) {
					//for(k = 0; k < min(10, (int)rows); k++) {
					//	for(l = 0; l < min(10, (int)cols); l++) {
							(*outstream) << data[i*batchElem + j*channelElem + l*rows + k] << ", ";
						}
						(*outstream) << endl;
					}
					(*outstream) << endl;
				}
				(*outstream) << endl;
			}
		} else {
			for(i = 0; i < batches; i++) {
				for(j = 0; j < channels; j++) {
					(*outstream) << "[" << i << "x" << j << "]" << endl;

					for(k = 0; k < rows; k++) {
						for(l = 0; l < cols; l++) {
							(*outstream) << data[i*batchElem + j*channelElem + k*cols + l] << ", ";
						}
						(*outstream) << endl;
					}
					(*outstream) << endl;
				}
				(*outstream) << endl;
			}
		}
		(*outstream) << "-------------------------------------" << endl;
	}

}



template class SyncMem<float>;
//template class SyncMem<double>;
template class SyncMem<uint32_t>;
template class SyncMem<int>;













