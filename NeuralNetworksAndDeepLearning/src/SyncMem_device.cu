/*
 * SyncMem.cpp
 *
 *  Created on: 2016. 8. 24.
 *      Author: jhkim
 */

#include <math_functions.hpp>
#include "SyncMem.h"
#include <cfloat>


//#define SYNCMEM_LOG

#define MEM_MAX (FLT_MAX / 10)


/*
__global__ void IsNan(const float* mem, bool* result, const unsigned int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;

	if(isnan(mem[idx])) *result = true;
}


__global__ void IsInf(const float* mem, bool* result, const unsigned int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;

	if(isinf(mem[idx])) *result = true;
}
*/



template <typename Dtype>
__global__ void BoundMem(Dtype* mem, const Dtype bound, uint32_t* updateCount, const unsigned int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	if(mem[idx] > bound) {
		mem[idx] = bound;
		*updateCount++;
	} else if(mem[idx] < -bound) {
		mem[idx] = -bound;
		*updateCount++;
	}
}



/*
template <typename Dtype>
bool SyncMem<Dtype>::is_nan_mem() {

	const Dtype* d_mem = device_mem();
	_h_bool = false;
	checkCudaErrors(cudaMemcpy(_d_bool, &_h_bool, sizeof(bool), cudaMemcpyHostToDevice));
	IsNan<<<RoundUp((unsigned int)_size, BW), BW>>>(d_mem, _d_bool, (unsigned int)_size);
	checkCudaErrors(cudaMemcpyAsync(&_h_bool, _d_bool, sizeof(bool), cudaMemcpyDeviceToHost));

	return _h_bool;
}

template <typename Dtype>
bool SyncMem<Dtype>::is_inf_mem() {
	const Dtype* d_mem = device_mem();
	_h_bool = false;
	checkCudaErrors(cudaMemcpy(_d_bool, &_h_bool, sizeof(bool), cudaMemcpyHostToDevice));
	IsInf<<<RoundUp((unsigned int)_size, BW), BW>>>(d_mem, _d_bool, (unsigned int)_size);
	checkCudaErrors(cudaMemcpyAsync(&_h_bool, _d_bool, sizeof(bool), cudaMemcpyDeviceToHost));

	return _h_bool;
}
*/


template <>
uint32_t SyncMem<float>::bound_mem() {
	//const float* d_mem = device_mem();
	//float asum = 0;
	//checkCudaErrors(cublasSasum(Cuda::cublasHandle, static_cast<int>(_size), d_mem, 1, &asum));
	//float bound = 1000*(asum / _size);

	/*
	float* h_mem = mutable_host_mem();
	//double average = 0.0;
	int updateCount = 0;
	for(size_t i = 0; i < _size; i++) {
		//if(std::abs(h_mem[i]) > bound) {
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
	 */

	float* d_mem = mutable_device_mem();
	const float bound = MEM_MAX;
	//const float bound = 1.0;
	_h_int = 0;
	checkCudaErrors(cudaMemcpy(_d_int, &_h_int, sizeof(uint32_t), cudaMemcpyHostToDevice));
	BoundMem<<<RoundUp((unsigned int)_size, BW), BW>>>(d_mem, bound, _d_int, (unsigned int)_size);
	checkCudaErrors(cudaMemcpyAsync(&_h_int, _d_int, sizeof(uint32_t), cudaMemcpyDeviceToHost));

	return _h_int;
}






//template bool SyncMem<float>::is_nan_mem();
//template bool SyncMem<float>::is_inf_mem();






