/*
 * MathFunctions.h
 *
 *  Created on: Nov 25, 2016
 *      Author: jkim
 */

#ifndef SOOOA_MATHFUNCTIONS_H_
#define SOOOA_MATHFUNCTIONS_H_

#include "common.h"

template <typename Dtype>
void soooa_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void soooa_set(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void soooa_gpu_set(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void soooa_gpu_axpy(const uint32_t N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void soooa_gpu_scal(const uint32_t N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void soooa_gpu_sub(const uint32_t N, const Dtype* a, const Dtype* b, Dtype* y);

/**
 * element-wise multiplication
 */
template <typename Dtype>
void soooa_gpu_mul(const uint32_t N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void soooa_gpu_dot(const uint32_t n, const Dtype* x, const Dtype* y, Dtype* out);

template <typename Dtype>
void soooa_gpu_axpby(const uint32_t N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
void soooa_gpu_asum(const int n, const Dtype* x, Dtype* y);





void soooa_gpu_memcpy(const size_t N, const void *X, void *Y);


#endif /* SOOOA_MATHFUNCTIONS_H_ */
