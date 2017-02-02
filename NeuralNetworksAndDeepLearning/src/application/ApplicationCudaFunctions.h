/*
 * ApplicationCudaFunctions.h
 *
 *  Created on: Jan 31, 2017
 *      Author: jkim
 */

#ifndef APPLICATIONCUDAFUNCTIONS_H_
#define APPLICATIONCUDAFUNCTIONS_H_

#include "common.h"

template <typename Dtype>
void diff_content_loss(const uint32_t n, const Dtype* f, const Dtype* p, Dtype* df);

template <typename Dtype>
void diff_style_loss(const uint32_t n, const Dtype* f, Dtype* a);

#endif /* APPLICATIONCUDAFUNCTIONS_H_ */
