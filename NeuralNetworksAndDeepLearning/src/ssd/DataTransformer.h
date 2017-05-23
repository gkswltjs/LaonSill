/*
 * DataTransformer.h
 *
 *  Created on: May 17, 2017
 *      Author: jkim
 */

#ifndef DATATRANSFORMER_H_
#define DATATRANSFORMER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

#include "Data.h"

template <typename Dtype>
void transformInv(const int num, const int singleImageSize,
		const int imageHeight, const int imageWidth,
		const int height, const int width, const std::vector<Dtype>& pixelMeans,
		const Dtype* dataData, Data<Dtype>& temp);


#endif /* DATATRANSFORMER_H_ */
