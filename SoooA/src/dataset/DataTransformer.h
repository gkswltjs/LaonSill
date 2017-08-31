/*
 * DataTransformer.h
 *
 *  Created on: Jul 19, 2017
 *      Author: jkim
 */

#ifndef DATATRANSFORMER_H_
#define DATATRANSFORMER_H_

#include <opencv2/core/core.hpp>

#include "Datum.h"
#include "Data.h"


class DataTransformParam {
public:
	DataTransformParam() {
		this->mean.clear();
		this->scale = 1.0f;
		this->cropSize = 0;
		this->mirror = false;
	}

	DataTransformParam(const std::vector<float> mean, const float scale, const int cropSize,
			const bool mirror) {
		this->mean = mean;
		this->scale = scale;
		this->cropSize = cropSize;
		this->mirror = mirror;
	}

public:
	std::vector<float> mean;
	float scale;
	int cropSize;
	bool mirror;
};






template <typename Dtype>
class DataTransformer {
public:
	DataTransformer(DataTransformParam* param = NULL);
	virtual ~DataTransformer();


	void transformWithMeanScale(Datum* datum, const std::vector<float>& mean,
			const float scale, Dtype* dataPtr);

	void transform(Datum* datum, Dtype* dataPtr);
	void transform(cv::Mat& im, Data<Dtype>* data, int batchIdx = 0);

private:
	int rand(int n);


private:
	DataTransformParam param;
	bool hasMean;
	bool hasCropSize;
	bool hasScale;
	bool hasMirror;

};















/**
 * from old DataTransformer.h
 */
template <typename Dtype>
void transformInv(const int num, const int singleImageSize,
		const int imageHeight, const int imageWidth,
		const int height, const int width, const std::vector<Dtype>& pixelMeans,
		const Dtype* dataData, Data<Dtype>& temp);












#endif /* DATATRANSFORMER_H_ */
