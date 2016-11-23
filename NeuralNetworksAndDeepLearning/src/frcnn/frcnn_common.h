/*
 * frcnn_common.h
 *
 *  Created on: Nov 14, 2016
 *      Author: jkim
 */

#ifndef FRCNN_COMMON_H_
#define FRCNN_COMMON_H_


#include <iostream>
#include <ostream>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>

#include "common.h"
#include "Data.h"



const uint32_t GT = 0;
const uint32_t GE = 1;
const uint32_t EQ = 2;
const uint32_t LE = 3;
const uint32_t LT = 4;
const uint32_t NE = 5;

const float TRAIN_FG_THRESH = 0.5f;
const float TRAIN_BG_THRESH_HI = 0.5f;
const float TRAIN_BG_THRESH_LO = 0.0f;
const float TRAIN_BBOX_THRESH = 0.5f;

template <typename Dtype>
static void printArray(const std::string& name, std::vector<Dtype>& array,
		const bool printName=true, const bool landscape=false) {
	if (printName) {
		std::cout << name << ": " << array.size() << std::endl;
	}

	const uint32_t arraySize = array.size();
	std::cout << "[ ";
	for (uint32_t i = 0; i < arraySize; i++) {
		if (!landscape) {
			std::cout << i << "\t\t: ";
		}

		std::cout << array[i];

		if (landscape)
			std::cout << ", ";
		else
			std::cout << std::endl;

	}
	std::cout << "]" << std::endl;
}

/*
static void printArray(const std::string& name, std::vector<float>& array,
		const bool printName=true, const bool landscape=false) {
	if (printName) {
		std::cout << name << ": " << array.size() << std::endl;
	}

	const uint32_t arraySize = array.size();
	std::cout << "[ ";
	for (uint32_t i = 0; i < arraySize; i++) {
		if (!landscape) {
			std::cout << i << "\t\t: ";
		}

		std::cout << "" << array[i] << ", ";

		if (landscape)
			std::cout << ", ";
		else
			std::cout << std::endl;

	}
	std::cout << "]" << std::endl;
}
*/




template <typename Dtype>
static void print2dArray(const std::string& name, std::vector<std::vector<Dtype>>& array,
		const bool printName=true) {
	if (printName) {
		std::cout << name << ": " << array.size();
		if (array.size() > 0)
			std::cout << " x " << array[0].size() << std::endl;
		else
			std::cout << " x 0" << std::endl;
	}

	const uint32_t arraySize = array.size();
	std::cout << "[ " << std::endl;
	for (uint32_t i = 0; i < arraySize; i++) {
		std::cout << i << "\t\t: ";
		printArray(name, array[i], false, true);
	}
	std::cout << "]" << std::endl;
}

template <typename Dtype>
static void printPrimitive(const std::string& name, const Dtype data,
		const bool printName=true) {
	std::cout << name << ": " << data << std::endl;
}


/**
 * 각 row의 vector에서 최대값을 추출
 */
template <typename Dtype>
static void np_maxByAxis(const std::vector<std::vector<Dtype>>& array,
		std::vector<Dtype>& result) {
	const uint32_t numArrayElem = array.size();
	assert(numArrayElem > 0);
	const uint32_t numAxisElem = array[0].size();
	assert(numAxisElem > 0);

	result.clear();
	result.resize(numArrayElem);
	Dtype max;
	for (uint32_t i = 0; i < numArrayElem; i++) {
		for (uint32_t j = 0; j < numAxisElem; j++) {
			if (j == 0) max = array[i][0];
			else if (array[i][j] > max) {
				max = array[i][j];
			}
		}
		result[i] = max;
	}
}

/**
 * 해당 axis에서 index에 해당하는 값을 조회
 */
template <typename Dtype>
static void np_array_value_by_index_array(const std::vector<std::vector<Dtype>>& array,
		const uint32_t axis, const std::vector<uint32_t>& inAxisIndex,
		std::vector<Dtype>& result) {
	assert(axis >= 0 && axis < 2);

	const uint32_t numArrayElem = array.size();
	assert(numArrayElem > 0);
	const uint32_t numAxisElem = array[0].size();
	assert(numAxisElem > 0);

	result.clear();

	if (axis == 0) {
		result.resize(numAxisElem);
		for (uint32_t i = 0; i < numAxisElem; i++) {
			result[i] = array[inAxisIndex[i]][i];
		}
	} else if (axis == 1) {
		result.resize(numArrayElem);
		for (uint32_t i = 0; i < numArrayElem; i++) {
			result[i] = array[i][inAxisIndex[i]];
		}
	}
}



/**
 * 각 column의 최대값을 추출
 */
template <typename Dtype>
static void np_array_max(const std::vector<std::vector<Dtype>>& array,
		std::vector<Dtype>& result) {
	const uint32_t numArrayElem = array.size();
	assert(numArrayElem > 0);
	const uint32_t numAxisElem = array[0].size();
	assert(numAxisElem > 0);

	result.clear();
	result.resize(numAxisElem);
	Dtype max;
	for (uint32_t j = 0; j < numAxisElem; j++) {
		for (uint32_t i = 0; i < numArrayElem; i++) {
			if (i == 0) max = array[0][j];
			else if (array[i][j] > max) {
				max = array[i][j];
			}
		}
		result[j] = max;
	}
}


template <typename Dtype>
static void np_scalar_divided_by_array(const Dtype scalar,
		const std::vector<Dtype>& array, std::vector<Dtype>& result) {

	const uint32_t arraySize = array.size();
	result.resize(arraySize);

	for (uint32_t i = 0; i < arraySize; i++) {
		result[i] = scalar / array[i];
	}
}

template <typename Dtype>
static void np_scalar_multiplied_by_array(const Dtype scalar,
		const std::vector<Dtype>& array, std::vector<Dtype>& result) {

	const uint32_t arraySize = array.size();
	result.resize(arraySize);

	for (uint32_t i = 0; i < arraySize; i++) {
		result[i] = scalar * array[i];
	}
}


template <typename Dtype1, typename Dtype2>
static void np_array_elementwise_mul(const std::vector<Dtype1>& a,
		const std::vector<Dtype2>& b, std::vector<Dtype2>& result) {

	assert(a.size() == b.size());

	const uint32_t arraySize = a.size();
	result.resize(arraySize);
	for (uint32_t i = 0; i < arraySize; i++) {
		result[i] = a[i] * b[i];
	}
}



template <typename Dtype>
static void np_sqrt(const std::vector<Dtype>& a, std::vector<Dtype>& result) {

	const uint32_t arraySize = a.size();
	result.resize(arraySize);

	for (uint32_t i = 0; i < arraySize; i++) {
		result[i] = std::sqrt(a[i]);
	}
}



template <typename Dtype>
static Dtype np_min(const std::vector<Dtype>& array,
		uint32_t begin, uint32_t end) {
	Dtype min;
	const uint32_t arraySize = array.size();
	assert(end < arraySize);

	for (uint32_t i = begin; i < end; i++) {
		if (i == 0) min = array[0];
		else if (array[i] < min) min = array[i];
	}
	return min;
}

template <typename Dtype>
static Dtype np_max(const std::vector<Dtype>& array,
		uint32_t begin, uint32_t end) {
	Dtype max;
	const uint32_t arraySize = array.size();
	assert(end < arraySize);

	for (uint32_t i = begin; i < end; i++) {
		if (i == 0) max = array[0];
		else if (array[i] > max) max = array[i];
	}
	return max;
}




template <typename Dtype>
static void np_argmax(const std::vector<std::vector<Dtype>>& array, const uint32_t axis,
		std::vector<uint32_t>& result) {
	assert(axis >= 0 && axis < 2);

	const uint32_t numArrayElem = array.size();
	assert(numArrayElem >= 1);
	const uint32_t numAxisElem = array[0].size();
	assert(numAxisElem >= 1);

	result.clear();

	if (axis == 0) {
		result.resize(numAxisElem);
		Dtype max;
		uint32_t maxIndex;
		for (uint32_t i = 0; i < numAxisElem; i++) {
			for (uint32_t j = 0; j < numArrayElem; j++) {
				if (j == 0) {
					max = array[0][i];
					maxIndex = 0;
				}
				else if (array[j][i] > max) {
					max = array[j][i];
					maxIndex = j;
				}
			}
			result[i] = maxIndex;
		}
	} else if (axis == 1) {
		result.resize(numArrayElem);
		Dtype max;
		uint32_t maxIndex;
		for (uint32_t i = 0; i < numArrayElem; i++) {
			for (uint32_t j = 0; j < numAxisElem; j++) {
				if (j == 0) {
					max = array[i][0];
					maxIndex = 0;
				}
				else if (array[i][j] > max) {
					max = array[i][j];
					maxIndex = j;
				}
			}
			result[i] = maxIndex;
		}
	}
}

/**
 * np_where 단수 조건 버전, compare 옵션 있음.
 */
template <typename Dtype, typename Dtype2>
static void np_where_s(const std::vector<Dtype>& array, const uint32_t comp, const Dtype2 criteria,
		std::vector<uint32_t>& result) {
	const uint32_t numArrayElem = array.size();
	assert(numArrayElem > 0);

	switch (comp) {
	case GT:
		for (uint32_t i = 0; i < numArrayElem; i++) if (array[i] > criteria) result.push_back(i);
		break;
	case GE:
		for (uint32_t i = 0; i < numArrayElem; i++) if (array[i] >= criteria) result.push_back(i);
		break;
	case EQ:
		for (uint32_t i = 0; i < numArrayElem; i++) if (array[i] == criteria) result.push_back(i);
		break;
	case LE:
		for (uint32_t i = 0; i < numArrayElem; i++) if (array[i] <= criteria) result.push_back(i);
		break;
	case LT:
		for (uint32_t i = 0; i < numArrayElem; i++) if (array[i] < criteria) result.push_back(i);
		break;
	case NE:
		for (uint32_t i = 0; i < numArrayElem; i++) if (array[i] != criteria) result.push_back(i);
		break;
	default:
		std::cout << "invalid comp: " << comp << std::endl;
		exit(1);
	}
}


/**
 * np_where 단수 조건 버전, equality에 대해서만 테스트
 */
template <typename Dtype>
static void np_where_s(const std::vector<std::vector<Dtype>>& array, const Dtype criteria,
		const uint32_t loc, std::vector<uint32_t>& result) {
	const uint32_t numArrayElem = array.size();
	assert(numArrayElem > 0);
	assert(loc < array[0].size());

	result.clear();
	for (uint32_t i = 0; i < numArrayElem; i++) {
		if (array[i][loc] == criteria) {
			result.push_back(i);
		}
	}
}

/**
 * np_where 단수 조건 버전, equality에 대해서만 테스트
 * 한 row의 기준값과 array값에 대해 각각의 axis에 대해 하나라도 일치할 경우 찾은 것으로 판정
 */
template <typename Dtype>
static void np_where_s(const std::vector<std::vector<Dtype>>& array,
		const std::vector<Dtype>& criteria,
		std::vector<uint32_t>& result) {
	if (array.size() < 1) return;

	assert(array[0].size() == criteria.size());
	result.clear();

	const uint32_t numArrayElem = array.size();
	const uint32_t numAxisElem = criteria.size();

	for (uint32_t i = 0; i < numArrayElem; i++) {
		for (uint32_t j = 0; j < numAxisElem; j++) {
			if (array[i][j] == criteria[j]) {
				result.push_back(i);
				break;
			}
		}
	}
}






/**
 * np_where 복수 조건 버전
 */
static void np_where(std::vector<float>& array, const std::vector<uint32_t>& comp,
		const std::vector<float>& criteria, std::vector<uint32_t>& result) {

	if (comp.size() < 1 ||
			criteria.size() < 1 ||
			comp.size() != criteria.size() ||
			array.size() < 1) {

		std::cout << "invalid array dimension ... " << std::endl;
		exit(1);
	}

	result.clear();

	const uint32_t numComps = comp.size();
	const uint32_t numArrayElem = array.size();
	bool cond;
	for (uint32_t i = 0; i < numArrayElem; i++) {
		cond = true;

		for (uint32_t j = 0; j < numComps; j++) {
			switch (comp[j]) {
			case GT: if (array[i] <= criteria[j])	cond = false; break;
			case GE: if (array[i] <	criteria[j])	cond = false; break;
			case EQ: if (array[i] != criteria[j])	cond = false; break;
			case LE: if (array[i] > criteria[j])	cond = false; break;
			case LT: if (array[i] >= criteria[j])	cond = false; break;
			default:
				std::cout << "invalid comp: " << comp[j] << std::endl;
				exit(1);
			}
			if (!cond) break;
		}
		if (cond) result.push_back(i);
	}
}




template <typename Dtype>
static void np_tile(const std::vector<Dtype>& array, const uint32_t repeat,
		std::vector<std::vector<Dtype>>& result) {

	result.clear();
	for (uint32_t i = 0; i < repeat; i++) {
		result.push_back(array);
	}
}

static uint32_t np_round(float a) {
	uint32_t lower = static_cast<uint32_t>(floorf(a) + 0.1f);
	if (lower % 2 == 0) return lower;

	uint32_t upper = static_cast<uint32_t>(ceilf(a) + 0.1f);
	return upper;
}

static void np_round(const std::vector<float>& a, std::vector<uint32_t>& result) {
	const uint32_t arraySize = a.size();
	result.resize(arraySize);

	for (uint32_t i = 0; i < arraySize; i++) {
		//result[i] = np_round(a[i]);
		result[i] = roundf(a[i]);
	}
}



static void npr_randint(const uint32_t lb, const uint32_t ub, const uint32_t size,
		std::vector<uint32_t>& result) {

	srand((uint32_t)time(NULL));
	result.resize(size);

	for (uint32_t i = 0; i < size; i++) {
		result[i] = lb + rand()%(ub-lb);
	}
}

template <typename Dtype>
static void npr_choice(const std::vector<Dtype>& array, const uint32_t size,
		std::vector<Dtype>& result) {

	const uint32_t arraySize = array.size();
	assert(size <= arraySize);

	std::vector<uint32_t> indexArray(arraySize);
	iota(indexArray.begin(), indexArray.end(), 0);
	random_shuffle(indexArray.begin(), indexArray.end());

	result.resize(size);
	for (uint32_t i = 0; i < size; i++) {
		result[i] = array[indexArray[i]];
	}
	printArray("result-final", result);
}

static std::vector<uint32_t> np_arange(int start, int stop) {
	assert(start < stop);

	std::vector<uint32_t> result;
	for (int i = start; i < stop; i++) {
		result.push_back(i);
	}
	return result;
}


template <typename Dtype>
static void py_arrayElemsWithArrayInds(const std::vector<Dtype>& array,
		const std::vector<uint32_t>& inds, std::vector<Dtype>& result) {

	const uint32_t arraySize = array.size();
	const uint32_t indsSize = inds.size();

	//result.clear();
	result.resize(indsSize);
	for (uint32_t i = 0; i < indsSize; i++) {
		assert(inds[i] < arraySize);
		result[i] = array[inds[i]];
	}
}


template <typename Dtype>
static Dtype vec_max(const std::vector<Dtype>& array) {
	Dtype max;
	const uint32_t arraySize = array.size();
	for (uint32_t i = 0; i < arraySize; i++) {
		if (i == 0) max = array[i];
		else if (max < array[i]) max = array[i];
	}
	return max;
}


template <typename Dtype, typename Dtype2>
static void fillDataWith2dVec(const std::vector<std::vector<Dtype>>& array,
		Data<Dtype2>* data) {
	assert(array.size() > 0);

	const uint32_t dim1 = array.size();
	const uint32_t dim2 = array[0].size();

	data->shape({1, 1, dim1, dim2});
	Dtype2* dataPtr = data->mutable_host_data();

	for (uint32_t i = 0; i < dim1; i++) {
		memcpy(dataPtr + i*dim2, &array[i][0], sizeof(Dtype)*dim2);
	}
}

/*
template <typename Dtype, typename Dtype2>
static void fillDataWith2dVec(const std::vector<std::vector<Dtype>>& array,
		const std::vector<uint32_t>& transpose,	Data<Dtype2>* data) {
	assert(array.size() > 0);
	const uint32_t dim1 = array.size();
	const uint32_t dim2 = array[0].size();

	const std::vector<uint32_t>& shape = data->getShape();
	assert(shape[3]%dim2 == 0);

	const uint32_t tBatchSize = shape[transpose[1]]*shape[transpose[2]]*shape[transpose[3]];
	const uint32_t tHeightSize = shape[transpose[2]]*shape[transpose[3]];
	const uint32_t tWidthSize = shape[transpose[3]];

	Dtype2* dataPtr = data->mutable_host_data();
	const uint32_t shape3 = shape[3] / dim2;
	const uint32_t batchSize = shape[1]*shape[2]*shape3;
	const uint32_t heightSize = shape[2]*shape3;
	const uint32_t widthSize = shape3;

	uint32_t s[4];
	uint32_t& ts0 = s[transpose[0]];
	uint32_t& ts1 = s[transpose[1]];
	uint32_t& ts2 = s[transpose[2]];
	uint32_t& ts3 = s[transpose[3]];

	uint32_t q, r;
	// batch
	for (s[0] = 0; s[0] < shape[0]; s[0]++) {
		// height
		for (s[1] = 0; s[1] < shape[1]; s[1]++) {
			// width
			for (s[2] = 0; s[2] < shape[2]; s[2]++) {
				// Anchors
				for (s[3] = 0; s[3] < shape[3]; s[3]++) {
					q = s[3] / dim2;
					r = s[3] % dim2;
					dataPtr[ts0*tBatchSize+ts1*tHeightSize+ts2*tWidthSize+ts3] =
							array[s[0]*batchSize+s[1]*heightSize+s[2]*widthSize+q][r];
				}
			}
		}
	}
}

template <typename Dtype, typename Dtype2>
static void fillDataWith1dVec(const std::vector<Dtype>& array,
		const std::vector<uint32_t>& transpose,
		Data<Dtype2>* data) {
	assert(array.size() > 0);
	const uint32_t dim1 = array.size();

	const std::vector<uint32_t>& shape = data->getShape();

	Dtype2* dataPtr = data->mutable_host_data();
	const uint32_t batchSize = shape[1]*shape[2]*shape[3];
	const uint32_t heightSize = shape[2]*shape[3];
	const uint32_t widthSize = shape[3];

	const uint32_t tBatchSize = shape[transpose[1]]*shape[transpose[2]]*shape[transpose[3]];
	const uint32_t tHeightSize = shape[transpose[2]]*shape[transpose[3]];
	const uint32_t tWidthSize = shape[transpose[3]];

	uint32_t s[4];
	uint32_t& ts0 = s[transpose[0]];
	uint32_t& ts1 = s[transpose[1]];
	uint32_t& ts2 = s[transpose[2]];
	uint32_t& ts3 = s[transpose[3]];

	// batch
	for (s[0] = 0; s[0] < shape[0]; s[0]++) {
		// height
		for (s[1] = 0; s[1] < shape[1]; s[1]++) {
			// width
			for (s[2] = 0; s[2] < shape[2]; s[2]++) {
				// Anchors
				for (s[3] = 0; s[3] < shape[3]; s[3]++) {
					dataPtr[ts0*tBatchSize+ts1*tHeightSize+ts2*tWidthSize+ts3]
					        = Dtype2(array[s[0]*batchSize+s[1]*heightSize+s[2]*widthSize+s[3]]);
				}
			}
		}
	}
}
*/






template <typename Dtype, typename Dtype2>
static void fill1dVecWithData(Data<Dtype>* data,
		std::vector<Dtype2>& array) {

	const std::vector<uint32_t>& dataShape = data->getShape();
	assert(dataShape[0] == 1);
	assert(dataShape[1] == 1);
	assert(dataShape[2] == 1);

	const uint32_t dim1 = dataShape[3];
	array.resize(dim1);
	const Dtype* dataPtr = data->host_data();

	for (uint32_t i = 0; i < dim1; i++) {
		array[i] = Dtype2(dataPtr[i]);
	}
}

template <typename Dtype, typename Dtype2>
static void fill2dVecWithData(Data<Dtype>* data,
		std::vector<std::vector<Dtype2>>& array) {

	const std::vector<uint32_t>& dataShape = data->getShape();
	assert(dataShape[0] == 1);
	assert(dataShape[1] == 1);

	const uint32_t dim1 = dataShape[2];
	const uint32_t dim2 = dataShape[3];

	array.resize(dim1);
	const Dtype* dataPtr = data->host_data();

	for (uint32_t i = 0; i < dim1; i++) {
		array[i].resize(dim2);
		for (uint32_t j = 0; j < dim2; j++) {
			array[i][j] = Dtype2(dataPtr[i*dim2+j]);
		}
	}
}



/*
static std::string cv_type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch ( depth ) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans+'0');

	return r;
}
*/





struct Size {
	uint32_t width;
	uint32_t height;
	uint32_t depth;

	void print() {
		std::cout << "Size:" << std::endl <<
				"\twidth: " << width << std::endl <<
				"\theight: " << height << std::endl <<
				"\tdepth: " << depth << std::endl;
	}
};

struct Object {
	std::string name;
	uint32_t label;
	uint32_t difficult;
	uint32_t xmin;
	uint32_t ymin;
	uint32_t xmax;
	uint32_t ymax;

	void print() {
		std::cout << "Object: " << std::endl <<
				"\tname: " << name << std::endl <<
				"\tlabel: " << label << std::endl <<
				"\tdifficult: " << difficult << std::endl <<
				"\txmin: " << xmin << std::endl <<
				"\tymin: " << ymin << std::endl <<
				"\txmax: " << xmax << std::endl <<
				"\tymax: " << ymax << std::endl;
	}
};

struct Annotation {
	std::string filename;
	Size size;
	std::vector<Object> objects;

	void print() {
		std::cout << "Annotation:" << std::endl <<
				"\tfilename: " << filename << std::endl;
		size.print();
		for (uint32_t i = 0; i < objects.size(); i++) {
			objects[i].print();
		}
	}
};



#endif /* FRCNN_COMMON_H_ */
