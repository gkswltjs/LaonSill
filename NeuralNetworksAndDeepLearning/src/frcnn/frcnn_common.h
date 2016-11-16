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



const uint32_t GT = 0;
const uint32_t GE = 1;
const uint32_t EQ = 2;
const uint32_t LE = 3;
const uint32_t LT = 4;

const float TRAIN_FG_THRESH = 0.5f;
const float TRAIN_BG_THRESH_HI = 0.5f;
const float TRAIN_BG_THRESH_LO = 0.0f;
const float TRAIN_BBOX_THRESH = 0.5f;

template <typename Dtype>
static void printArray(const std::string& name, const std::vector<Dtype>& array,
		const bool printName=true) {
	if (printName) {
		std::cout << name << ": " << std::endl;
	}

	const uint32_t arraySize = array.size();
	std::cout << "[ ";
	for (uint32_t i = 0; i < arraySize; i++) {
		if (i < arraySize-1) {
			std::cout << array[i] << ", ";
		} else {
			std::cout << array[i];
		}
	}
	std::cout << "]" << std::endl;
}

template <typename Dtype>
static void print2dArray(const std::string& name, const std::vector<std::vector<Dtype>>& array,
		const bool printName=true) {
	if (printName) {
		std::cout << name << ": " << std::endl;
	}

	const uint32_t arraySize = array.size();
	std::cout << "[ " << std::endl;
	for (uint32_t i = 0; i < arraySize; i++) {
		printArray(name, array[i], false);
	}
	std::cout << "]" << std::endl;
}

template <typename Dtype>
static void printPrimitive(const std::string& name, const Dtype data,
		const bool printName=true) {
	std::cout << name << ": " << data << std::endl;
}


template <typename Dtype>
static void np_maxByAxis(std::vector<std::vector<Dtype>>& array, std::vector<Dtype>& result) {
	const uint32_t numArrayElem = array.size();
	assert(numArrayElem >= 1);
	const uint32_t numAxisElem = array[0].size();
	assert(numAxisElem >= 1);

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

template <typename Dtype>
static void np_argmax(const std::vector<std::vector<Dtype>>& array, std::vector<uint32_t>& result) {
	const uint32_t numArrayElem = array.size();
	assert(numArrayElem >= 1);
	const uint32_t numAxisElem = array[0].size();
	assert(numAxisElem >= 1);

	result.clear();
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


static void np_where_s(std::vector<float>& array, uint32_t comp, float criteria,
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
	default:
		std::cout << "invalid comp: " << comp << std::endl;
		exit(1);
	}
}

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
