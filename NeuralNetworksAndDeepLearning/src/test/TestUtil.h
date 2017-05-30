/*
 * TestUtil.h
 *
 *  Created on: Feb 16, 2017
 *      Author: jkim
 */

#ifndef TESTUTIL_H_
#define TESTUTIL_H_

#include <map>
#include <vector>
#include <string>

#include "cnpy.h"
#include "Data.h"

#if 0

const std::string DELIM 		= "*";
const std::string TYPE_DATA 	= "data";
const std::string TYPE_DIFF 	= "diff";
const std::string SIG_BOTTOM 	= DELIM + "bottom" + DELIM;
const std::string SIG_TOP 		= DELIM + "top" + DELIM;
const std::string SIG_PARAMS 	= DELIM + "params" + DELIM;
const std::string BLOBS_PREFIX	= "anonymous" + DELIM + "blobs" + DELIM;
const std::string NPZ_PATH 		= "/home/monhoney/caffe_soa_test/save/";

const float COMPARE_ERROR 		= 0.00001;


// cuda device 설정
// cublas, cudnn handle 생성
void setUpCuda(int gpuid);
// cublas, cudnn handle 파괴
void cleanUpCuda();

// npz_path의 npz file로부터 layer_name에 해당하는 레이어 데이터를 조회, nameDataMap을 채움
void buildNameDataMapFromNpzFile(const std::string& npz_path, const std::string& layer_name,
		std::map<std::string, Data<float>*>& nameDataMap);

// dataNameVec에 해당하는 만큼 data 객체를 생성, dataVec에 추가
void fillLayerDataVec(const std::vector<std::string>& dataNameVec,
		std::vector<Data<float>*>& dataVec);
// dataVec에 data_prefix에 해당하는 Data를 nameDataMap으로부터 조회하여 복사
void fillData(std::map<std::string, Data<float>*>& nameDataMap,
		const std::string& data_prefix, std::vector<Data<float>*>& dataVec);
// paramVec에 param_prefix에 해당하는 Data를 nameDataMap으로부터 조회하여 복사
void fillParam(std::map<std::string, Data<float>*>& nameDataMap,
		const std::string& param_prefix, std::vector<Data<float>*>& paramVec);

//
bool compareData(std::map<std::string, Data<float>*>& nameDataMap,
		const std::string& data_prefix, std::vector<Data<float>*>& dataVec,
		uint32_t compareType);
bool compareData(std::map<std::string, Data<float>*>& nameDataMap,
		const std::string& data_prefix, Data<float>* targetData, uint32_t compareType);

bool compareParam(std::map<std::string, Data<float>*>& nameDataMap,
		const std::string& param_prefix, std::vector<Data<float>*>& paramVec,
		uint32_t compareType);


void printNpzFiles(cnpy::npz_t& cnpy_npz);
void printNameDataMap(std::map<std::string, Data<float>*>& nameDataMap, bool printData);
void printData(std::vector<Data<float>*>& dataVec);

void printConfigOn();
void printConfigOff();


template <typename T, typename S>
void cleanUpMap(std::map<T, S*>& dict);

template <typename T>
void cleanUpObject(T* obj);

template <typename T, typename S>
S* retrieveValueFromMap(std::map<T, S*>& dict, const T& key);



#endif

#endif /* TESTUTIL_H_ */
