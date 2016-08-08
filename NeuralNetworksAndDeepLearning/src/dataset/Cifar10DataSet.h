/**
 * @file	Cifar10DataSet.h
 * @date	2016/5/19
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef DATASET_CIFAR10DATASET_H_
#define DATASET_CIFAR10DATASET_H_

#include <armadillo>
#include <vector>
#include <string>

#include "DataSet.h"
#include "DataSample.h"
#include "ImageInfo.h"

using namespace std;
using namespace arma;


/**
 * @brief cifar10 데이터를 읽어 들이기 위해 DataSet을 구현한 클래스
 * @details
 * @todo 사용할 경우 전반적인 수정이 필요. 현재 사용하지 않음.
 */
class Cifar10DataSet : public DataSet {
public:
	Cifar10DataSet();
	virtual ~Cifar10DataSet();

	virtual void load();
	void shuffleTrainDataSet();
	void shuffleValidationDataSet();
	void shuffleTestDataSet();


private:
	/**
	 * @details 지정된 데이터 파일 경로로부터 지정된 데이터셋 메모리로 데이터를 읽어들인다.
	 * @param resources 데이터 파일 경로들을 담고 있는 포인터.
	 * @param numResources 데이터 파일 경로들 중 사용할 파일의 수.
	 * @param dataSet 학습,유효,테스트 데이터셋 중 데이터를 로드할 ...
	 * @param dataSize 읽어들일 데이터의 수.
	 * @return 읽어들인 데이터의 수.
	 */
	int loadDataSetFromResource(string *resources, int numResources, DataSample *&dataSet, int dataSize);
	double validationSetRatio;			///< 학습데이터에서 학습데이터와 유효데이터의 비율. (적용하지 않고 있음)
};

#endif /* DATASET_CIFAR10DATASET_H_ */
