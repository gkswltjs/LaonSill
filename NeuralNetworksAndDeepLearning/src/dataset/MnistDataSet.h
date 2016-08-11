/**
 * @file	MnistDataSet.h
 * @date	2016/4/23
 * @author	jhkim
 * @brief
 * @details
 */



#ifndef MNISTDATASET_DATASET_H_
#define MNISTDATASET_DATASET_H_

#include <string>
#include <vector>

#include "UbyteDataSet.h"

using namespace std;
using namespace arma;


/**
 * @brief Mnist 데이터 셋을 로드하기 위해 구현된 DataSet 클래스.
 * @details http://yann.lecun.com/exdb/mnist/의 데이터를 사용.
 * @todo UbyteDataSet이 channel값을 읽도록 수정되었고, label값을 1000범위까지 읽도록(unsigned int) 수정되어
 *       unsigned char로 label이 저장된 현재의 original 파일을 읽을 경우 error가 발생할 것.
 */
class MnistDataSet : public UbyteDataSet {
public:
	MnistDataSet(double validationSetRatio)
		: UbyteDataSet(
				"/home/jhkim/data/learning/mnist/train-images.idx3-ubyte",
				"/home/jhkim/data/learning/mnist/train-labels.idx1-ubyte",
				1,
				"/home/jhkim/data/learning/mnist/t10k-images.idx3-ubyte",
				"/home/jhkim/data/learning/mnist/t10k-labels.idx1-ubyte",
				1,
				1,
				validationSetRatio) {
		this->channels = 1;
		this->mean[0] = 0.13066047740;
	}
	virtual ~MnistDataSet() {}

	/*
	virtual void load();
	void shuffleTrainDataSet();
	void shuffleValidationDataSet();
	void shuffleTestDataSet();
	*/

	/*
private:
#ifndef GPU_MODE
	int loadDataSetFromResource(string resources[2], DataSample *&dataSet, int offset, int size);
#else
	int loadDataSetFromResource(string resources[2], vector<DATATYPE> *&dataSet, vector<UINT> *&labelSet, int offset, int size);
#endif

	double validationSetRatio;
	*/
};

#endif /* MNISTDATASET_DATASET_H_ */
















