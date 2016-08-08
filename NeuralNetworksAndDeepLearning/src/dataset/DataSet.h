/**
 * @file DataSet.h
 * @date 2016/4/21
 * @author jhkim
 * @brief
 * @details
 */

#ifndef DATASET_DATASET_H_
#define DATASET_DATASET_H_

#include <armadillo>
#include <vector>
#include <string>
#include "ImageInfo.h"
#include "DataSample.h"
#include "../exception/Exception.h"

using namespace std;
using namespace arma;



/**
 * @brief 데이터셋 최상위 추상 클래스. 데이터셋의 특성에 따라 DataSet 클래스를 상속하여 구현.
 * @details
 */
class DataSet {
public:
	DataSet() {
		mean[0] = 0;
		mean[1] = 0;
		mean[2] = 0;

		trainDataSet = 0;
		trainLabelSet = 0;
		validationDataSet = 0;
		validationLabelSet = 0;
		testDataSet = 0;
		testLabelSet = 0;
	}
	DataSet(UINT rows, UINT cols, UINT channels, UINT numTrainData, UINT numTestData) {
		this->rows = rows;
		this->cols = cols;
		this->channels = channels;
		this->dataSize = rows*cols*channels;
		this->numTrainData = numTrainData;
		this->numTestData = numTestData;

		trainDataSet = new vector<DATATYPE>(this->dataSize*numTrainData);
		trainLabelSet = new vector<UINT>(numTrainData);
		testDataSet = new vector<DATATYPE>(this->dataSize*numTestData);
		testLabelSet = new vector<UINT>(numTestData);

		mean[0] = 0;
		mean[1] = 0;
		mean[2] = 0;
	}
	virtual ~DataSet() {
		if(trainDataSet) delete trainDataSet;
		if(trainLabelSet) delete trainLabelSet;
		if(validationDataSet) delete validationDataSet;
		if(validationLabelSet) delete validationLabelSet;
		if(testDataSet) delete testDataSet;
		if(testLabelSet) delete testLabelSet;
	}

	UINT getRows() const { return this->rows; }
	UINT getCols() const { return this->cols; }
	UINT getChannels() const { return this->channels; }
	UINT getNumTrainData() const { return this->numTrainData; }
	UINT getNumValidationData() const { return this->numValidationData; }
	UINT getNumTestData() const { return this->numTestData; }

	/**
	 * @details index번째 학습데이터에 대한 포인터 조회.
	 * @param index 조회하고자 하는 학습 데이터의 index
	 * @return index번째 학습데이터에 대한 포인터.
	 */
	virtual const DATATYPE *getTrainDataAt(int index) {
		if(index >= numTrainData) throw Exception();
		return &(*trainDataSet)[dataSize*index];
	}
	/**
	 * @details index번째 학습데이터의 정답 레이블에 대한 포인터 조회.
	 * @param index 조회하고자 하는 학습 데이터 정답 레이블의 index
	 * @return index번째 학습데이터 정답 레이블에 대한 포인터.
	 */
	virtual const UINT *getTrainLabelAt(int index) {
		if(index >= numTrainData) throw Exception();
		return &(*trainLabelSet)[index];
	}
	/**
	 * @details index번째 유효데이터에 대한 포인터 조회.
	 * @param index 조회하고자 하는 유효 데이터의 index
	 * @return index번째 유효데이터에 대한 포인터.
	 */
	virtual const DATATYPE *getValidationDataAt(int index) {
		if(index >= numValidationData) throw Exception();
		return &(*validationDataSet)[dataSize*index];
	}
	/**
	 * @details index번째 유효데이터의 정답 레이블에 대한 포인터 조회.
	 * @param index 조회하고자 하는 유효 데이터 정답 레이블의 index
	 * @return index번째 유효데이터 정답 레이블에 대한 포인터.
	 */
	virtual const UINT *getValidationLabelAt(int index) {
		if(index >= numValidationData) throw Exception();
		return &(*validationLabelSet)[index];
	}
	/**
	 * @details index번째 테스트데이터에 대한 포인터 조회.
	 * @param index 조회하고자 하는 테스트데이터의 index
	 * @return index번째 테스트데이터에 대한 포인터.
	 */
	virtual const DATATYPE *getTestDataAt(int index) {
		if(index >= numTestData) throw Exception();
		return &(*testDataSet)[dataSize*index];
	}
	/**
	 * @details index번째 테스트데이터의 정답 레이블에 대한 포인터 조회.
	 * @param index 조회하고자 하는 테스트데이터 정답 레이블의 index
	 * @return index번째 테스트데이터 정답 레이블에 대한 포인터.
	 */
	virtual const UINT *getTestLabelAt(int index) {
		if(index >= numTestData) throw Exception();
		return &(*testLabelSet)[index];
	}

	/**
	 * @details 학습데이터에 대한 포인터 조회.
	 * @return 학습데이터에 대한 포인터.
	 */
	const vector<DATATYPE> *getTrainDataSet() const { return this->trainDataSet; }
	/**
	 * @details 유효데이터에 대한 포인터 조회.
	 * @return 유효데이터에 대한 포인터.
	 */
	const vector<DATATYPE> *getValidationDataSet() const { return this->validationDataSet; }
	/**
	 * @details 테스트데이터에 대한 포인터 조회.
	 * @return 테스트데이터에 대한 포인터.
	 */
	const vector<DATATYPE> *getTestDataSet() const { return this->testDataSet; }

	/**
	 * @details 학습,유효,테스트 데이터를 메모리로 로드.
	 */
	virtual void load() = 0;
	/**
	 * @details 학습데이터를 임의의 순서로 섞는다.
	 */
	virtual void shuffleTrainDataSet() = 0;
	/**
	 * @details 유효데이터를 임의의 순서로 섞는다.
	 */
	virtual void shuffleValidationDataSet() = 0;
	/**
	 * @details 테스트데이터를 임의의 순서로 섞는다.
	 */
	virtual void shuffleTestDataSet() = 0;

	/**
	 * @details 학습데이터의 각 채널별 평균을 구하고 학습, 유효, 테스트 데이터에 대해 평균만큼 shift.
	 * @param hasMean 이미 계산된 평균값이 있는지 여부, 미리 계산된 평균값이 있는 경우 다시 평균을 계산하지 않는다.
	 */
	void zeroMean(bool hasMean=false) {
		//cout << "mean_0: " << mean[0] << ", mean_1: " << mean[1] << ", mean_2: " << mean[2] << endl;
		UINT di, ci, hi, wi;
		double sum[3] = {0.0, 0.0, 0.0};

		if(!hasMean) {
			for(di = 0; di < numTrainData; di++) {
				for(ci = 0; ci < channels; ci++) {
					for(hi = 0; hi < rows; hi++) {
						for(wi = 0; wi < cols; wi++) {
							sum[ci] += (*trainDataSet)[wi+hi*cols+ci*cols*rows+di*cols*rows*channels];
						}
					}
				}
				//cout << "mean_0: " << mean[0] << ", mean_1: " << mean[1] << ", mean_2: " << mean[2] << endl;
			}

			cout << "sum_0: " << sum[0] << ", sum_1: " << sum[1] << ", sum_2: " << sum[2] << endl;
			cout << "rows: " << rows << ", cols: " << cols << ", numTrainData: " << numTrainData << endl;
			for(ci = 0; ci < channels; ci++) {
				mean[ci] = (DATATYPE)(sum[ci] / (rows*cols*numTrainData));
			}
			cout << "mean_0: " << mean[0] << ", mean_1: " << mean[1] << ", mean_2: " << mean[2] << endl;
		}

		for(di = 0; di < numTrainData; di++) {
			for(ci = 0; ci < channels; ci++) {
				for(hi = 0; hi < rows; hi++) {
					for(wi = 0; wi < cols; wi++) {
						(*trainDataSet)[wi+hi*cols+ci*cols*rows+di*cols*rows*channels] -= mean[ci];
					}
				}
			}
		}

		for(di = 0; di < numTestData; di++) {
			for(ci = 0; ci < channels; ci++) {
				for(hi = 0; hi < rows; hi++) {
					for(wi = 0; wi < cols; wi++) {
						(*testDataSet)[wi+hi*cols+ci*cols*rows+di*cols*rows*channels] -= mean[ci];
					}
				}
			}
		}
	}

	/**
	 * @details 특정 채널의 평균값을 조회한다.
	 * @param channel 조회하고자 하는 채널의 index
	 * @return 지정한 채널의 평균값.
	 */
	DATATYPE getMean(UINT channel) {
		return mean[channel];
	}
	/**
	 * @details 전체 채널의 평균값 배열의 첫번째 위치에 대한 포인터를 조회한다.
	 * @return 전체 채널 평균값 배열의 첫번째 위치에 대한 포인터.
	 */
	DATATYPE *getMean() {
		return mean;
	}

private:



protected:
	UINT rows;								///< 데이터의 rows (height)값.
	UINT cols;								///< 데이터의 cols (width)값.
	UINT channels;							///< 데이터의 channel값.
	size_t dataSize;						///< 데이터셋 데이터 하나의 요소수 (rows*cols*channels)
	UINT numTrainData;						///< 학습데이터의 수.
	UINT numValidationData;					///< 유효데이터의 수.
	UINT numTestData;						///< 테스트데이터의 수.

	//DataSample *trainDataSet;
	//DataSample *validationDataSet;
	//DataSample *testDataSet;
	vector<DATATYPE> *trainDataSet;			///< 학습데이터셋 벡터에 대한 포인터.
	vector<UINT> *trainLabelSet;			///< 학습데이터셋의 정답 레이블 벡터에 대한 포인터.
	vector<DATATYPE> *validationDataSet;	///< 유효데이터셋 벡터에 대한 포인터.
	vector<UINT> *validationLabelSet;		///< 유효데이터셋의 정답 레이블 벡터에 대한 포인터.
	vector<DATATYPE> *testDataSet;			///< 테스트데이터셋 벡터에 대한 포인터.
	vector<UINT> *testLabelSet;				///< 테스트데이터셋의 정답 레이블 벡터에 대한 포인터.

	DATATYPE mean[3];						///< 학습데이터셋의 각 채널별 평균값을 저장하는 배열.

	//vector<const DataSample *> trainDataSet;
	//vector<const DataSample *> validationDataSet;
	//vector<const DataSample *> testDataSet;

};

#endif /* DATASET_DATASET_H_ */
