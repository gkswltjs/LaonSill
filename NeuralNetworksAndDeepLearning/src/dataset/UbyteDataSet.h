/**
 * @file	UbyteDataSet.h
 * @date	2016/7/13
 * @author	jhkim
 * @brief
 * @details
 */





#ifndef UBYTEDATASET_H_
#define UBYTEDATASET_H_

#include <armadillo>
#include <vector>
#include <string>

#include "DataSet.h"
#include "DataSample.h"
#include "ImageInfo.h"

using namespace std;
using namespace arma;


/**
 * @brief 수정된 Mnist 파일 형태의 데이터셋 파일을 읽기위해 구현된 DataSet
 * @details Mnist(http://yann.lecun.com/exdb/mnist/)의 파일 포맷에서
 *          - # of rows, # of columns 후, # of channels 추가.
 *          - 256개 이상의 레이블 사이즈를 수용하기 위해 label 데이터를 unsigned char -> unsigned int로 변경
 *          (mnist의 경우 10진법 숫자를 구별하기 위해 10개의 카테고리가 있었고 이는 2^8=256, 8bit으로 수용가능해 unsigned char를 사용)
 * @todo mnist 파일의 수정없이 파라미터를 통해 mnist 원본을 그대로 읽을 수 있도록 수정할 수도 있음.
 */
class UbyteDataSet : public DataSet {
public:
	/**
	 * @details UbyteDataSet 생성자.
	 * @param train_image 학습데이터셋 파일 경로.
	 *        파일명은 train_data[숫자] 형식으로 숫자는 0에서 numTrainFile-1의 범위.
	 * @param train_label 학습데이터셋 정답레이블 파일 경로.
	 *        파일명은 train_label[숫자] 형식으로 숫자는 0에서 numTrainFile-1의 범위.
	 * @param numTrainFile 학습데이터셋 파일의 수.
	 * @param test_image 테스트데이터셋 파일 경로.
	 *        파일명은 test_data[숫자] 형식으로 숫자는 0에서 numTestFile-1위 범위.
	 * @param test_label 테스트데이터셋 정답레이블 파일 경로.
	 *        파일명은 test_label[숫자] 형식으로 숫자는 0에서 numTestFile-1위 범위.
	 * @param numTestFile 테스트데이터셋 파일의 수.
	 * @param channel 데이터셋 데이터의 채널 수 (deprecated.)
	 * @param validationSetRatio 학습데이터셋에서 유효데이터 비율 (not used yet.)
	 */
	UbyteDataSet(
			string train_image,
			string train_label,
			int numTrainFile,
			string test_image,
			string test_label,
			int numTestFile,
			int channel=0,
			double validationSetRatio=0.0
			);
	virtual ~UbyteDataSet();

	virtual void load();

	virtual const DATATYPE *getTrainDataAt(int index);
	virtual const UINT *getTrainLabelAt(int index);
	virtual const DATATYPE *getValidationDataAt(int index);
	virtual const UINT *getValidationLabelAt(int index);
	virtual const DATATYPE *getTestDataAt(int index);
	virtual const UINT *getTestLabelAt(int index);


	void shuffleTrainDataSet();
	void shuffleValidationDataSet();
	void shuffleTestDataSet();


#if CPU_MODE
protected:
	int loadDataSetFromResource(string resources[2], DataSample *&dataSet, int offset, int size);
#else
protected:
	int load(int type, int page=0);
	int loadDataSetFromResource(
			string data_path,
			string label_path,
			vector<DATATYPE> *&dataSet,
			vector<UINT> *&labelSet,
			int offset,
			int size);
#endif

protected:
	string train_image;						///< 학습데이터셋 파일 경로
	string train_label;						///< 학습데이터셋 정답레이블 파일 경로
	int numTrainFile;						///< 학습데이터셋 파일 수
	string test_image;						///< 테스트데이터셋 파일 경로
	string test_label;						///< 테스트데이터셋 정답레이블 파일 경로
	int numTestFile;						///< 테스트데이터셋 파일 수
	double validationSetRatio;				///< 학습데이터셋의 유효데이터 비율

	int trainFileIndex;						///< 현재 학습데이터 파일 인덱스
	int testFileIndex;						///< 현재 테스트 파일 인덱스
	int numImagesInTrainFile;				///< 학습데이터셋 파일 하나에 들어있는 데이터의 수
	int numImagesInTestFile;				///< 테스트데이터셋 파일 하나에 들어있는 데이터의 수

	vector<uint8_t> *bufDataSet;			///< 데이터셋 데이터를 로드할 버퍼. 파일의 uint8_t타입 데이터를 버퍼에 올려 uint32_t타입으로 변환하기 위한 버퍼.
};

#endif /* UBYTEDATASET_H_ */








































