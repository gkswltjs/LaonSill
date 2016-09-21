/**
 * @file ImagePacker.h
 * @date 2016/7/13
 * @author jhkim
 * @brief
 * @details
 */


#ifndef IMAGEPACKER_H_
#define IMAGEPACKER_H_


#include <string>
#include <vector>


using namespace std;

/**
 * @brief 한 카테고리에 해당하는 파일목록 자료구조
 * @details 한 디렉토리에 들어있는 파일들을 동일 카테고리로 간주한다.
 */
struct category_t {
	int id;						///< 카테고리 아이디, 데이터셋상에서 레이블에 해당하는 값
	string name;				///< 카테고리 이름, 해당 디렉토리의 이름에 해당하는 값
	vector<string> fileList;	///< 파일 이름 문자열 목록 벡터, 해당 디렉토리 내의 모든 이미지 이름 목록에 해당하는 값
	int fileIndex;				///< 현재 해당 카테고리내에서 Pack된 파일의 수, 중복으로 pack하지 않기 위한 index
	int sizePerCategory;

	category_t() {
		fileIndex = 0;
		sizePerCategory = 0;
	}

	/**
	 * @details 카테고리 내에 아직 pack되지 않은 다음 파일의 이름을 조회한다.
	 * @return 카테고리 내 pack되지 않은 다음 파일의 이름
	 */
	string getCurrentFile() {
		if(end()) {	exit(1); }
		return fileList[fileIndex++];
	}

	int getFileIndex() {
		return fileIndex;
	}

	void addSizePerCategory(int sizePerCategory) {
		this->sizePerCategory += sizePerCategory;
	}

	/**
	 * @details 카테고리 내에 아직 pack되지 않은 파일이 있는지 여부를 확인한다.
	 * @return 카테고리 내 pack되지 않은 파일의 존재 여부
	 */
	bool end() {
		return (fileList.size() <= fileIndex || sizePerCategory <= fileIndex);
	}
};


/**
 * @brief 이미지 파일을 대상으로 적절한 사이즈의 디코드되어 그룹화된 Image Pack을 생성한다.
 * @details 많은 이미지를 대상으로 매번 디코딩을 하고 많은 파일에 대해 읽기를 수행해야 하는 낭비를 막는다.
 */
class ImagePacker {
public:
	/**
	 * @details ImagePacker 생성자
	 * @param image_dir 작업 대상이 되는 경로 문자열
	 * @param numCategory pack할 카테고리의 수
	 * @param numTrain 학습 데이터의 수
	 * @param numTest 테스트 데이터의 수
	 * @param numImagesInTrainFile 학습 파일 하나에 포함될 이미지의 수
	 * @param numImagesInTestFile 테스트 파일 하나에 포함될 이미지의 수
	 * @param numChannels 대상 이미지의 채널 수
	 */
	ImagePacker(string image_dir,
			int numCategory,
			int numTrain,
			int numTest,
			int numImagesInTrainFile,
			int numImagesInTestFile,
			int numChannels);
	virtual ~ImagePacker();
	/**
	 * @details 대상이 되는 파일의 이름들을 카테고리별로 메모리에 읽어 들인다.
	 */
	void load();
	/**
	 * @details load()를 통해 메모리로 읽어들인 정보를 화면에 출력한다.
	 */
	void show();
	/**
	 * @details load()된 파일들을 설정정보에 따라 파일로 pack한다.
	 */
	void pack();


	void sample();

private:
	/**
	 * @details 지정된 디렉토리내의 모든 파일을 메모리로 읽어 들인다.
	 * @param categoryPath 카테고리에 해당하는 디렉토리의 경로
	 * @param fileList 카테고리 디렉토리 내의 모든 파일 이름을 읽어들일 문자열 목록 벡터
	 */
	void loadFilesInCategory(string categoryPath, vector<string>& fileList);


	void writeCategoryLabelFile(string categoryLabelPath);
	/**
	 * @details 실제 pack을 수행한다.
	 * @param dataPath 데이터 파일 경로 문자열
	 * @param labelPath 데이터 정답 파일 경로 문자열
	 * @param numImagesInFile 파일 하나에 포함될 이미지의 수
	 * @param size 전체 데이터의 수
	 * @param sizePerCategory 하나의 카테고리당 pack할 이미지의 수
	 */
	void _pack(string dataPath, string labelPath, int numImagesInFile, int size, int sizePerCategory);


	string image_dir;						///< Pack 대상의 Root 경로 문자열

	static const string path_crop;			///< "crop"
	static const string path_save;			///< "save"

	vector<category_t> categoryList;		///< Pack 대상의 카테고리와 카테고리의 이미지 파일 정보 목록 벡터
	int numCategory;						///< pack 대상의 카테고리 수
	int numTrain;							///< 학습 이미지 수
	int numTest;							///< 테스트 이미지 수
	int numImagesInTrainFile;				///< 학습 파일 하나에 포함될 이미지의 수
	int numImagesInTestFile;				///< 테스트 파일 하나에 포함될 이미지의 수
	int numChannels;						///< 이미지 채널의 수 (이미지 마다 채널의 수가 다를 수 있고 다른 경우 주어진 채널수로 고정하게 한다.)

	uint32_t categoryIndex;					///< 다음 대상 카테고리 index, 카테고리별로 균등하게 pack하기 위해 카테고리별로 돌아가며 파일 하나씩 pack할 때 사용

};

#endif /* IMAGEPACKER_H_ */









