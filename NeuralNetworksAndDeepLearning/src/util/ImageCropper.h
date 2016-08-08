/**
 * @file ImageCropper.h
 * @date 2016/7/21
 * @author jhkim
 * @brief
 * @details
 */

#ifndef IMAGECROPPER_H_
#define IMAGECROPPER_H_

#include <thread>
#include <vector>
#include <string>
#include <mutex>

/**
 * @brief Directory와 해당 Directory 내의 작업대상 이미지 이름 문자열 목록을 저장하는 구조체
 */
struct DirTaskArg_t {
	char dir_path[256];							///< 대상 이미지들을 포함하는 디렉토리 경로
	char dir_name[32];							///< 대상 이미지들을 포함하는 디렉토리 이름
	std::vector<std::string> file_name_list;	///< 디렉토리 내의 이미지 파일 이름 목록 벡터
};


/**
 * @brief 지정된 경로에 포함된 디렉토리의 이미지들을 리사이즈하여 지정된 위치에서 지정된 크기로 크롭한다.
 */
class ImageCropper {
public:
	/**
	 * @details ImageCropper 생성자
	 * @param image_dir 이미지 디렉토리들이 위치한 경로
	 * @param scale 이미지들을 리사이즈할 단축의 크기
	 * @param numThread 작업을 동시 수행할 thread의 수
	 */
	ImageCropper(const char* image_dir, int scale, int numThreads=ImageCropper::system_num_cores);
	/**
	 * @details ImageCropper 소멸자
	 */
	virtual ~ImageCropper();
	/**
	 * @details 이미지 크롭을 수행한다.
	 */
	void crop();

private:
	/**
	 * @details 대상 디렉토리별 작업을 수행한다. 디렉토리의 전체 이미지 파일에 대해 각 이미지당 작업을 수행하도록 요청한다.
	 * @param dirTaskArg 대상 디렉토리의 정보 구조체
	 */
	void foreachDirectory(DirTaskArg_t *dirTaskArg);
	/**
	 * @details 파일 하나에 대한 작업을 수행한다. 주어진 scale로 리사이즈하고 정해진 위치에서 크롭을 수행한다.
	 * @param full_path 이미지에 대한 전체 경로 문자열 포인터
	 * @param dir_name 이미지가 들어있는 디렉토리 경로 문자열 포인터
	 * @param file_name 이미지의 이름 문자열 포인터
	 */
	void foreachFile(const char* full_path, const char* dir_name, const char* file_name);

	char image_dir[256];					///< 작업 대상 Root 경로 문자 배열
	char original_dir[256];					///< 원본 이미지 경로 문자 배열
	char crop_dir[256];						///< 크롭 이미지 경로 문자 배열

	static const char *path_original;		///< "oiginal"
	static const char *path_crop;			///< "crop"
	static const int system_num_cores;		///< 시스템 CPU 코어 수
	static std::mutex barrier;				///< 파일 쓰기 뮤텍스

	int scale;								///< 이미지 리사이즈 및 크롭 사이즈
	int numThreads;							///< 작업을 동시 수행할 thread의 수
};

#endif /* IMAGECROPPER_H_ */
