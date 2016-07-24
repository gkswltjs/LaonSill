/*
 * ImageCropper.h
 *
 *  Created on: 2016. 7. 21.
 *      Author: jhkim
 */

#ifndef IMAGECROPPER_H_
#define IMAGECROPPER_H_

#include <thread>
#include <vector>
#include <string>
#include <mutex>


struct DirTaskArg_t {
	char dir_path[256];
	char dir_name[32];
	std::vector<std::string> file_name_list;
};



class ImageCropper {
public:
	ImageCropper(const char* image_dir, int scale, int numThreads=ImageCropper::system_num_cores);
	virtual ~ImageCropper();

	void crop();

private:
	void foreachDirectory(DirTaskArg_t *dirTaskArg);
	void foreachFile(const char* full_path, const char* dir_name, const char* file_name);

	char image_dir[256];
	char original_dir[256];
	char crop_dir[256];

	static const char *path_original;
	static const char *path_crop;
	static const int system_num_cores;
	static std::mutex barrier;

	int scale;
	int numThreads;
};

#endif /* IMAGECROPPER_H_ */
