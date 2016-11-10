/*
 * ImageCropper.cpp
 *
 *  Created on: 2016. 7. 21.
 *      Author: jhkim
 */

#include "ImageCropper.h"
#include "Util.h"

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <CImg.h>
#include <dirent.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace std;
using namespace cimg_library;

namespace fs = boost::filesystem;

const char *ImageCropper::path_original = "original";
const char *ImageCropper::path_crop = "crop";
const int ImageCropper::system_num_cores = thread::hardware_concurrency();
mutex ImageCropper::barrier;





ImageCropper::ImageCropper(const char *image_dir, int scale, int numThreads)
	:	scale(scale), numThreads(numThreads) {
	sprintf(this->image_dir, "%s", image_dir);
	sprintf(this->original_dir, "%s/%s", image_dir, path_original);
	sprintf(this->crop_dir, "%s/%s", image_dir, path_crop);

	this->scale = scale;
	this->numThreads = numThreads;

	cout << "image_dir: " << image_dir << endl;
	cout << "original_dir: " << original_dir << endl;
	cout << "crop_dir: " << crop_dir << endl;
	cout << "num threads: " << numThreads << endl;
}

ImageCropper::~ImageCropper() {}


void temp() {
	cout << "temp ... " << endl;
}

void ImageCropper::crop() {
	int dirCnt = 0;


	DIR *dir;
	if((dir = opendir(original_dir)) != NULL) {

		Timer timer;
		timer.start();

		thread *threadList = new thread[numThreads];
		DirTaskArg_t dirTaskArgList[numThreads];
		int taskCount = 0;

		struct dirent ent;
		struct dirent *result;
		while(readdir_r(dir, &ent, &result) == 0) {
			if(result == NULL) break;
			if(ent.d_type == 4 && ent.d_name[0] != '.') {
				strcpy(dirTaskArgList[taskCount].dir_name, ent.d_name);
				sprintf(dirTaskArgList[taskCount].dir_path, "%s/%s", original_dir, ent.d_name);

				DIR *fdir;
				if((fdir = opendir(dirTaskArgList[taskCount].dir_path)) != NULL) {
					struct dirent fent;
					struct dirent *fresult;
					while(readdir_r(fdir, &fent, &fresult) == 0) {
						if(fresult == NULL) break;
						if(fent.d_type == 8) {
							dirTaskArgList[taskCount].file_name_list.push_back(fent.d_name);
						}
					}
					closedir(fdir);
				} else {
					perror("could not load ... ");
					exit(-1);
				}

				if(++taskCount >= numThreads) {
					for(int i = 0; i < numThreads; i++) {
						threadList[i] = thread(&ImageCropper::foreachDirectory, this, &dirTaskArgList[i]);
						cout << ++dirCnt << ": " << dirTaskArgList[i].dir_path << endl;
					}

					for(int i = 0; i < numThreads; i++) {
						threadList[i].join();
					}
					delete [] threadList;
					threadList = new thread[numThreads];
					taskCount = 0;
					for(int i = 0; i < numThreads; i++) {
						dirTaskArgList[i].file_name_list.clear();
					}
					cout << "time: " << timer.stop(false) << endl;
				}
			}
		}
		closedir(dir);
	} else {
		perror("could not load ... ");
		exit(-1);
	}
}


void ImageCropper::foreachDirectory(DirTaskArg_t *dirTaskArg) {

	char full_path[256];
	sprintf(full_path, "%s/%s", crop_dir, dirTaskArg->dir_name);
	mkdir(full_path, 0755);

	int file_size = dirTaskArg->file_name_list.size();

	for(int i = 0; i < file_size; i++) {
		sprintf(full_path, "%s/%s", dirTaskArg->dir_path, dirTaskArg->file_name_list[i].c_str());
		foreachFile(full_path, dirTaskArg->dir_name, dirTaskArg->file_name_list[i].c_str());
	}



	/*
	DIR *dir;
	if((dir = opendir(dirTaskArg->dir_path)) != NULL) {

		char crop_full_dir[256];
		sprintf(crop_full_dir, "%s/%s", crop_dir, dirTaskArg->dir_name);
		mkdir(crop_full_dir, 0755);

		char file_full_path[256];
		struct dirent ent;
		struct dirent *result;

		while(readdir_r(dir, &ent, &result) == 0) {
			if(ent.d_type == 8) {
				sprintf(file_full_path, "%s/%s", dirTaskArg->dir_path, ent.d_name);
				foreachFile(file_full_path, dirTaskArg->dir_name, ent.d_name);
			}
		}
		closedir(dir);
	} else {
		perror("could not load ... ");
		exit(-1);
	}
	*/
}


void ImageCropper::foreachFile(const char* full_path, const char* dir_name, const char* file_name) {
	CImg<unsigned char> image(full_path);

	const int width = image.width();
	const int height = image.height();
	int rwidth = 0;
	int rheight = 0;
	if(width >= height) {
		rwidth = (int)(width/(float)height*scale);
		rheight = scale;
	} else {
		rwidth = scale;
		rheight = (int)(height/(float)width*scale);
	}
	image.resize(rwidth, rheight, -100, -100, 5);

	char crop_file_name[256];
	for(int i = 0; i < 3; i++) {
		int x0, y0, x1, y1;
		switch(i) {
		case 0:
			x0 = 0;
			y0 = 0;
			x1 = scale-1;
			y1 = scale-1;
			break;
		case 1:
			x0 = (rwidth-scale)/2;
			y0 = (rheight-scale)/2;
			x1 = (rwidth+scale)/2-1;
			y1 = (rheight+scale)/2-1;
			break;
		case 2:
			x0 = rwidth-scale;
			y0 = rheight-scale;
			x1 = rwidth-1;
			y1 = rheight-1;
			break;
		default:
			break;
		}
		//CImg<unsigned char> crop_image = image;



		sprintf(crop_file_name, "%s/%s/", crop_dir, dir_name);

		int cfnj = strlen(crop_file_name), fnj = 0;
		do {
			crop_file_name[cfnj++] = file_name[fnj];
		}
		while(file_name[fnj++] != '.');
		crop_file_name[cfnj++]='0'+i;
		crop_file_name[cfnj++]='.';
		do {
			crop_file_name[cfnj++] = file_name[fnj];
		}
		while(file_name[fnj++] != 0);

		//cout << "crop file: " << crop_file_name << endl;
		//lock_guard<mutex> block_threads_until_finish_this_job(barrier);
		(image.get_crop(x0, y0, x1, y1, 1)).save_jpeg(crop_file_name, 100);
	}
}


















