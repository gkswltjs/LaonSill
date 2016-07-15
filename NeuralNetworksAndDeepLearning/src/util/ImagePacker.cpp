/*
 * ImagePacker.cpp
 *
 *  Created on: 2016. 7. 13.
 *      Author: jhkim
 */

#include "ImagePacker.h"

#include <dirent.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <CImg.h>

#include "../Util.h"
#include "UByteImage.h"

using namespace cimg_library;




ImagePacker::ImagePacker(const char *image_dir, const char *outfile_image_path, const char *outfile_label_path) {
	strcpy(this->image_dir, image_dir);
	strcpy(this->outfile_image_path, outfile_image_path);
	strcpy(this->outfile_label_path, outfile_label_path);
}

ImagePacker::~ImagePacker() {}


void ImagePacker::pack() {
	ofstream ofsImage(outfile_image_path, ios::out | ios::binary);
	ofstream ofsLabel(outfile_label_path, ios::out | ios::binary);

	UByteImageDataset imageDataSet;
	UByteLabelDataset labelDataSet;

	imageDataSet.magic = UBYTE_IMAGE_MAGIC;
	labelDataSet.magic = UBYTE_LABEL_MAGIC;

	// find number of images in dir
	UINT imageCount = 0;
	DIR *dir;
	if((dir = opendir(image_dir)) != NULL) {
		struct dirent *ent;
		while((ent = readdir(dir)) != NULL) {
			if(strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
			imageCount++;
		}
		closedir(dir);
	} else { perror("could not load ... "); }
	cout << "imageCount: " << imageCount << endl;

	imageDataSet.length = imageCount;
	labelDataSet.length = imageCount;
	labelDataSet.Swap();
	ofsLabel.write((char *)&labelDataSet, sizeof(UByteLabelDataset));

	if((dir = opendir(image_dir)) != NULL) {
		int set_width = 0;
		int set_height = 0;
		char image_path[256];
		bool first = true;

		struct dirent *ent;
		while((ent = readdir(dir)) != NULL) {
			//if(strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
			if(ent->d_type == 4) continue;

			image_path[0] = 0;
			strcat(image_path, image_dir);
			strcat(image_path, ent->d_name);

			cout << "load image " << image_path << endl;
			CImg<unsigned char> image(image_path);
			int width = image.width();
			int height = image.height();

			if(first) {
				set_width = width;
				set_height = height;
				imageDataSet.width = width;
				imageDataSet.height = height;
				imageDataSet.Swap();
				ofsImage.write((char *)&imageDataSet, sizeof(UByteImageDataset));
				first = false;
			}

			if(set_width != width || set_height != height) {
				exit(1);
			}

			unsigned char *ptr = image.data(0);
			ofsImage.write((char *)ptr, sizeof(unsigned char)*width*height*3);

			unsigned char label = ent->d_name[0]-'1';
			ofsLabel.write((char *)&label, sizeof(unsigned char));
		}
		closedir(dir);
	} else { perror("could not load ... "); }

	ofsImage.close();
	ofsLabel.close();

}

