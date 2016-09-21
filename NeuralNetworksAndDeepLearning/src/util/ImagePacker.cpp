/*
 * ImagePacker.cpp
 *
 *  Created on: 2016. 7. 13.
 *      Author: jhkim
 */

#include "ImagePacker.h"

#include <CImg.h>
#include <dirent.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "UByteImage.h"

using namespace std;
using namespace cimg_library;


const string ImagePacker::path_save = "/save";
//const string ImagePacker::path_crop = "/crop_sample_10cat_100img";
const string ImagePacker::path_crop = "/crop";




ImagePacker::ImagePacker(string image_dir,
		int numCategory,
		int numTrain,
		int numTest,
		int numImagesInTrainFile,
		int numImagesInTestFile,
		int numChannels) {
	this->image_dir = image_dir;

	this->numCategory = numCategory;
	this->numTrain = numTrain;
	this->numTest = numTest;
	this->numImagesInTrainFile = numImagesInTrainFile;
	this->numImagesInTestFile = numImagesInTestFile;
	this->numChannels = numChannels;

	this->categoryIndex = 0;
}

ImagePacker::~ImagePacker() {}


void ImagePacker::load() {
	int categoryCount = 0;
	DIR *dir;
	if((dir = opendir((image_dir+path_crop).c_str())) != NULL) {
		struct dirent ent;
		struct dirent *result;
		while(readdir_r(dir, &ent, &result) == 0) {
			if(result == NULL) break;
			if(ent.d_type == 4 && ent.d_name[0] != '.') {
				category_t category;
				category.id = categoryCount;
				category.name = ent.d_name;
				loadFilesInCategory(image_dir+path_crop+"/"+category.name, category.fileList);
				categoryList.push_back(category);
				if(++categoryCount >= numCategory) break;
			}
		}
		closedir(dir);
		numCategory = categoryCount;
	} else {
		perror("could not load ... ");
		exit(-1);
	}
}

void ImagePacker::loadFilesInCategory(string categoryPath, vector<string>& fileList) {
	const char* suffix = ".1.JPEG";
	const size_t suffixLength = strlen(suffix);

	int fileCount = 0;
	DIR *dir;
	if((dir = opendir(categoryPath.c_str())) != NULL) {
		struct dirent ent;
		struct dirent *result;
		while(readdir_r(dir, &ent, &result) == 0) {
			if(result == NULL) break;
			if(ent.d_type == 8) {

				size_t nameLength = strlen(ent.d_name);
				if(strncmp(ent.d_name + nameLength - suffixLength, suffix, suffixLength) == 0) {
					fileList.push_back(ent.d_name);
				}

			}
		}
		random_shuffle(&fileList[0], &fileList[fileList.size()]);
		closedir(dir);
	} else {
		perror("could not load ... ");
		exit(-1);
	}
}

void ImagePacker::show() {
	for(int i = 0; i < categoryList.size(); i++) {
		category_t& category = categoryList[i];
		cout << i << "th category: " << category.name << ", files: " << category.fileList.size() << endl;
		for(int j = 0; j < category.fileList.size(); j++) {
			cout << "\t" << j << "th file: " << category.fileList[j] << endl;
		}
	}
}



void ImagePacker::pack() {
	const string saveBase = image_dir+path_save;

	writeCategoryLabelFile(saveBase+"/category_label");

	// for train
	_pack(saveBase+"/train_data", saveBase+"/train_label", numImagesInTrainFile, numTrain, numTrain/numCategory);
	// for test
	_pack(saveBase+"/test_data", saveBase+"/test_label", numImagesInTestFile, numTest, numTest/numCategory);
}


void ImagePacker::writeCategoryLabelFile(string categoryLabelPath) {
	ofstream ofsCategoryLabel(categoryLabelPath.c_str(), ios::out);
	for(int i = 0; i < categoryList.size(); i++) {
		ofsCategoryLabel << categoryList[i].name << "\t" << categoryList[i].id << endl;
	}
	ofsCategoryLabel.close();
}


void ImagePacker::_pack(string dataPath, string labelPath, int numImagesInFile, int size, int sizePerCategory) {

	srand((unsigned int)time(NULL));

	for(uint32_t i = 0; i < numCategory; i++) {
		categoryList[i].addSizePerCategory(sizePerCategory);
	}


	UByteImageDataset imageDataSet;
	imageDataSet.magic = UBYTE_IMAGE_MAGIC;
	imageDataSet.length = numImagesInFile;

	UByteLabelDataset labelDataSet;
	labelDataSet.magic = UBYTE_LABEL_MAGIC;
	labelDataSet.length = numImagesInFile;
	labelDataSet.Swap();

	int imagesInFileCount = 0;
	ofstream *ofsData = 0;
	ofstream *ofsLabel = 0;
	int width = 0;
	int height = 0;

	for(int i = 0; i < size; i++) {
		if(i%numImagesInFile == 0) {
			if(ofsData) {
				ofsData->close();
				ofsData = 0;
				ofsLabel->close();
				ofsLabel = 0;
				cout << "images in file: " << imagesInFileCount << endl;
				imagesInFileCount = 0;
			}
			string dataFile = dataPath+to_string(i/numImagesInFile);
			cout << "dataFile: " << dataFile << endl;
			ofsData = new ofstream(dataFile.c_str(), ios::out | ios::binary);
			if(i > 0) ofsData->write((char *)&imageDataSet, sizeof(UByteImageDataset));

			string labelFile = labelPath+to_string(i/numImagesInFile);
			ofsLabel = new ofstream(labelFile.c_str(), ios::out | ios::binary);
			ofsLabel->write((char *)&labelDataSet, sizeof(UByteLabelDataset));
		}

		do {
			categoryIndex = rand()%numCategory;
		} while(categoryList[categoryIndex].end());

			/*
		while(categoryList[categoryIndex].end()) {
			if(++categoryIndex >= numCategory) {
				categoryIndex = 0;
			}
		}
		*/

		string imageFile = image_dir+path_crop+"/"+categoryList[categoryIndex].name+"/"+categoryList[categoryIndex].getCurrentFile();
		//cout << i << ": imageFile: " << imageFile << endl;
		CImg<unsigned char> image(imageFile.c_str());
		// 첫 이미지일때 DataSet header에 width, height 지정,
		if(i == 0) {
			width = image.width();
			height = image.height();
			imageDataSet.width = width;
			imageDataSet.height = height;
			imageDataSet.channel = numChannels;
			imageDataSet.Swap();
			ofsData->write((char *)&imageDataSet, sizeof(UByteImageDataset));
		}

		if(width != image.width() || height != image.height()) {
			exit(1);
		}

		// dataset에 주어진 channel수와 image의 channel수가 동일한 경우 channel수만큼 그대로 복사.
		if(image.spectrum() == numChannels) { ofsData->write((char *)image.data(), sizeof(unsigned char)*width*height*numChannels); }
		// color dataset에 grayscale 이미지가 주어진 경우를 상정했지만,
		// 일반적으로 color dataset의 channel수가 image channel수의 배수가 된다면 배수만큼 복사. (이런 경우는 없다고 보임)
		else if(image.spectrum() < numChannels && numChannels%image.spectrum()==0) {
			int numRepeat = numChannels / image.spectrum();
			for(int i = 0; i < numRepeat; i++) ofsData->write((char *)image.data(), sizeof(unsigned char)*width*height*image.spectrum());
		}
		else {
			cout << "image invalid channel num ... " << image.spectrum() << endl;
			exit(1);
		}

		imagesInFileCount++;
		ofsLabel->write((char *)&categoryIndex, sizeof(uint32_t));
		//cout << "label: " << categoryIndex << endl;

		if(++categoryIndex >= numCategory) {
			categoryIndex = 0;
		}

		if((i+1)%1000 == 0) {
			cout << "processed " << (i+1) << "images ... " << endl;
		}

	}

	if(ofsData) {
		ofsData->close();
		ofsData = 0;
	}
	if(ofsLabel) {
		ofsLabel->close();
		ofsLabel = 0;
	}

	cout << "Category Pack Stat: " << endl;
	for(uint32_t i = 0; i < numCategory; i++) {
		cout << "category " << i << ": " << categoryList[i].getFileIndex() << endl;
	}

}


// 카테고리는 일단 100개만 있는 걸로 합시다.
// char는 256, 사인 골치아픈거 생각하면 100까지만.
void ImagePacker::sample() {
	string dataPath = "/home/jhkim/image/ILSVRC2012/save/sample/train_data";
	string labelPath = "/home/jhkim/image/ILSVRC2012/save/sample/train_label";

	const int numCategory = 30;
	const int numImagesInFile = 10;
	const int size = 30;
	categoryIndex = 0;

	const uint32_t width = 2;
	const uint32_t height = 2;
	const uint32_t channel = 1;

	const uint32_t imageSize = width*height*channel;
	char buffer[imageSize];


	UByteImageDataset imageDataSet;
	imageDataSet.magic = UBYTE_IMAGE_MAGIC;
	imageDataSet.length = numImagesInFile;

	UByteLabelDataset labelDataSet;
	labelDataSet.magic = UBYTE_LABEL_MAGIC;
	labelDataSet.length = numImagesInFile;
	labelDataSet.Swap();

	int imagesInFileCount = 0;
	ofstream *ofsData = 0;
	ofstream *ofsLabel = 0;

	for(int i = 0; i < size; i++) {
		if(i%numImagesInFile == 0) {
			if(ofsData) {
				ofsData->close();
				ofsData = 0;
				ofsLabel->close();
				ofsLabel = 0;
				cout << "images in file: " << imagesInFileCount << endl;
				imagesInFileCount = 0;
			}
			string dataFile = dataPath+to_string(i/numImagesInFile);
			cout << "dataFile: " << dataFile << endl;
			ofsData = new ofstream(dataFile.c_str(), ios::out | ios::binary);
			if(i > 0) ofsData->write((char *)&imageDataSet, sizeof(UByteImageDataset));

			string labelFile = labelPath+to_string(i/numImagesInFile);
			ofsLabel = new ofstream(labelFile.c_str(), ios::out | ios::binary);
			ofsLabel->write((char *)&labelDataSet, sizeof(UByteLabelDataset));
		}

		/*
		do {
			categoryIndex = rand()%numCategory;
		} while(categoryList[categoryIndex].end());
		*/

		if(++categoryIndex >= numCategory) {
			categoryIndex = 0;
		}


		for(uint32_t i = 0; i < imageSize; i++) {
			buffer[i] = categoryIndex;
		}


		//string imageFile = image_dir+path_crop+"/"+categoryList[categoryIndex].name+"/"+categoryList[categoryIndex].getCurrentFile();
		//CImg<unsigned char> image(imageFile.c_str());
		// 첫 이미지일때 DataSet header에 width, height 지정,
		if(i == 0) {
			imageDataSet.width = width;
			imageDataSet.height = height;
			imageDataSet.channel = channel;
			imageDataSet.Swap();
			ofsData->write((char *)&imageDataSet, sizeof(UByteImageDataset));
		}

		// dataset에 주어진 channel수와 image의 channel수가 동일한 경우 channel수만큼 그대로 복사.
		ofsData->write((char *)buffer, sizeof(unsigned char)*imageSize);

		imagesInFileCount++;
		ofsLabel->write((char *)&categoryIndex, sizeof(uint32_t));
		//cout << "label: " << categoryIndex << endl;


		if((i+1)%1000 == 0) {
			cout << "processed " << (i+1) << "images ... " << endl;
		}
	}

	if(ofsData) {
		ofsData->close();
		ofsData = 0;
	}
	if(ofsLabel) {
		ofsLabel->close();
		ofsLabel = 0;
	}

	cout << "Category Pack Stat: " << endl;
	for(uint32_t i = 0; i < numCategory; i++) {
		cout << "category " << i << ": " << categoryList[i].getFileIndex() << endl;
	}

}









