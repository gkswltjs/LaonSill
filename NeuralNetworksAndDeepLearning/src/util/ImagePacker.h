/*
 * ImagePacker.h
 *
 *  Created on: 2016. 7. 13.
 *      Author: jhkim
 */

#ifndef IMAGEPACKER_H_
#define IMAGEPACKER_H_


#include <string>
#include <vector>


using namespace std;


struct category_t {
	int id;
	string name;
	vector<string> fileList;
	int fileIndex;

	category_t() {
		fileIndex = 0;
	}

	string getCurrentFile() {
		if(end()) {	exit(1); }
		return fileList[fileIndex++];
	}
	bool end() {
		return (fileList.size() <= fileIndex);
	}
};



class ImagePacker {
public:
	ImagePacker(string image_dir,
			int numCategory,
			int numTrain,
			int numTest,
			int numImagesInTrainFile,
			int numImagesInTestFile,
			int numChannels);
	virtual ~ImagePacker();

	void load();
	void show();
	void pack();

private:
	void loadFilesInCategory(string categoryPath, vector<string>& fileList);
	void _pack(string dataPath, string labelPath, int numImagesInFile, int size);


	string image_dir;

	static const string path_crop;
	static const string path_save;

	vector<category_t> categoryList;
	int numCategory;											// pack 대상 category 수
	int numTrain;												// train 이미지 수
	int numTest;												// test 이미지 수
	int numImagesInTrainFile;									// train pack file 하나당 이미지 수
	int numImagesInTestFile;									// test pack file 하나당 이미지 수
	int numChannels;

	uint32_t categoryIndex;

};

#endif /* IMAGEPACKER_H_ */









