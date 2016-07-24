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

	bool end() {
		return (fileList.size() >= fileIndex);
	}
};



class ImagePacker {
public:
	ImagePacker(string image_dir,
			int numCategory,
			int numTrain,
			int numTest,
			int numImagesInFile);
	virtual ~ImagePacker();

	void load();
	void show();
	void pack();

private:
	void loadFilesInCategory(string categoryPath, vector<string>& fileList);


	string image_dir;

	static const string path_crop;
	static const string path_save;

	vector<category_t> categoryList;
	int numCategory;
	int numTrain;
	int numTest;
	int numImagesInFile;

};

#endif /* IMAGEPACKER_H_ */









