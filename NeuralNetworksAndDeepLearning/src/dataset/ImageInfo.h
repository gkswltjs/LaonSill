/**
 * @file ImageInfo.h
 * @date 2016/4/21
 * @author jhkim
 * @brief 사용하지 않음.
 * @details
 */

#ifndef DATASET_IMAGEINFO_H_
#define DATASET_IMAGEINFO_H_

#include <fstream>
#include <iostream>
#include <string>

using namespace std;


class ImageInfo {
public:
	ImageInfo(string filename) {
		this->filename = filename;
		this->filesize = 0;
		this->buffer = 0;
	}
	virtual ~ImageInfo() {
		delete this->buffer;
	}

	string getFilename() const { return this->filename; }
	size_t getFilesize() const { return this->filesize; }


	int load() {
		ifstream fin(filename.c_str(), ios::in | ios::ate | ios::binary);

		fin.seekg(0, ios::end);   // 끝위치 이동
		filesize = fin.tellg();  // 파일사이즈 구하긔
		fin.seekg(0, ios::beg);   // 다시 시작으로 갖다놓긔

		buffer = new unsigned char[filesize];

		fin.read((char*)buffer, filesize);
		fin.close();

		return 0;
	}

	unsigned char *getBufferPtrAt(size_t position) const {
		if(position >= filesize) return 0;
		return &buffer[position];
	}


private:
	string filename;
	size_t filesize;
	unsigned char *buffer;
};

#endif /* DATASET_IMAGEINFO_H_ */
