/*
 * Tools.h
 *
 *  Created on: Jul 28, 2017
 *      Author: jkim
 */

#ifndef TOOLS_H_
#define TOOLS_H_


#include "Datum.h"



class ConvertMnistDataParam {
public:
	std::string imageFilePath;
	std::string labelFilePath;
	std::string outFilePath;
};

class ConvertImageSetParam {
public:
	ConvertImageSetParam() {
		this->gray = false;
		this->shuffle = false;
		this->multiLabel = false;
		this->channelSeparated = true;
		this->resizeWidth = 0;
		this->resizeHeight = 0;
		this->checkSize = false;
		this->encoded = true;
		this->encodeType = "jpg";
		this->basePath = "";
		this->datasetPath = "";
		this->outPath = "";
	}
	bool gray;
	bool shuffle;
	bool multiLabel;
	bool channelSeparated;
	int resizeWidth;
	int resizeHeight;
	bool checkSize;
	bool encoded;
	std::string encodeType;
	std::string basePath;
	std::string datasetPath;
	std::string outPath;
};

class ConvertAnnoSetParam : public ConvertImageSetParam {
public:
	ConvertAnnoSetParam() {
		this->annoType = "detection";
		this->labelType = "xml";
		this->labelMapFile = "";
		this->checkLabel = true;
		this->minDim = 0;
		this->maxDim = 0;
	}
	std::string annoType;
	std::string labelType;
	std::string labelMapFile;
	bool checkLabel;
	int minDim;
	int maxDim;
};




void denormalizeTest(int argc, char** argv);
void denormalize(const std::string& oldParamPath, const std::string& newParamPath);

void convertMnistDataTest(int argc, char** argv);
void convertMnistData(ConvertMnistDataParam& param);
//void convertMnistData(const std::string& imageFilePath, const std::string& labelFilePath,
//		const std::string& outFilePath);

void convertImageSetTest(int argc, char** argv);
void convertImageSet(ConvertImageSetParam& param);
/*
void convertImageSet(bool gray, bool shuffle, bool multiLabel, bool channelSeparated,
		int resizeWidth, int resizeHeight, bool checkSize, bool encoded,
		const std::string& encodeType, const std::string& imagePath,
		const std::string& datasetPath, const std::string& outPath);
		*/

void convertAnnoSetTest(int argc, char** argv);
void convertAnnoSet(ConvertAnnoSetParam& param);
/*
void convertAnnoSet(bool gray, bool shuffle, bool multiLabel, bool channelSeparated,
		int resizeWidth, int resizeHeight, bool checkSize, bool encoded,
		const std::string& encodeType, const std::string& annoType, const std::string& labelType,
		const std::string& labelMapFile, bool checkLabel, int minDim, int maxDim,
		const std::string& basePath, const std::string& datasetPath,
		const std::string& outPath);*/

void computeImageMeanTest(int argc, char** argv);
void computeImageMean(const std::string& sdfPath);



#endif /* TOOLS_H_ */
