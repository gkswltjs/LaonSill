/*
 * DataSet.cpp
 *
 *  Created on: 2016. 4. 21.
 *      Author: jhkim
 */

#include "MnistDataSet.h"

#include <iostream>


#include "../Util.h"
#include "DataSample.h"
#include "ImageInfo.h"


#define UBYTE_IMAGE_MAGIC 2051
#define UBYTE_LABEL_MAGIC 2049

#ifdef _MSC_VER
	#define bswap(x) _byteswap_ulong(x)
#else
	#define bswap(x) __builtin_bswap32(x)
#endif


struct UByteImageDataset {
	uint32_t magic;			/// Magic number (UBYTE_IMAGE_MAGIC).
	uint32_t length;		/// Number of images in dataset.
	uint32_t height;		/// The height of each image.
	uint32_t width;			/// The width of each image.
	void Swap() {
		magic = bswap(magic);
		length = bswap(length);
		height = bswap(height);
		width = bswap(width);
	}
};

struct UByteLabelDataset {
	uint32_t magic;			/// Magic number (UBYTE_LABEL_MAGIC).
	uint32_t length;		/// Number of labels in dataset.
	void Swap() {
		magic = bswap(magic);
		length = bswap(length);
	}
};





MnistDataSet::MnistDataSet(double validationSetRatio = 0.0) {
	this->validationSetRatio = validationSetRatio;
}

MnistDataSet::~MnistDataSet() {
	if(trainDataSet) { delete trainDataSet; }
	if(trainLabelSet) { delete trainLabelSet; }
	if(validationDataSet) { delete validationDataSet; }
	if(validationLabelSet) { delete validationLabelSet; }
	if(testDataSet) { delete testDataSet; }
	if(testLabelSet) { delete testLabelSet; }
}

void MnistDataSet::load() {

	string filenames[2][2] = {
			//{"./data/mnist/train-images.idx3-ubyte", "./data/mnist/train-labels.idx1-ubyte"},
			//{"./data/mnist/t10k-images.idx3-ubyte", "./data/mnist/t10k-labels.idx1-ubyte"}
			{"/home/jhkim/dev/git/neuralnetworksanddeeplearning/NeuralNetworksAndDeepLearning/data/mnist/train-images.idx3-ubyte",
			 "/home/jhkim/dev/git/neuralnetworksanddeeplearning/NeuralNetworksAndDeepLearning/data/mnist/train-labels.idx1-ubyte"},
			{"/home/jhkim/dev/git/neuralnetworksanddeeplearning/NeuralNetworksAndDeepLearning/data/mnist/t10k-images.idx3-ubyte",
			 "/home/jhkim/dev/git/neuralnetworksanddeeplearning/NeuralNetworksAndDeepLearning/data/mnist/t10k-labels.idx1-ubyte"}
	};


#if CPU_MODE
	numTrainData = loadDataSetFromResource(filenames[0], trainDataSet, 0, 10000);
	numTestData = loadDataSetFromResource(filenames[1], testDataSet, 0, 0);
#else
	numTrainData = loadDataSetFromResource(filenames[0], trainDataSet, trainLabelSet, 0, 50000);
	numTestData = loadDataSetFromResource(filenames[1], testDataSet, testLabelSet, 0, 10000);
	//numTrainData = 10;
	//numTestData = 10;
#endif

	//trainDataSize = loadDataSetFromResource(filenames[0], trainDataSet);
	//testDataSize = loadDataSetFromResource(filenames[1], testDataSet);

}



/*
int MnistDataSet::loadDataSetFromResource(string resources[2], vector<const DataSample *> &dataSet) {

	// LOAD IMAGE DATA
	ImageInfo dataInfo(resources[0]);
	dataInfo.load();

	// READ IMAGE DATA META DATA
	unsigned char *dataPtr = dataInfo.getBufferPtrAt(0);
	int dataMagicNumber = Util::pack4BytesToInt(dataPtr);
	if(dataMagicNumber != 0x00000803) return -1;
	dataPtr += 4;
	int dataSize = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;
	int dataNumRows = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;
	int dataNumCols = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;


	// LOAD LABEL DATA
	ImageInfo targetInfo(resources[1]);
	targetInfo.load();

	// READ LABEL DATA META DATA
	unsigned char *targetPtr = targetInfo.getBufferPtrAt(0);
	int targetMagicNumber = Util::pack4BytesToInt(targetPtr);
	if(targetMagicNumber != 0x00000801) return -1;
	targetPtr += 4;
	//int labelSize = Util::pack4BytesToInt(targetPtr);
	targetPtr += 4;



	int dataArea = dataNumRows * dataNumCols;
	for(int i = 0; i < dataSize; i++) {
		const DataSample *dataSample = new DataSample(dataPtr, dataArea, targetPtr, 10);
		dataSet.push_back(dataSample);

		if(i < 10) {
			//Util::printVec(dataSample->getData(), "data");
			//Util::printVec(dataSample->getTarget(), "target");
		}
	}
	return dataSize;
}
*/


#if CPU_MODE
int MnistDataSet::loadDataSetFromResource(string resources[2], DataSample *&dataSet, int offset, int size) {
	// LOAD IMAGE DATA
	ImageInfo dataInfo(resources[0]);
	dataInfo.load();

	// READ IMAGE DATA META DATA
	unsigned char *dataPtr = dataInfo.getBufferPtrAt(0);
	int dataMagicNumber = Util::pack4BytesToInt(dataPtr);
	if(dataMagicNumber != 0x00000803) return -1;
	dataPtr += 4;
	int dataSize = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;
	int dataNumRows = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;
	int dataNumCols = Util::pack4BytesToInt(dataPtr);
	dataPtr += 4;

	// LOAD LABEL DATA
	ImageInfo targetInfo(resources[1]);
	targetInfo.load();

	// READ LABEL DATA META DATA
	unsigned char *targetPtr = targetInfo.getBufferPtrAt(0);
	int targetMagicNumber = Util::pack4BytesToInt(targetPtr);
	if(targetMagicNumber != 0x00000801) return -1;
	targetPtr += 4;
	//int labelSize = Util::pack4BytesToInt(targetPtr);
	targetPtr += 4;

	if(offset >= dataSize) return 0;

	dataPtr += offset;
	targetPtr += offset;

	int stop = dataSize;
	if(size > 0) stop = min(dataSize, offset+size);

	//int dataArea = dataNumRows * dataNumCols;
	if(dataSet) delete dataSet;
	dataSet = new DataSample[stop-offset];

	for(int i = offset; i < stop; i++) {
		//const DataSample *dataSample = new DataSample(dataPtr, dataArea, targetPtr, 10);
		//dataSet.push_back(dataSample);
		//dataSet[i-offset].readData(dataPtr, dataNumRows, dataNumCols, 1, targetPtr, 10);
		dataSet[i-offset].readData(dataPtr, dataNumRows, dataNumCols, 1, targetPtr, 10);
	}
	return stop-offset;
}

#else
int MnistDataSet::loadDataSetFromResource(string resources[2], vector<DATATYPE> *&dataSet, vector<UINT> *&labelSet, int offset, int size) {
	FILE *imfp = fopen(resources[0].c_str(), "rb");
	if(!imfp) {
		printf("ERROR: Cannot open image dataset %s\n", resources[0].c_str());
		return 0;
	}
	FILE *lbfp = fopen(resources[1].c_str(), "rb");
	if(!lbfp) {
		fclose(imfp);
		printf("ERROR: Cannot open label dataset %s\n", resources[1].c_str());
		return 0;
	}

	UByteImageDataset image_header;
	UByteLabelDataset label_header;

	// Read and verify file headers
	if(fread(&image_header, sizeof(UByteImageDataset), 1, imfp) != 1) {
		printf("ERROR: Invalid dataset file (image file header)\n");
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}
	if(fread(&label_header, sizeof(UByteLabelDataset), 1, lbfp) != 1) {
		printf("ERROR: Invalid dataset file (label file header)\n");
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}

	// Byte-swap data structure values (change endianness)
	image_header.Swap();
	label_header.Swap();

	// Verify datasets
	if(image_header.magic != UBYTE_IMAGE_MAGIC) {
		printf("ERROR: Invalid dataset file (image file magic number)\n");
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}
	if (label_header.magic != UBYTE_LABEL_MAGIC) 	{
		printf("ERROR: Invalid dataset file (label file magic number)\n");
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}
	if (image_header.length != label_header.length) {
		printf("ERROR: Dataset file mismatch (number of images does not match the number of labels)\n");
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}

	// Output dimensions
	size_t width = image_header.width;
	size_t height = image_header.height;
	this->cols = width;
	this->rows = height;
	this->channels = 1;
	this->dataSize = rows*cols*channels;

	// Read images and labels (if requested)
	size_t dataSetSize = image_header.length*width*height;
	dataSet = new vector<DATATYPE>(dataSetSize);
	vector<uint8_t> *tempDataSet = new vector<uint8_t>(dataSetSize);
	labelSet = new vector<UINT>(label_header.length);
	vector<uint8_t> *tempLabelSet = new vector<uint8_t>(label_header.length);

	if(fread(&(*tempDataSet)[0], sizeof(uint8_t), dataSetSize, imfp) != dataSetSize) {
		printf("ERROR: Invalid dataset file (partial image dataset)\n");
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}

	for(size_t i = 0; i < dataSetSize; i++) {
		//if((*tempDataSet)[i] > 0) {
		//	cout << bswap((*tempDataSet)[i]) << endl;
		//}
		(*dataSet)[i] = (*tempDataSet)[i]/255.0f;
	}

	if (fread(&(*tempLabelSet)[0], sizeof(uint8_t), label_header.length, lbfp) != label_header.length) {
		printf("ERROR: Invalid dataset file (partial label dataset)\n");
		fclose(imfp);
		fclose(lbfp);
		return 0;
	}

	for(size_t i = 0; i < label_header.length; i++) {
		(*labelSet)[i] = (*tempLabelSet)[i];
	}


	fclose(imfp);
	fclose(lbfp);

	return image_header.length;
}
#endif



void MnistDataSet::shuffleTrainDataSet() {
	random_shuffle(&trainDataSet[0], &trainDataSet[numTrainData]);
}

void MnistDataSet::shuffleValidationDataSet() {
	random_shuffle(&validationDataSet[0], &validationDataSet[numValidationData]);
}

void MnistDataSet::shuffleTestDataSet() {
	random_shuffle(&testDataSet[0], &testDataSet[numTestData]);
}











































