#include <iostream>

#include "../dataset/ImagePackDataSet.h"
#include "../Timer.h"
#include "../util/ImagePacker.h"
#include "../Util.h"

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>


using namespace std;


void imagepacker_test(int numCategory, int numTrain, int numTest, int numImagesInTrainFile, int numImagesInTestFile, int numChannels);
void imagepackdataset_test();
void xavier_test();
void mnist_test(const string data_path, const string label_path, const string dataFile, const string labelFile);

int main_test(int argc, char** argv) {

	/*
	int numCategory = atoi(argv[1]);
	int numTrain = atoi(argv[2]);
	int numTest = atoi(argv[3]);
	int numImagesInTrainFile = atoi(argv[4]);
	int numImagesInTestFile = atoi(argv[5]);
	int numChannels = atoi(argv[6]);
	imagepacker_test(numCategory, numTrain, numTest, numImagesInTrainFile, numImagesInTestFile, numChannels);
	*/

	//"/home/jhkim/data/learning/mnist/train-images.idx3-ubyte",
	//"/home/jhkim/data/learning/mnist/train-labels.idx1-ubyte",
	//"/home/jhkim/data/learning/mnist/t10k-images.idx3-ubyte",
	//"/home/jhkim/data/learning/mnist/t10k-labels.idx1-ubyte",
	/*
	mnist_test(
			"/home/jhkim/data/learning/mnist/train-images.idx3-ubyte",
			"/home/jhkim/data/learning/mnist/train-labels.idx1-ubyte",
			"/home/jhkim/data/learning/mnist/train_data0",
			"/home/jhkim/data/learning/mnist/train_label0");*/
	mnist_test(
			"/home/jhkim/data/learning/mnist/t10k-images.idx3-ubyte",
			"/home/jhkim/data/learning/mnist/t10k-labels.idx1-ubyte",
			"/home/jhkim/data/learning/mnist/test_data0",
			"/home/jhkim/data/learning/mnist/test_label0");

	//imagepackdataset_test();
	//xavier_test();

	return 0;
}


#include "../util/UByteImage.h"

struct UByteImageDatasetLocal {
	uint32_t magic;			///< 매직 넘버 (UBYTE_IMAGE_MAGIC).
	uint32_t length;		///< 데이터셋 파일에 들어있는 이미지의 수
	uint32_t height;		///< 각 이미지의 높이값
	uint32_t width;			///< 각 이미지의 너비값
	/**
	 * @details 헤더의 각 필드별로 swap하여 저장한다.
	 */
	void Swap() {
		magic = bswap(magic);
		length = bswap(length);
		height = bswap(height);
		width = bswap(width);
	}
};
/**
 * @brief 데이터셋 정답 파일 헤더 구조체
 */
struct UByteLabelDatasetLocal {
	uint32_t magic;			///< 매직 넘버 (UBYTE_LABEL_MAGIC).
	uint32_t length;		///< 데이터셋 파일에 들어있는 정답의 수
	/**
	 * @details 헤더의 각 필드별로 swap하여 저장한다.
	 */
	void Swap() {
		magic = bswap(magic);
		length = bswap(length);
	}
};


void mnist_test(const string data_path, const string label_path, const string dataFile, const string labelFile) {
	FILE *imfp = fopen(data_path.c_str(), "rb");
	FILE *lbfp = fopen(label_path.c_str(), "rb");
	UByteImageDatasetLocal image_header;
	UByteLabelDatasetLocal label_header;

	// Read and verify file headers
	if(fread(&image_header, sizeof(UByteImageDatasetLocal), 1, imfp) != 1) {
		cout << "ERROR: Invalid dataset file (image file header)" << endl;
		fclose(imfp);
		fclose(lbfp);
		exit(1);
	}
	if(fread(&label_header, sizeof(UByteLabelDatasetLocal), 1, lbfp) != 1) {
		cout << "ERROR: Invalid dataset file (label file header)" << endl;
		fclose(imfp);
		fclose(lbfp);
		exit(1);
	}

	// Byte-swap data structure values (change endianness)
	image_header.Swap();
	label_header.Swap();

	// Verify datasets
	if(image_header.magic != UBYTE_IMAGE_MAGIC) {
		printf("ERROR: Invalid dataset file (image file magic number)\n");
		fclose(imfp);
		fclose(lbfp);
		exit(1);
	}
	if (label_header.magic != UBYTE_LABEL_MAGIC) 	{
		printf("ERROR: Invalid dataset file (label file magic number)\n");
		fclose(imfp);
		fclose(lbfp);
		exit(1);
	}
	if (image_header.length != label_header.length) {
		printf("ERROR: Dataset file mismatch (number of images does not match the number of labels)\n");
		fclose(imfp);
		fclose(lbfp);
		exit(1);
	}

	// Output dimensions
	size_t width = image_header.width;
	size_t height = image_header.height;
	size_t dataSize = width*height;

	// Read images and labels (if requested)
	size_t dataSetSize = ((size_t)image_header.length)*dataSize;
	uint8_t* dataSet = new uint8_t[dataSetSize];
	uint8_t* labelSet8 = new uint8_t[label_header.length];
	uint32_t* labelSet32 = new uint32_t[label_header.length];

	if(fread(dataSet, sizeof(uint8_t), dataSetSize, imfp) != dataSetSize) {
		printf("ERROR: Invalid dataset file (partial image dataset)\n");
		fclose(imfp);
		fclose(lbfp);
		exit(1);
	}
	if (fread(labelSet8, sizeof(uint8_t), label_header.length, lbfp) != label_header.length) {
		printf("ERROR: Invalid dataset file (partial label dataset)\n");
		fclose(imfp);
		fclose(lbfp);
		exit(1);
	}
	for(uint32_t i = 0; i < label_header.length; i++) {
		labelSet32[i] = labelSet8[i];
	}

	fclose(imfp);
	fclose(lbfp);

















	UByteImageDataset imageDataSet;
	imageDataSet.magic = image_header.magic;
	imageDataSet.length = image_header.length;
	imageDataSet.height = image_header.height;
	imageDataSet.width = image_header.width;
	imageDataSet.channel = 1;

	UByteLabelDataset labelDataSet;
	labelDataSet.magic = label_header.magic;
	labelDataSet.length = label_header.length;

	imageDataSet.Swap();
	labelDataSet.Swap();

	int imagesInFileCount = 0;
	ofstream* ofsData = new ofstream(dataFile.c_str(), ios::out | ios::binary);
	ofsData->write((char *)&imageDataSet, sizeof(UByteImageDataset));
	ofstream* ofsLabel = new ofstream(labelFile.c_str(), ios::out | ios::binary);
	ofsLabel->write((char *)&labelDataSet, sizeof(UByteLabelDataset));


	ofsData->write((char*)dataSet, sizeof(uint8_t)*dataSetSize);
	ofsLabel->write((char*)labelSet32, sizeof(uint32_t)*label_header.length);


	if(ofsData) {
		ofsData->close();
		ofsData = 0;
	}
	if(ofsLabel) {
		ofsLabel->close();
		ofsLabel = 0;
	}

}


void xavier_test() {
	int size = 10000;
	float max = -100;
	float min = 100;
	float sd_xavier = -0.5;
	std::random_device rd_xavier;
	std::mt19937 gen_xavier(rd_xavier());
	//std::uni _distribution<DATATYPE> normal_dist(0.0, 1.0);
	std::uniform_real_distribution<DATATYPE> unifrom_dist(-sd_xavier, sd_xavier);
	for(int i = 0; i < size; i++) {
		float g = unifrom_dist(gen_xavier);
		//cout << g << endl;
		if(g > max) max = g;
		else if(g < min) min = g;
	}
	cout << "min: " << min << ", max: " << max << endl;
}



void imagepacker_test(int numCategory,
		int numTrain,
		int numTest,
		int numImagesInTrainFile,
		int numImagesInTestFile,
		int numChannels) {

	cout << "numCategory: " << numCategory << endl <<
			"numTrain: " << numTrain << endl <<
			"numTest: " << numTest << endl <<
			"numImagesInTrainFile: " << numImagesInTrainFile << endl <<
			"numImagesInTestFile: " << numImagesInTestFile << endl <<
			"numChannels: " << numChannels << endl;

	Timer timer;
	ImagePacker imagePacker("/home/jhkim/image/ILSVRC2012", numCategory, numTrain, numTest, numImagesInTrainFile, numImagesInTestFile, numChannels);
	cout << "start to load ... " << endl;
	timer.start();
	imagePacker.load();
	cout << "load done ... : " << timer.stop(false) << endl;
	//imagePacker.show();
	timer.start();
	imagePacker.pack();
	cout << "pack done ... : " << timer.stop(false) << endl;

}

void imagepackdataset_test() {

	ImagePackDataSet dataSet(
			"/home/jhkim/image/ILSVRC2012/save/train_data",
			"/home/jhkim/image/ILSVRC2012/save/train_label",
			10,
			"/home/jhkim/image/ILSVRC2012/save/test_data",
			"/home/jhkim/image/ILSVRC2012/save/test_label",
			1);

	dataSet.load();

	const DATATYPE* data = 0;
	const UINT* label = 0;
	data = dataSet.getTrainDataAt(0);
	label = dataSet.getTrainLabelAt(0);

	data = dataSet.getTrainDataAt(367);
	label = dataSet.getTrainLabelAt(367);

	cout << "done ... " << endl;
	cout << "done ... " << endl;

	return;

}



















