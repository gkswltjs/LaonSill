#include <iostream>

#include "../dataset/UbyteDataSet.h"
#include "../Timer.h"
#include "../util/ImagePacker.h"
#include "../Util.h"

#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>


using namespace std;


void imagepacker_test(int numCategory, int numTrain, int numTest, int numImagesInTrainFile, int numImagesInTestFile, int numChannels);
void ubytedataset_test();
void xavier_test();

int main(int argc, char** argv) {


	/*
	int numCategory = atoi(argv[1]);
	int numTrain = atoi(argv[2]);
	int numTest = atoi(argv[3]);
	int numImagesInTrainFile = atoi(argv[4]);
	int numImagesInTestFile = atoi(argv[5]);
	int numChannels = atoi(argv[6]);
	imagepacker_test(numCategory, numTrain, numTest, numImagesInTrainFile, numImagesInTestFile, numChannels);
	*/

	//ubytedataset_test();
	//xavier_test();

	return 0;
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

void ubytedataset_test() {

	UbyteDataSet dataSet(
			"/home/jhkim/image/ILSVRC2012/save/train_data",
			"/home/jhkim/image/ILSVRC2012/save/train_label",
			10,
			"/home/jhkim/image/ILSVRC2012/save/test_data",
			"/home/jhkim/image/ILSVRC2012/save/test_label",
			1,
			0.8);

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



















