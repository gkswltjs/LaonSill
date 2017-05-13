/*
 * AnnotationDataLayer.cpp
 *
 *  Created on: Apr 19, 2017
 *      Author: jkim
 */

#include "AnnotationDataLayer.h"
#include "tinyxml2/tinyxml2.h"
#include "NetworkConfig.h"
#include "StdOutLog.h"

using namespace std;




template <typename Dtype>
LabelMap<Dtype>::LabelMap(const string& labelMapPath)
: labelMapPath(labelMapPath) {

	// background
	this->colorList.push_back(cv::Scalar(0, 0, 0));

	this->colorList.push_back(cv::Scalar(10, 163, 240));
	this->colorList.push_back(cv::Scalar(44, 90, 130));
	this->colorList.push_back(cv::Scalar(239, 80, 0));
	this->colorList.push_back(cv::Scalar(37, 0, 162));
	this->colorList.push_back(cv::Scalar(226, 161, 27));

	this->colorList.push_back(cv::Scalar(115, 0, 216));
	this->colorList.push_back(cv::Scalar(0, 196, 164));
	this->colorList.push_back(cv::Scalar(255, 0, 106));
	this->colorList.push_back(cv::Scalar(23, 169, 96));
	this->colorList.push_back(cv::Scalar(0, 138, 0));

	this->colorList.push_back(cv::Scalar(138, 96, 118));
	this->colorList.push_back(cv::Scalar(100, 135, 109));
	this->colorList.push_back(cv::Scalar(0, 104, 250));
	this->colorList.push_back(cv::Scalar(208, 114, 244));
	this->colorList.push_back(cv::Scalar(0, 20, 229));

	this->colorList.push_back(cv::Scalar(63, 59, 122));
	this->colorList.push_back(cv::Scalar(135, 118, 100));
	this->colorList.push_back(cv::Scalar(169, 171, 0));
	this->colorList.push_back(cv::Scalar(255, 0, 170));
	this->colorList.push_back(cv::Scalar(0, 193, 216));

}

template <typename Dtype>
void LabelMap<Dtype>::build() {
	ifstream ifs(this->labelMapPath.c_str(), ios::in);
	if (!ifs.is_open()) {
		cout << "no such file: " << this->labelMapPath << endl;
		exit(1);
	}

	string part1;
	string part2;
	LabelItem labelItem;
	bool started = false;
	int line = 1;
	while (ifs >> part1 >> part2) {
		//cout << "part1: '" << part1 << "', part2: '" << part2 << "'" << endl;
		if (part1 == "item" && part2 == "{") {
			started = true;
		}
		else if (started && part1 == "name:") {
			part2 = part2.substr(1, part2.length()-2);
			labelItem.name = part2;
		}
		else if (started && part1 == "label:") {
			labelItem.label = atoi(part2.c_str());
		}
		else if (started && part1 == "display_name:") {
			part2 = part2.substr(1, part2.length()-2);
			labelItem.displayName = part2;

			ifs >> part1;
			if (part1 == "}") {
				this->labelItemList.push_back(labelItem);
				started = false;
			}
		}
		else {
			cout << "invalid label map format at line: " << line << endl;
			exit(1);
		}
		line++;
	}

	for (int i = 0; i < this->labelItemList.size(); i++) {
		LabelItem& labelItem = this->labelItemList[i];
		this->labelToIndMap[labelItem.displayName] = labelItem.label;
		this->indToLabelMap[labelItem.label] = labelItem.displayName;
	}
}

template <typename Dtype>
int LabelMap<Dtype>::convertLabelToInd(const string& label) {
	if (this->labelToIndMap.find(label) == this->labelToIndMap.end()) {
		cout << "invalid label: " << label << endl;
		exit(1);
	}
	return this->labelToIndMap[label];
}

template <typename Dtype>
string LabelMap<Dtype>::convertIndToLabel(int ind) {
	if (this->indToLabelMap.find(ind) == this->indToLabelMap.end()) {
		cout << "invalid ind: " << ind << endl;
		exit(1);
	}
	return this->indToLabelMap[ind];
}

template <typename Dtype>
void LabelMap<Dtype>::printLabelItemList() {
	for (int i = 0; i < this->labelItemList.size(); i++) {
		cout << "label item #" << i << endl;
		this->labelItemList[i].print();
	}
}

template <typename Dtype>
void LabelMap<Dtype>::LabelItem::print() {
	cout << "LabelItem: " 		<< this->name 			<< endl;
	cout << "\tlabel: " 		<< this->label 			<< endl;
	cout << "\tdisplay_name: " 	<< this->displayName 	<< endl;
}

template <typename Dtype>
void BoundingBox<Dtype>::print() {
	cout << "BoundingBox: " << this->name	<< endl;
	cout << "\tlabel: " 	<< this->label	<< endl;
	cout << "\txmin: " 		<< this->xmin	<< endl;
	cout << "\tymin: " 		<< this->ymin	<< endl;
	cout << "\txmax: " 		<< this->xmax	<< endl;
	cout << "\tymax: " 		<< this->ymax	<< endl;
	cout << "\tdiff: " 		<< this->diff	<< endl;
}


template <typename Dtype>
void ODRawData<Dtype>::print() {
	cout << "ODRawData" 						<< endl;
	cout << "\timPath: " 	<< this->imPath 	<< endl;
	cout << "\tannoPath: "	<< this->annoPath	<< endl;
	cout << "\twidth: " 	<< this->width 		<< endl;
	cout << "\theight: " 	<< this->height 	<< endl;
	cout << "\tdepth: " 	<< this->depth 		<< endl;
	cout << "\tboundingBoxes: " 				<< endl;

	for (int i = 0; i < this->boundingBoxes.size(); i++) {
		this->boundingBoxes[i].print();
	}
}

template <typename Dtype>
void ODRawData<Dtype>::displayBoundingBoxes(const string& baseDataPath,
		vector<cv::Scalar>& colorList) {
	cv::Mat im = cv::imread(baseDataPath + this->imPath);

	for (int i = 0; i < this->boundingBoxes.size(); i++) {
		BoundingBox<Dtype>& bb = this->boundingBoxes[i];

		cv::rectangle(im, cv::Point(bb.xmin, bb.ymin), cv::Point(bb.xmax, bb.ymax),
				colorList[bb.label], 2);
		cv::putText(im, bb.name , cv::Point(bb.xmin, bb.ymin+15.0f), 2, 0.5f,
				colorList[bb.label]);
	}

	const string windowName = this->imPath;
	cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	cv::imshow(windowName, im);

	cv::waitKey(0);
	cv::destroyAllWindows();
}























template <typename Dtype>
AnnotationDataLayer<Dtype>::AnnotationDataLayer(Builder* builder)
: InputLayer<Dtype>(builder),
  flip(builder->_flip),
  imageHeight(builder->_imageHeight),
  imageWidth(builder->_imageWidth),
  imageSetPath(builder->_imageSetPath),
  baseDataPath(builder->_baseDataPath),
  labelMap(builder->_labelMapPath),
  pixelMeans(builder->_pixelMeans),
  data("data", true) {

	initialize();
}

template <typename Dtype>
AnnotationDataLayer<Dtype>::~AnnotationDataLayer() {

}


template <typename Dtype>
void AnnotationDataLayer<Dtype>::initialize() {
	this->data.reshape({1, this->imageHeight, this->imageWidth, 3});
	this->labelMap.build();

	loadODRawDataPath();
	//loadODRawDataIm();
	loadODRawDataAnno();

	/*
	for (int i = 0; i < this->odRawDataList.size(); i++) {
		this->odRawDataList[i].displayBoundingBoxes(this->baseDataPath,
				this->labelMap.colorList);
		this->odRawDataList[i].print();
		cout << endl;
	}
	*/
	loadODMetaData();

	this->perm.resize(this->odMetaDataList.size());
	std::iota(this->perm.begin(), this->perm.end(), 0);
	shuffle();
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::reshape() {
	if (this->_inputData.size() < 1) {
		for (int i = 0; i < this->_outputs.size(); i++) {
			this->_inputs.push_back(this->_outputs[i]);
			this->_inputData.push_back(this->_outputData[i]);
		}
	}
	bool adjusted = Layer<Dtype>::_adjustInputShape();
	if (adjusted) {
		// AnnotationDataLayer에서 inputShape를 확인하지 않을 것이므로
		// inputShape를 별도로 갱신하지는 않음.
		// for "data" Data
		this->_inputData[0]->reshape(
				{this->networkConfig->_batchSize, 3, this->imageHeight, this->imageWidth});
		// for "label" Data
		// 1. item_id 2. group_label 3. instance_id
		// 4. xmin 5. ymin 6. xmax 7. ymax 8. difficult
		// cf. [2]: numBBs
		// numBBs를 max 추정하여 미리 충분히 잡아 두면
		// 실행시간에 추가 메모리 할당이 없을 것. batch * 10으로 추정
		//this->_inputData[1]->reshape({1, 1, this->networkConfig->_batchSize * 10, 8});
		this->_inputData[1]->reshape({1, 1, 2, 8});
	}
}

template <typename Dtype>
int AnnotationDataLayer<Dtype>::getNumTrainData() {
	return this->odMetaDataList.size();
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::shuffleTrainDataSet() {
	return;
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::feedforward() {
	reshape();
	getNextMiniBatch();
	//verifyData();
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
	reshape();
	getNextMiniBatch();
	//verifyData();
}


template <typename Dtype>
void AnnotationDataLayer<Dtype>::loadODRawDataPath() {
	ifstream ifs(this->imageSetPath.c_str(), ios::in);
	if (!ifs.is_open()) {
		cout << "no such file: " << this->imageSetPath << endl;
		exit(1);
	}

	ODRawData<Dtype> odRawData;
	string imPath;
	string annoPath;
	while (ifs >> imPath >> annoPath) {
		odRawData.imPath = imPath;
		odRawData.annoPath = annoPath;
		this->odRawDataList.push_back(odRawData);
	}

	ifs.close();
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::loadODRawDataIm() {
	const int numODRawData = this->odRawDataList.size();
	for (int i = 0; i < numODRawData; i++) {
		if (i % 1000 == 0) {
			cout << "loadODRawDataIm(): " << i << endl;
		}

		ODRawData<Dtype>& odRawData = this->odRawDataList[i];
		cv::Mat im = cv::imread(this->baseDataPath + odRawData.imPath);
		im.convertTo(im, CV_32FC3);

		//float imHeightScale = float(this->imageHeight) / float(im.rows);
		//float imWidthScale = float(this->imageWidth) / float(im.cols);
		cv::resize(im, im, cv::Size(this->imageWidth, this->imageHeight), 0, 0, CV_INTER_LINEAR);

		odRawData.im = im;
	}
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::loadODRawDataAnno() {
	const int numODRawData = this->odRawDataList.size();
	for (int i = 0; i < numODRawData; i++) {
		ODRawData<Dtype>& odRawData = this->odRawDataList[i];
		readAnnotation(odRawData);

		//normalize bounding box coordinates
		for (int j = 0; j < odRawData.boundingBoxes.size(); j++) {
			BoundingBox<Dtype>& bb = odRawData.boundingBoxes[j];
			bb.buf[0] = 0;									// item_id
			bb.buf[1] = Dtype(bb.label);					// group_label
			bb.buf[2] = 0;									// instance_id
			bb.buf[3] = Dtype(bb.xmin) / odRawData.width;	// xmin
			bb.buf[4] = Dtype(bb.ymin) / odRawData.height;	// ymin
			bb.buf[5] = Dtype(bb.xmax) / odRawData.width;	// xmax
			bb.buf[6] = Dtype(bb.ymax) / odRawData.height;	// ymax
			bb.buf[7] = Dtype(bb.diff);						// difficult
		}
	}
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::readAnnotation(ODRawData<Dtype>& odRawData) {
	tinyxml2::XMLDocument annotationDocument;
	tinyxml2::XMLNode* annotationNode;

	const string filePath = this->baseDataPath + odRawData.annoPath;
	annotationDocument.LoadFile(filePath.c_str());
	annotationNode = annotationDocument.FirstChild();

	// filename
	//tinyxml2::XMLElement* filenameElement = annotationNode->FirstChildElement("filename");
	//annotation.filename = filenameElement->GetText();

	// size
	tinyxml2::XMLElement* sizeElement = annotationNode->FirstChildElement("size");
	sizeElement->FirstChildElement("width")->QueryIntText((int*)&odRawData.width);
	sizeElement->FirstChildElement("height")->QueryIntText((int*)&odRawData.height);
	sizeElement->FirstChildElement("depth")->QueryIntText((int*)&odRawData.depth);

	// object
	for (tinyxml2::XMLElement* objectElement =
			annotationNode->FirstChildElement("object"); objectElement != 0;
			objectElement = objectElement->NextSiblingElement("object")) {
		BoundingBox<Dtype> boundingBox;
		boundingBox.name = objectElement->FirstChildElement("name")->GetText();
		boundingBox.label = this->labelMap.convertLabelToInd(boundingBox.name);
		objectElement->FirstChildElement("difficult")
					 ->QueryIntText((int*)&boundingBox.diff);

		tinyxml2::XMLElement* bndboxElement = objectElement->FirstChildElement("bndbox");
		bndboxElement->FirstChildElement("xmin")->QueryIntText((int*)&boundingBox.xmin);
		bndboxElement->FirstChildElement("ymin")->QueryIntText((int*)&boundingBox.ymin);
		bndboxElement->FirstChildElement("xmax")->QueryIntText((int*)&boundingBox.xmax);
		bndboxElement->FirstChildElement("ymax")->QueryIntText((int*)&boundingBox.ymax);

		//if (boundingBox.diff == 0) {
			odRawData.boundingBoxes.push_back(boundingBox);
		//}
	}
}



template <typename Dtype>
void AnnotationDataLayer<Dtype>::loadODMetaData() {
	const int numODRawData = this->odRawDataList.size();

	ODMetaData<Dtype> odMetaData;
	for (int i = 0; i < numODRawData; i++) {
		odMetaData.rawIdx = i;
		odMetaData.flip = false;

		this->odMetaDataList.push_back(odMetaData);

		if (this->flip) {
			odMetaData.flip = true;
			this->odMetaDataList.push_back(odMetaData);
		}
	}
}



template <typename Dtype>
void AnnotationDataLayer<Dtype>::shuffle() {
	std::random_shuffle(this->perm.begin(), this->perm.end());
	//cout << "***shuffle is temporaray disabled ... " << endl;
	this->cur = 0;
}


template <typename Dtype>
void AnnotationDataLayer<Dtype>::getNextMiniBatch() {
	vector<int> inds;
	getNextMiniBatchInds(inds);
	getMiniBatch(inds);
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::getNextMiniBatchInds(vector<int>& inds) {
	if (this->cur + this->networkConfig->_batchSize > this->odMetaDataList.size()) {
		shuffle();
	}

	inds.clear();
	inds.insert(inds.end(), this->perm.begin() + this->cur,
			this->perm.begin() + this->cur + this->networkConfig->_batchSize);

	this->cur += this->networkConfig->_batchSize;
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::getMiniBatch(const vector<int>& inds) {

	// Count total bounding boxes in this batch.
	uint32_t totalBBs = 0;
	for (int i = 0; i < inds.size(); i++) {
		ODMetaData<Dtype>& odMetaData = this->odMetaDataList[inds[i]];
		ODRawData<Dtype>& odRawData = this->odRawDataList[odMetaData.rawIdx];
		totalBBs += odRawData.boundingBoxes.size();
	}

	// "data"의 경우 shape가 초기에 결정, reshape가 필요없음.
	//this->_outputData[0]->reshape({this->networkConfig->_batchSize, 3, this->imageHeight,
	//	this->imageWidth});
	// "label"의 경우 현재 batch의 bounding box 수에 따라 shape 변동.
	this->_outputData[1]->reshape({1, 1, totalBBs, 8});

	Dtype* dataData = this->_outputData[0]->mutable_host_data();
	Dtype* labelData = this->_outputData[1]->mutable_host_data();

	// "data" 1개에 대한 shape. cv::Mat의 경우 BGR, BGR, ... 의 구조.
	const vector<uint32_t> dataShape = {1, this->imageHeight, this->imageWidth, 3};
	int bbIdx = 0;
	Dtype buf[8];
	for (int i = 0; i < inds.size(); i++) {
		ODMetaData<Dtype>& odMetaData = this->odMetaDataList[inds[i]];
		ODRawData<Dtype>& odRawData = this->odRawDataList[odMetaData.rawIdx];


		if (odRawData.im.empty()) {
			cv::Mat im = cv::imread(this->baseDataPath + odRawData.imPath);
			im.convertTo(im, CV_32FC3);
			cv::resize(im, im, cv::Size(this->imageWidth, this->imageHeight), 0, 0, CV_INTER_LINEAR);
			odRawData.im = im;
		}

		// transform image ... 추가 transform option 추가될 경우 여기서 처리.
		// 현재는 flip만 적용되어 있음.
		// flip image
		cv::Mat copiedIm;
		if (odMetaData.flip) {
			// 1 means flipping around y-axis
			cv::flip(odRawData.im, copiedIm, 1);
		} else {
			odRawData.im.copyTo(copiedIm);
		}

		// subtract mean
		float* imPtr = (float*)copiedIm.data;
		int n = copiedIm.rows * copiedIm.cols * copiedIm.channels();
		for (int j = 0; j < n; j += 3) {
			imPtr[j + 0] -= this->pixelMeans[0];
			imPtr[j + 1] -= this->pixelMeans[1];
			imPtr[j + 2] -= this->pixelMeans[2];
		}

		// data
		this->data.reshape(dataShape);
		this->data.set_host_data((Dtype*)copiedIm.data);
		this->data.transpose({0, 3, 1, 2}); // BGR,BGR,... to BB..,GG..,RR..

		std::copy(this->data.host_data(), this->data.host_data() + this->data.getCount(),
				dataData + i * this->data.getCount());

		// label
		const int numBBs = odRawData.boundingBoxes.size();
		for (int j = 0; j < numBBs; j++) {
			buildLabelData(odMetaData, j, buf);
			buf[0] = i;
			// XXX: for debugging.
			buf[2] = inds[i];
			std::copy(buf, buf + 8, labelData + bbIdx * 8);
			bbIdx++;
		}
	}

	//this->_printOn();
	//this->_outputData[0]->print_data({}, false);
	//this->_outputData[1]->print_data({}, false, -1);
	//this->_printOff();
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::buildLabelData(ODMetaData<Dtype>& odMetaData, int bbIdx,
		Dtype buf[8]) {
	ODRawData<Dtype>& odRawData = this->odRawDataList[odMetaData.rawIdx];
	BoundingBox<Dtype>& bb = odRawData.boundingBoxes[bbIdx];

	std::copy(bb.buf, bb.buf + 8, buf);

	// flip horizontally only
	if (odMetaData.flip) {
		buf[3] = 1.0 - bb.buf[5];				// xmin
		buf[5] = 1.0 - bb.buf[3];				// xmax
	}
}




template <typename Dtype>
void AnnotationDataLayer<Dtype>::verifyData() {
	STDOUT_LOG("VERIFYING DATA ... ");
	const string windowName = "AnnotationDataLayer::verifyData()";
	const int batches = this->networkConfig->_batchSize;
	const int singleImageSize = this->_outputData[0]->getCountByAxis(1);
	const Dtype* dataData = this->_outputData[0]->host_data();
	const Dtype* labelData = this->_outputData[1]->host_data();
	const uint32_t numBBs = this->_outputData[1]->getShape(2);

	Dtype* data = this->data.mutable_host_data();
	Dtype buf[8];
	int bbIdx = 0;
	for (int i = 0; i < batches; i++) {
		int idx = int(labelData[bbIdx * 8 + 2]);
		ODMetaData<Dtype>& odMetaData = this->odMetaDataList[idx];
		ODRawData<Dtype>& odRawData = this->odRawDataList[odMetaData.rawIdx];
		//odRawData.displayBoundingBoxes(this->baseDataPath, this->labelMap.colorList);

		this->data.reshape({1, 3, this->imageHeight, this->imageWidth});
		std::copy(dataData + i * singleImageSize, dataData + (i + 1) * singleImageSize, data);

		// transpose
		this->data.transpose({0, 2, 3, 1});

		// pixel mean
		for (int j = 0; j < singleImageSize; j += 3) {
			data[j + 0] += this->pixelMeans[0];
			data[j + 1] += this->pixelMeans[1];
			data[j + 2] += this->pixelMeans[2];
		}

		cv::Mat im = cv::Mat(this->imageHeight, this->imageWidth, CV_32FC3, data);
		cv::resize(im, im, cv::Size(odRawData.width, odRawData.height), 0, 0, CV_INTER_LINEAR);


		while (bbIdx < numBBs) {
			idx = int(labelData[bbIdx * 8 + 0]);
			if (idx == i) {
				std::copy(labelData + bbIdx * 8, labelData + (bbIdx + 1) * 8, buf);
				//printArray(buf, 8);

				// 1. label 4. xmin 5. ymin 6. xmax 7. ymax
				int label = int(buf[1]);
				int xmin = int(buf[3] * odRawData.width);
				int ymin = int(buf[4] * odRawData.height);
				int xmax = int(buf[5] * odRawData.width);
				int ymax = int(buf[6] * odRawData.height);

				cv::rectangle(im, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
						this->labelMap.colorList[label], 2);
				cv::putText(im, this->labelMap.convertIndToLabel(label),
						cv::Point(xmin, ymin+15.0f), 2, 0.5f,
						this->labelMap.colorList[label]);
				bbIdx++;
			} else {
				break;
			}
		}

		im.convertTo(im, CV_8UC3);
		cv::namedWindow(windowName, CV_WINDOW_AUTOSIZE);
		cv::imshow(windowName, im);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
}


template <typename Dtype>
void AnnotationDataLayer<Dtype>::printMat(cv::Mat& im, int type) {

	if (type == CV_32F) {
		float* data = (float*)im.data;

		//for (int r = 0; r < )

		for (int i = 0; i < 9; i++) {
			cout << data[i] << ", ";
		}
		cout << endl;
	} else if (type == CV_8U) {
		uchar* data = (uchar*)im.data;
		for (int i = 0; i < 9; i++) {
			cout << uint32_t(data[i]) << ", ";
		}
		cout << endl;
	}
}


template <typename Dtype>
void AnnotationDataLayer<Dtype>::printArray(Dtype* array, int n) {

	for (int i = 0; i < n; i++) {
		cout << array[i] << ", ";
	}
	cout << endl;
}




template class AnnotationDataLayer<float>;





























