/*
 * AnnotationDataLayer.cpp
 *
 *  Created on: Apr 19, 2017
 *      Author: jkim
 */

#include "AnnotationDataLayer.h"
#include "tinyxml2/tinyxml2.h"
#include "NetworkConfig.h"

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
  pixelMeans(builder->_pixelMeans) {

	initialize();
}

template <typename Dtype>
AnnotationDataLayer<Dtype>::~AnnotationDataLayer() {
	if (this->data) {
		delete this->data;
	}
	if (this->label) {
		delete this->label;
	}
}


template <typename Dtype>
void AnnotationDataLayer<Dtype>::initialize() {
	this->data = new Data<Dtype>("data", true);
	this->data->reshape({1, this->imageHeight, this->imageWidth, 3});
	this->label = new Data<Dtype>("label", true);
	this->label->reshape({1, 1, 1, 8});

	this->labelMap.build();

	loadODRawDataPath();
	loadODRawDataIm();
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
	Layer<Dtype>::_adjustInputShape();

	const uint32_t inputSize = this->_inputData.size();
	for (uint32_t i = 0; i < inputSize; i++) {
		if (!Layer<Dtype>::_isInputShapeChanged(i))
			continue;

		// "data"
		if (i == 0) {


		}
		// "label"
		else if (i == 1) {

		}
	}
}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::feedforward() {
	reshape();
	getNextMiniBatch();

}

template <typename Dtype>
void AnnotationDataLayer<Dtype>::feedforward(const uint32_t baseIndex, const char* end) {
	reshape();
	getNextMiniBatch();
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
		ODRawData<Dtype>& odRawData = this->odRawDataList[i];
		cv::Mat im = cv::imread(this->baseDataPath + odRawData.imPath);
		im.convertTo(im, CV_32F);

		float imHeightScale = float(this->imageHeight) / float(im.rows);
		float imWidthScale = float(this->imageWidth) / float(im.cols);
		cv::resize(im, im, cv::Size(), imHeightScale, imWidthScale, CV_INTER_LINEAR);

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

		if (boundingBox.diff == 0) {
			odRawData.boundingBoxes.push_back(boundingBox);
		}
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

	uint32_t totalBBs = 0;
	for (int i = 0; i < inds.size(); i++) {
		ODMetaData<Dtype>& odMetaData = this->odMetaDataList[inds[i]];
		ODRawData<Dtype>& odRawData = this->odRawDataList[odMetaData.rawIdx];
		totalBBs += odRawData.boundingBoxes.size();
	}

	this->_outputData[0]->reshape({this->networkConfig->_batchSize, 3, this->imageHeight,
		this->imageWidth});
	this->_outputData[1]->reshape({1, 1, totalBBs, 8});

	Dtype* hData = this->_outputData[0]->mutable_host_data();
	Dtype* hLabel = this->_outputData[1]->mutable_host_data();

	const vector<uint32_t> dataShape = {1, this->imageHeight, this->imageWidth, 3};
	int bbIdx = 0;
	Dtype buf[8];
	for (int i = 0; i < inds.size(); i++) {
		ODMetaData<Dtype>& odMetaData = this->odMetaDataList[inds[i]];
		ODRawData<Dtype>& odRawData = this->odRawDataList[odMetaData.rawIdx];

		// flip image
		cv::Mat copiedIm;
		if (odMetaData.flip) {
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
		this->data->reshape(dataShape);
		this->data->set_host_data((Dtype*)copiedIm.data);
		this->data->transpose({0, 3, 1, 2}); // BGR,BGR,... to BB..,GG..,RR..

		std::copy(this->data->host_data(), this->data->host_data() + this->data->getCount(),
				hData + i*this->data->getCount());

		// label
		const int numBBs = odRawData.boundingBoxes.size();
		for (int j = 0; j < numBBs; j++) {
			buildLabelData(odMetaData, j, buf);
			buf[0] = i;
			std::copy(buf, buf + 8, hLabel + bbIdx*8);
			bbIdx++;
		}
	}

	this->_printOn();
	this->_outputData[0]->print_data({}, false);
	this->_outputData[1]->print_data({}, false, -1);
	this->_printOff();

	this->_inputShape[0] = this->_outputData[0]->getShape();
	this->_inputShape[1] = this->_outputData[1]->getShape();
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


template class AnnotationDataLayer<float>;





























