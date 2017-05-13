/*
 * AnnotationDataLayer.h
 *
 *  Created on: Apr 19, 2017
 *      Author: jkim
 */

#ifndef ANNOTATIONDATALAYER_H_
#define ANNOTATIONDATALAYER_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "InputLayer.h"


template <typename Dtype>
class BoundingBox {
public:
	void print();
public:
	std::string name;
	int label;
	// unnormalized coords
	int xmin;
	int ymin;
	int xmax;
	int ymax;

	int diff;

	// normalized coords, ready for use
	Dtype buf[8];
};

// Obejct Detection Raw Data
template <typename Dtype>
class ODRawData {
public:
	void print();
	void displayBoundingBoxes(const std::string& baseDataPath,
			std::vector<cv::Scalar>& colorList);
public:
	cv::Mat im;
	std::string imPath;
	std::string annoPath;

	int width;
	int height;
	int depth;

	std::vector<BoundingBox<Dtype>> boundingBoxes;
};

// Object Detection Meta Data
template <typename Dtype>
class ODMetaData {
public:
	int rawIdx;
	bool flip;
};

template <typename Dtype>
class LabelMap {
public:
	class LabelItem {
	public:
		void print();
	public:
		std::string name;
		int label;
		std::string displayName;
	};


public:
	LabelMap(const std::string& labelMapPath);
	void build();

	int convertLabelToInd(const std::string& label);
	std::string convertIndToLabel(int ind);

	void printLabelItemList();

public:
	std::string labelMapPath;
	std::vector<LabelItem> labelItemList;
	std::map<std::string, int> labelToIndMap;
	std::map<int, std::string> indToLabelMap;
	std::vector<cv::Scalar> colorList;
};


template <typename Dtype>
class AnnotationDataLayer : public InputLayer<Dtype> {
public:
	class Builder : public InputLayer<Dtype>::Builder {
	public:
		bool _flip;
		uint32_t _imageHeight;
		uint32_t _imageWidth;

		std::string _imageSetPath;
		std::string _baseDataPath;
		std::string _labelMapPath;

		std::vector<Dtype> _pixelMeans;

		Builder() {
			this->type = Layer<Dtype>::AnnotationData;
			this->_flip = false;
			this->_imageHeight = 300;
			this->_imageWidth = 300;
			this->_pixelMeans = {Dtype(0.0), Dtype(0.0), Dtype(0.0)};
		}
		virtual Builder* name(const std::string name) {
			Layer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			Layer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			Layer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			Layer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			Layer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		virtual Builder* flip(const bool flip) {
			this->_flip = flip;
			return this;
		}
		virtual Builder* imageHeight(const uint32_t imageHeight) {
			this->_imageHeight = imageHeight;
			return this;
		}
		virtual Builder* imageWidth(const uint32_t imageWidth) {
			this->_imageWidth = imageWidth;
			return this;
		}
		virtual Builder* imageSetPath(const std::string& imageSetPath) {
			this->_imageSetPath = imageSetPath;
			return this;
		}
		virtual Builder* baseDataPath(const std::string& baseDataPath) {
			this->_baseDataPath = baseDataPath;
			return this;
		}
		virtual Builder* labelMapPath(const std::string& labelMapPath) {
			this->_labelMapPath = labelMapPath;
			return this;
		}
		virtual Builder* pixelMeans(const std::vector<float>& pixelMeans) {
			this->_pixelMeans = pixelMeans;
			return this;
		}
		Layer<Dtype>* build() {
			return new AnnotationDataLayer(this);
		}
	};

	AnnotationDataLayer(Builder* builder);
	virtual ~AnnotationDataLayer();

	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

	virtual void reshape();

	virtual int getNumTrainData();
	virtual void shuffleTrainDataSet();

private:
	void initialize();
	void shuffle();

	void getNextMiniBatch();
	void getNextMiniBatchInds(std::vector<int>& inds);
	void getMiniBatch(const std::vector<int>& inds);

	/**
	 * @details 데이터셋 파일로부터 img, anno 파일 경로로 ODRawDataList 초기화
	 */
	void loadODRawDataPath();
	/**
	 * @details img 파일을 cv::Mat 형태로 ODRawDataList에 읽어들임. 스케일링 적용.
	 */
	void loadODRawDataIm();
	/**
	 * @details anno 파일을 ODRawDataList에 읽어들임.
	 */
	void loadODRawDataAnno();
	void readAnnotation(ODRawData<Dtype>& odRawData);
	void loadODMetaData();

	void buildLabelData(ODMetaData<Dtype>& odMetaData, int bbIdx, Dtype buf[8]);


	void verifyData();
	void printMat(cv::Mat& im, int type);
	void printArray(Dtype* array, int n);
private:
	bool flip;
	uint32_t imageHeight;	///< 네트워크로 입력되는 이미지 높이. 리사이즈 높이.
	uint32_t imageWidth;	///< 네트워크로 입력되는 이미지 너비. 리사이즈 너비.

	std::string imageSetPath;	///< OD 데이터셋을 정의한 파일의 경로. <img, anno> 페어정보
	std::string baseDataPath;

	std::vector<Dtype> pixelMeans;

	std::vector<ODRawData<Dtype>> odRawDataList;
	std::vector<ODMetaData<Dtype>> odMetaDataList;

	LabelMap<Dtype> labelMap;

	std::vector<int> perm;
	int cur;

	Data<Dtype> data;	///< feedforward할 data를 준비하는 buffer.
};

#endif /* ANNOTATIONDATALAYER_H_ */
