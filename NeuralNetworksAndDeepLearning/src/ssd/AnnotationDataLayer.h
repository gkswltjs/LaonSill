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
#include "ssd_common.h"



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

		bool _shuffle;

		Builder() {
			this->type = Layer<Dtype>::AnnotationData;
			this->_flip = false;
			this->_imageHeight = 300;
			this->_imageWidth = 300;
			this->_pixelMeans = {Dtype(0.0), Dtype(0.0), Dtype(0.0)};
			this->_shuffle = true;
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
		virtual Builder* shuffle(const bool shuffle) {
			this->_shuffle = shuffle;
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
	bool bShuffle;

	std::vector<ODRawData<Dtype>> odRawDataList;
	std::vector<ODMetaData<Dtype>> odMetaDataList;

	LabelMap<Dtype> labelMap;

	std::vector<int> perm;
	int cur;

	Data<Dtype> data;	///< feedforward할 data를 준비하는 buffer.



public:
	std::map<std::string, int> refCount;
};

#endif /* ANNOTATIONDATALAYER_H_ */
