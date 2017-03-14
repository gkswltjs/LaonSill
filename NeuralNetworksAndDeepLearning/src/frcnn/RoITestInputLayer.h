/*
 * RoITestInputLayer.h
 *
 *  Created on: Dec 16, 2016
 *      Author: jkim
 */

#ifndef ROITESTINPUTLAYER_H_
#define ROITESTINPUTLAYER_H_

#include <opencv2/highgui/highgui.hpp>

#include "frcnn_common.h"
#include "InputLayer.h"
#include "IMDB.h"

template <typename Dtype>
class RoITestInputLayer : public InputLayer<Dtype> {
public:
	/**
	 * @brief 입력 레이어 객체 빌더
	 * @details 입력 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 레이어 입력 객체를 생성한다.
	 */
	class Builder : public InputLayer<Dtype>::Builder {
	public:
		uint32_t _numClasses;
		std::vector<float> _pixelMeans;

		Builder() {
			this->type = Layer<Dtype>::RoITestInput;
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
		virtual Builder* numClasses(const uint32_t numClasses) {
			this->_numClasses = numClasses;
			return this;
		}
		virtual Builder* pixelMeans(const std::vector<float>& pixelMeans) {
			this->_pixelMeans = pixelMeans;
			return this;
		}
		Layer<Dtype>* build() {
			return new RoITestInputLayer(this);
		}
	};

	RoITestInputLayer(Builder* builder);
	virtual ~RoITestInputLayer();

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	virtual void reshape();
	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

private:
	void initialize();

	void getNextMiniBatch();

	void imDetect(cv::Mat& im);
	void getBlobs();
	void getImageBlob(cv::Mat& im);
	void imToBlob(cv::Mat& im);

	IMDB* combinedRoidb(const std::string& imdb_name);
	IMDB* getRoidb(const std::string& imdb_name);
	IMDB* getImdb(const std::string& imdb_name);


public:
	IMDB* imdb;

	uint32_t numClasses;
	std::vector<float> pixelMeans;
	std::vector<uint32_t> perm;
	uint32_t cur;



	std::vector<cv::Scalar> boxColors;
};

#endif /* ROITESTINPUTLAYER_H_ */


















