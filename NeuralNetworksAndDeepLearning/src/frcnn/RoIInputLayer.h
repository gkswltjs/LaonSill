/*
 * RoIInputLayer.h
 *
 *  Created on: Nov 11, 2016
 *      Author: jkim
 */

#ifndef ROIINPUTLAYER_H_
#define ROIINPUTLAYER_H_

#include <opencv2/highgui/highgui.hpp>

#include "frcnn_common.h"
#include "DataSet.h"
#include "InputLayer.h"
#include "IMDB.h"

template <typename Dtype>
class RoIInputLayer : public InputLayer<Dtype> {
public:
	/*
	struct InputStat {
		InputStat() {
			nfcnt = 0;
			fcnt = 0;
			scaleCnt[0] = 0;
			scaleCnt[1] = 0;
			scaleCnt[2] = 0;
			scaleCnt[3] = 0;
		}
		int nfcnt;
		int fcnt;
		int scaleCnt[4];
	};
	*/
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
			this->type = Layer<Dtype>::RoIInput;
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
			return new RoIInputLayer(this);
		}
	};

	RoIInputLayer(Builder* builder);
	virtual ~RoIInputLayer();


	virtual void feedforward();
	using InputLayer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	void reshape();

private:
	void initialize();

	IMDB* getImdb(const std::string& imdb_name);
	void getTrainingRoidb(IMDB* imdb);
	IMDB* getRoidb(const std::string& imdb_name);
	IMDB* combinedRoidb(const std::string& imdb_name);
	bool isValidRoidb(RoIDB& roidb);
	void filterRoidb(std::vector<RoIDB>& roidb);

	void shuffleRoidbInds();
	void getNextMiniBatch();
	void getNextMiniBatchInds(std::vector<uint32_t>& inds);
	void getMiniBatch(const std::vector<RoIDB>& roidb, const std::vector<uint32_t>& inds);
	std::vector<cv::Mat> getImageBlob(const std::vector<RoIDB>& roidb,
			const std::vector<uint32_t>& scaleInds, std::vector<float>& imScales);
	float prepImForBlob(cv::Mat& im, cv::Mat& imResized, const std::vector<float>& pixelMeans,
			const uint32_t targetSize, const uint32_t maxSize);
	void imListToBlob(std::vector<cv::Mat>& ims);


public:
	std::vector<std::vector<float>> bboxMeans;
	std::vector<std::vector<float>> bboxStds;
	IMDB* imdb;

	uint32_t numClasses;
	std::vector<float> pixelMeans;
	std::vector<uint32_t> perm;
	uint32_t cur;

	std::vector<std::vector<Data<Dtype>*>> proposalTargetDataList;

	std::vector<cv::Scalar> boxColors;



	//std::map<std::string, InputStat*> inputStatMap;
};

#endif /* ROIINPUTLAYER_H_ */
