/*
 * RoIInputLayer.h
 *
 *  Created on: Nov 11, 2016
 *      Author: jkim
 */

#ifndef ROIINPUTLAYER_H_
#define ROIINPUTLAYER_H_

#include "frcnn_common.h"
#include "DataSet.h"
#include "InputLayer.h"
#include "IMDB.h"

template <typename Dtype>
class RoIInputLayer : public InputLayer<Dtype> {
public:
	/**
	 * @brief 입력 레이어 객체 빌더
	 * @details 입력 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 레이어 입력 객체를 생성한다.
	 */
	class Builder : public Layer<Dtype>::Builder {
	public:
		uint32_t _numClasses;

		Builder() {
			this->type = Layer<Dtype>::Input;
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
		virtual Builder* numClasses(const uint32_t numClasses) {
			this->_numClasses = numClasses;
			return this;
		}
		Layer<Dtype>* build() {
			return new RoIInputLayer(this);
		}
		virtual void save(std::ofstream& ofs) {
			Layer<Dtype>::Builder::save(ofs);
		}
		virtual void load(std::ifstream& ifs) {
			Layer<Dtype>::Builder::load(ifs);
		}
	};

	RoIInputLayer();
	RoIInputLayer(Builder* builder);
	virtual ~RoIInputLayer();


	void feedforward();
	using Layer<Dtype>::feedforward;
	void feedforward(const uint32_t baseIndex, const char* end=0);
	void shape();

private:
	void initialize();

	IMDB* getImdb(const std::string& imdb_name);
	void getTrainingRoidb(IMDB* imdb);
	IMDB* getRoidb(const std::string& imdb_name);
	IMDB* combinedRoidb(const std::string& imdb_name);
	bool isValidRoidb(RoIDB& roidb);
	void filterRoidb(std::vector<RoIDB>& roidb);


public:
	//DataSet<Dtype>* _dataSet;
	std::vector<std::vector<float>> bboxMeans;
	std::vector<std::vector<float>> bboxStds;
	IMDB* imdb;

};

#endif /* ROIINPUTLAYER_H_ */
