/**
 * @file ALEInputLayer.h
 * @date 2016-12-09
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef ALEINPUTLAYER_H
#define ALEINPUTLAYER_H 

#include <map>
#include <string>
#include <vector>

#include "common.h"
#include "Layer.h"
#include "DQNState.h"
#include "DQNTransition.h"
#include "DQNImageLearner.h"

template<typename Dtype> class DQNImageLearner;

template <typename Dtype>
class ALEInputLayer : public Layer<Dtype> {
public:
	class Builder : public Layer<Dtype>::Builder {
	public:
		std::vector<uint32_t> _shape;
		std::string _source;
		std::string _sourceType;

		Builder() {
			this->type = Layer<Dtype>::Input;
		}
		virtual Builder* shape(const std::vector<uint32_t>& shape) {
			this->_shape = shape;
			return this;
		}
		virtual Builder* source(const std::string& source) {
			this->_source = source;
			return this;
		}
		virtual Builder* sourceType(const std::string& sourceType) {
			this->_sourceType = sourceType;
			return this;
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
		Layer<Dtype>* build() {
			return new ALEInputLayer(this);
		}
		virtual void save(std::ofstream& ofs) {
			Layer<Dtype>::Builder::save(ofs);
		}
		virtual void load(std::ifstream& ifs) {
			Layer<Dtype>::Builder::load(ifs);
		}
	};

    ALEInputLayer();
	ALEInputLayer(const std::string name);
	ALEInputLayer(Builder* builder);

    virtual ~ALEInputLayer();

	int getInputSize() const;

	void feedforward();
	using Layer<Dtype>::feedforward;
	void feedforward(const uint32_t baseIndex, const char* end=0);

	void shape();

protected:
	void initialize();
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();

public:
    void setInputCount(int rows, int cols, int channels);

    int                     rowCnt;     // scaled row count of ALE screen
    int                     colCnt;     // scaled column count of ALE screen
    int                     chCnt;      // channel count of ALE screen

private:
    Dtype                  *preparedData;
    Dtype                  *preparedLabel;
    int                     allocBatchSize;

public:
    void allocInputData();
    void fillInputData(DQNImageLearner<Dtype> *dqnImage);
};

#endif /* ALEINPUTLAYER_H */
