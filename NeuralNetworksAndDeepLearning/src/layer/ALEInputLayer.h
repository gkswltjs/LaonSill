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
class ALEInputLayer : public InputLayer<Dtype> {
public:
	class Builder : public InputLayer<Dtype>::Builder {
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
	};

	ALEInputLayer(Builder* builder);
    ALEInputLayer(const std::string& name);

    virtual ~ALEInputLayer();

	void feedforward();
	using Layer<Dtype>::feedforward;
	void feedforward(const uint32_t baseIndex, const char* end=0);

    int getNumTrainData();
    int getNumTestData();
    void shuffleTrainDataSet();

	void reshape();

protected:
	void initialize();

public:
    void setInputCount(int rows, int cols, int channels, int actionCnt);

    int                     rowCnt;     // scaled row count of ALE screen
    int                     colCnt;     // scaled column count of ALE screen
    int                     chCnt;      // channel count of ALE screen
    int                     actionCnt;

private:
    Dtype                  *preparedData;
    Dtype                  *preparedLabel;
    int                     allocBatchSize;

public:
    void allocInputData();
    void fillData(DQNImageLearner<Dtype> *learner, bool useState1);
    void fillLabel(DQNImageLearner<Dtype> *learner);

public:
    /****************************************************************************
     * layer callback functions 
     ****************************************************************************/
    static void* initLayer();
    static void destroyLayer(void* instancePtr);
    static void setInOutTensor(void* instancePtr, void* tensorPtr, bool isInput, int index);
    static bool allocLayerTensors(void* instancePtr);
    static void forwardTensor(void* instancePtr, int miniBatchIndex);
    static void backwardTensor(void* instancePtr);
    static void learnTensor(void* instancePtr);
};

#endif /* ALEINPUTLAYER_H */
