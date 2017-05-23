/**
 * @file	InputLayer.h
 * @date	2016/5/11
 * @author	jhkim
 * @brief
 * @details
 */


#ifndef LAYER_INPUTLAYER_H_
#define LAYER_INPUTLAYER_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common.h"
#include "Util.h"
#include "Layer.h"
#include "DataSet.h"

/**
 * @brief 입력 레이어 클래스
 * @details 입력 데이터를 그대로 출력 데이터로 전달하는 역할을 한다.
 *          특별한 기능을 담당하지 않고 입력 데이터를 한 레벨 추상화하고
 *          약간의 레어어 쓰기, 읽기 등의 부가 기능을 수행
 *          입력 레이어의 경우 자신의 레이어값 읽기, 쓰기뿐 아니라 최초의 레이어로써 뒤에 
 *          연결된 모든 레이어의 메타 정보를 읽기, 쓰기를 수행한다.
 */
template <typename Dtype>
class InputLayer : public Layer<Dtype> {
public:
	/**
	 * @brief 입력 레이어 객체 빌더
	 * @details 입력 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 레이어 입력 객체를 생성한다.
	 */
	class Builder : public Layer<Dtype>::Builder {
	public:
		uint32_t _numTrainPack;
		uint32_t _numTestPack;
		std::vector<float> _mean;
		std::vector<uint32_t> _shape;
		std::string _source;
		std::string _sourceType;
		Dtype _scale;

		Builder() {
			this->type = Layer<Dtype>::Input;
			this->_numTrainPack = 1;
			this->_numTestPack = 1;
			this->_mean = {0.0f, 0.0f, 0.0f};
			this->_scale = Dtype(1.0);
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
		virtual Builder* numTrainPack(const uint32_t numTrainPack) {
			this->_numTrainPack = numTrainPack;
			return this;
		}
		virtual Builder* numTestPack(const uint32_t numTestPack) {
			this->_numTestPack = numTestPack;
			return this;
		}
		virtual Builder* mean(const std::vector<float>& mean) {
			this->_mean = mean;
			return this;
		}
		virtual Builder* scale(const Dtype scale) {
			this->_scale = scale;
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
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			Layer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		Layer<Dtype>* build() {
			return new InputLayer(this);
		}
	};


	/**
	 * @details InputLayer 생성자
	 * @param name 레이어의 이름 문자열
	 */
	InputLayer();
	InputLayer(const std::string name);
	InputLayer(Builder* builder);
	/**
	 * @details InputLayer 소멸자
	 */
	virtual ~InputLayer();

	//void feedforward(uint32_t idx, Data<Dtype>* input, const char* end=0);
	virtual void feedforward();
	using Layer<Dtype>::feedforward;
	virtual void feedforward(const uint32_t baseIndex, const char* end=0);

	void reshape();

    virtual int getNumTrainData();
    virtual int getNumTestData();
    virtual void shuffleTrainDataSet();


protected:
	void initialize();

public:
	DataSet<Dtype>* _dataSet;
	Data<Dtype>* _dataMean;
	Dtype _scale;

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


#endif /* LAYER_INPUTLAYER_H_ */
