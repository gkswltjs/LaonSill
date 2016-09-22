/**
 * @file	DepthConcatLayer.h
 * @date	2016/5/25
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_DEPTHCONCATLAYER_H_
#define LAYER_DEPTHCONCATLAYER_H_

#include "HiddenLayer.h"
#include "../exception/Exception.h"




/**
 * @brief Depth Concatenation 레이어
 * @details 복수의 입력 레이어로부터 전달된 결과값을 하나로 조합하는 레이어
 *          channel축을 기준으로 복수의 입력을 조합한다.
 *          input1: batch1-1(channel1-1-1/channel1-1-2)/batch1-2(channel1-2-1/channel1-2-2)
 *          input2: batch2-1(channel2-1-1/channel2-1-2)/batch2-2(channel2-2-1/channel2-2-2)
 *          output: batch1-1(channel1-1-1/channel1-1-2)/batch2-1(channel2-1-1/channel2-1-2)/batch1-2(channel1-2-1/channel1-2-2)/batch2-2(channel2-2-1/channel2-2-2)
 */
template <typename Dtype>
class DepthConcatLayer : public HiddenLayer<Dtype> {
public:
	/**
	 * @brief Depth Concatenation 레이어 객체 빌더
	 * @details Depth Concatenation 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 Depth Concatenation 레이어 객체를 생성한다.
	 */
	class Builder : public HiddenLayer<Dtype>::Builder {
	public:
		Builder() {
			this->type = Layer<Dtype>::DepthConcat;
		}
		virtual Builder* name(const string name) {
			HiddenLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			HiddenLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* nextLayerIndices(const vector<uint32_t>& nextLayerIndices) {
			HiddenLayer<Dtype>::Builder::nextLayerIndices(nextLayerIndices);
			return this;
		}
		virtual Builder* prevLayerIndices(const vector<uint32_t>& prevLayerIndices) {
			HiddenLayer<Dtype>::Builder::prevLayerIndices(prevLayerIndices);
			return this;
		}
		Layer<Dtype>* build() {
			return new DepthConcatLayer(this);
		}
		virtual void save(ofstream& ofs) {
			HiddenLayer<Dtype>::Builder::save(ofs);
		}
		virtual void load(ifstream& ifs) {
			HiddenLayer<Dtype>::Builder::load(ifs);
		}
	};

	DepthConcatLayer();
	DepthConcatLayer(Builder* builder);
	DepthConcatLayer(const string name);
	virtual ~DepthConcatLayer();

	virtual void shape(uint32_t idx, io_dim in_dim, shared_ptr<Data<Dtype>>& prevLayerOutput);
	virtual void reshape(uint32_t idx, io_dim in_dim);

protected:
	void initialize();

	virtual void _shape(bool recursive=true);
	virtual void _clearShape();
	//virtual void _load(ifstream &ifs, map<Layer<Dtype>*, Layer<Dtype>*> &layerMap);

	/**
	 * @details 일반적인 concat과 달리 channel을 기준으로 조합하므로 재정의한다.
	 */
	virtual void _concat(uint32_t idx, Data<Dtype>* input);
	/**
	 * @details 일반적인 deconcat과 달리 channel을 기준으로 해체하므로 재정의한다.
	 */
	//virtual void _deconcat(uint32_t idx, Data<Dtype>* next_delta_input, uint32_t offset);

	virtual void _backpropagation();


	/**
	 * @details _concat()에서 입력값이 합산되는 방식이 아니므로 합산에 대해
	 *          scaling을 적용하는 기본 _scaleInput()을 재정의하여 scale하지 않도록 한다.
	 */
	virtual void _scaleInput() {};
	/**
	 * @details _deconcat()에서 gradient값이 합산되는 방식이 아니므로 합산에 대해
	 *          scaling을 적용하는 기본 대_scaleGradient()를 재정의하여 scale하지 않도록 한다.
	 */
	virtual void _scaleGradient() {};
	/**
	 * @details 입력 그레디언트 전체를 이전 레이어에 전달하는 것이 아니라
	 *          이전 레이어의 출력 사이즈에 따라 입력 그레디언트를 나누어줘야 하므로
	 *          propBackpropagation()을 재정의한다.
	 */
	virtual void propBackpropagation();



protected:
#ifndef GPU_MODE
	rcube delta_input;
	rcube delta_input_sub;
	vector<int> offsets;
#else
	int offsetIndex;			///< 입력에 관한 gradient 호출한 횟수 카운터, getDeltaInput() 호출마다 증가되고 feedforward()가 수행될 때 reset된다.
#endif
};



#endif /* LAYER_DEPTHCONCATLAYER_H_ */
