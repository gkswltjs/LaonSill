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
class DepthConcatLayer : public HiddenLayer {
public:
	class Builder : public HiddenLayer::Builder {
	public:
		Builder() {}
		virtual Builder* name(const string name) {
			HiddenLayer::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			HiddenLayer::Builder::id(id);
			return this;
		}
		virtual Builder* nextLayerIndices(const vector<uint32_t>& nextLayerIndices) {
			HiddenLayer::Builder::nextLayerIndices(nextLayerIndices);
			return this;
		}
		virtual Builder* prevLayerIndices(const vector<uint32_t>& prevLayerIndices) {
			HiddenLayer::Builder::prevLayerIndices(prevLayerIndices);
			return this;
		}
		Layer* build() {
			return new DepthConcatLayer(this);
		}
	};

	DepthConcatLayer();
	DepthConcatLayer(Builder* builder);
	DepthConcatLayer(const string name);
#ifndef GPU_MODE
	DepthConcatLayer(const string name, int n_in);
#endif
	virtual ~DepthConcatLayer();

#ifndef GPU_MODE
	//rcube &getDeltaInput();
#else
	/**
	 * @details 조합되어있는 입력에 관한 gradient를 getDeltaInput의 호출 순서에 따라 다시 deconcatenation하여 조회한다.
	 *          getDeltaInput() 호출 순서에 의존성이 있기 때문에 조합한 순서대로 다시 호출해야하므로 호출 순서를 변경할 경우 주의해야 한다.
	 *          feedforward()를 다시 수행한 경우 해당 순서가 reset된다.
	 * @return getDeltaInput() 순서에 따라 deconcate된 d_delta_input 장치 메모리 포인터
	 */
	//DATATYPE *getDeltaInput();
#endif

	virtual void shape(UINT idx, io_dim in_dim);
	virtual void reshape(UINT idx, io_dim in_dim);

protected:
	void initialize();

	virtual void _shape(bool recursive=true);
	virtual void _clearShape();
	virtual void _load(ifstream &ifs, map<Layer *, Layer *> &layerMap);

#ifndef GPU_MODE
	virtual void _feedforward();
	void reset_nabla(UINT idx) {
		if(!isLastPrevLayerRequest(idx)) return;
		propResetNParam();
	}
#else
	/**
	 * @details 일반적인 concat과 달리 channel을 기준으로 조합하므로 재정의한다.
	 */
	virtual void _concat(UINT idx, Data* input);
	/**
	 * @details 일반적인 deconcat과 달리 channel을 기준으로 해체하므로 재정의한다.
	 */
	virtual void _deconcat(UINT idx, Data* next_delta_input, uint32_t offset);
	/**
	 * @details _concat()에서 입력값이 합산되는 방식이 아니므로 합산에 대해
	 *          scaling을 적용하는 기본 _scaleInput()을 재정의하여 scale하지 않도록 한다.
	 */
	virtual void _scaleInput();
	/**
	 * @details _deconcat()에서 gradient값이 합산되는 방식이 아니므로 합산에 대해
	 *          scaling을 적용하는 기본 대_scaleGradient()를 재정의하여 scale하지 않도록 한다.
	 */
	virtual void _scaleGradient();

	virtual void propBackpropagation();





#endif


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
