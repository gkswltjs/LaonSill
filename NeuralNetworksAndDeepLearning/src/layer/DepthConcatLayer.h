/**
 * @file	DepthConcatLayer.h
 * @date	2016/5/25
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_DEPTHCONCATLAYER_H_
#define LAYER_DEPTHCONCATLAYER_H_

#include "common.h"
#include "HiddenLayer.h"
#include "Exception.h"




/**
 * @brief Depth Concatenation 레이어
 * @details 복수의 입력 레이어로부터 전달된 결과값을 하나로 조합하는 레이어
 *          channel축을 기준으로 복수의 입력을 조합한다.
 *          input1: batch1-1(channel1-1-1/channel1-1-2)/batch1-2(channel1-2-1/channel1-2-2)
 *          input2: batch2-1(channel2-1-1/channel2-1-2)/batch2-2(channel2-2-1/channel2-2-2)
 *          output: batch1-1(channel1-1-1/channel1-1-2)/batch2-1(channel2-1-1/channel2-1-2)/
 *                  batch1-2(channel1-2-1/channel1-2-2)/batch2-2(channel2-2-1/channel2-2-2)
 */
template <typename Dtype>
class DepthConcatLayer : public HiddenLayer<Dtype> {
public:
	/**
	 * @brief Depth Concatenation 레이어 객체 빌더
	 * @details Depth Concatenation 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를
     *                              통해 해당 파라미터를 만족하는 Depth Concatenation 레이어
     *                              객체를 생성한다.
	 */
	class Builder : public HiddenLayer<Dtype>::Builder {
	public:
		Builder() {
			this->type = Layer<Dtype>::DepthConcat;
		}
		virtual Builder* name(const std::string name) {
			HiddenLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			HiddenLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			HiddenLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			HiddenLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			HiddenLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		Layer<Dtype>* build() {
			return new DepthConcatLayer(this);
		}
	};

	DepthConcatLayer();
	DepthConcatLayer(Builder* builder);
	DepthConcatLayer(const std::string name);
	virtual ~DepthConcatLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

protected:
	void initialize();

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

protected:
#ifndef GPU_MODE
	rcube delta_input;
	rcube delta_input_sub;
    std::vector<int> offsets;
#else
#endif
};

#endif /* LAYER_DEPTHCONCATLAYER_H_ */
