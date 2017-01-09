/**
 * @file	HiddenLayer.h
 * @date	2016/5/11
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_HIDDENLAYER_H_
#define LAYER_HIDDENLAYER_H_

#include "common.h"
#include "Layer.h"

/**
 * @brief 히든 레이어 기본 추상 클래스
 * @details 기본 레이어의 클래스에 backpropagation, parameter update와 같은
 *          파라미터 학습 관련 기능을 추가한다.
 */
template <typename Dtype>
class HiddenLayer : public Layer<Dtype> {
public:
	/**
	 * @brief 히든 레이어 객체 빌더
	 * @details 히든 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 히든 레이어 객체를 생성한다.
	 */
	class Builder : public Layer<Dtype>::Builder {
	public:
        std::vector<uint32_t> _prevLayerIndices;

		Builder() {}
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
		Layer<Dtype>* build() = 0;
	};

	HiddenLayer();
	HiddenLayer(Builder* builder);
	HiddenLayer(const std::string& name);
	virtual ~HiddenLayer();

	/**
	 * @details 네트워크 cost의 다음 레이어의 입력에 관한 gradient값을 전달 받아
	 *          현재 레이어의 parameter(parameter가 있는 경우), input에 관한 gradient를 
     *          계산하고 이전 레이어에 현재 레이어의 input에 관한 gradient값을 전달한다.
	 */
	virtual void backpropagation();
	virtual void reshape();
};

#endif /* LAYER_HIDDENLAYER_H_ */
