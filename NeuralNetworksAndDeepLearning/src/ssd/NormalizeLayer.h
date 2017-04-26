/*
 * NormalizeLayer.h
 *
 *  Created on: Apr 21, 2017
 *      Author: jkim
 */

#ifndef NORMALIZELAYER_H_
#define NORMALIZELAYER_H_

#include "common.h"
#include "Layer.h"
#include "LearnableLayer.h"

/**
 * @brief Normalizes the input to have L_p norm of 1 with scale learnable
 */
template <typename Dtype>
class NormalizeLayer : public LearnableLayer<Dtype> {
public:
	class Builder : public LearnableLayer<Dtype>::Builder {
	public:
		bool _acrossSpatial;
		bool _channelShared;
		update_param _scaleUpdateParam;
		param_filler<Dtype> _scaleFiller;
		Dtype _eps;

		Builder() {
			this->type = Layer<Dtype>::Normalize;
			this->_acrossSpatial = true;
			this->_channelShared = true;
			this->_scaleUpdateParam.lr_mult = 1.0;
			this->_scaleUpdateParam.decay_mult = 1.0;
			this->_scaleFiller.type = ParamFillerType::Constant;
			this->_scaleFiller.value = Dtype(1.0);
			this->_eps = 1.0e-10;
		}
		virtual Builder* name(const std::string name) {
			LearnableLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			LearnableLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			LearnableLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			LearnableLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			LearnableLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		virtual Builder* acrossSpatial(const bool acrossSpatial) {
			this->_acrossSpatial = acrossSpatial;
			return this;
		}
		virtual Builder* channelShared(const bool channelShared) {
			this->_channelShared = channelShared;
			return this;
		}
		virtual Builder* scaleUpdateParam(double lr_mult, double decay_mult) {
			this->_scaleUpdateParam.lr_mult = lr_mult;
			this->_scaleUpdateParam.decay_mult = decay_mult;
			return this;
		}
		virtual Builder* scaleFiller(ParamFillerType scaleFillerType, Dtype value) {
			this->_scaleFiller.type = scaleFillerType;
			this->_scaleFiller.value = value;
			return this;
		}
		virtual Builder* eps(const Dtype eps) {
			this->_eps = eps;
			return this;
		}
		Layer<Dtype>* build() {
			return new NormalizeLayer(this);
		}
	};

	NormalizeLayer(Builder* builder);
	virtual ~NormalizeLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

	virtual void update();
	void applyChanges(LearnableLayer<Dtype> *targetLayer);
	void syncParams(LearnableLayer<Dtype> *targetLayer);


protected:
	void initialize();
	void _updateParam(const uint32_t paramSize, const Dtype regScale, const Dtype learnScale,
		const Dtype epsilon, const Dtype decayRate, const Dtype beta1, const Dtype beta2,
		Data<Dtype>* dataHistory, Data<Dtype>* dataHistory2, Data<Dtype>* data);


private:
	bool acrossSpatial;
	bool channelShared;
	update_param scaleUpdateParam;
	param_filler<Dtype> scaleFiller;
	Dtype eps;

	// 적용될 norm term을 store
	// acrossSpatial인 경우 이미지 하나 전체에 대해 1개의 norm term. 이미지 갯수만큼
	// 아닌 경우 이미지 하나에 대해 채널간 norm term, spatialDim만큼의 norm term
	// norm 자체는 채널간 합, spatial 단위로 통합하느냐 여부에 따라 결정
	Data<Dtype> norm_;
	// 각 spatialDim 단위로 channel간 sum하기 위해 1로 채워진 vector
	Data<Dtype> sumChannelMultiplier_;
	Data<Dtype> sumSpatialMultiplier_;

	// 1장의 이미지 각 element에 대한 처리 결과를 담기 위한 buffer
	Data<Dtype> buffer_;
	// 이미지당 channel별 scalar를 저장하기 위한 buffer
	Data<Dtype> bufferChannel_;
	// 이미지당 spatialDim별 scalar를 저장하기 위한 buffer
	Data<Dtype> bufferSpatial_;
};

#endif /* NORMALIZELAYER_H_ */
