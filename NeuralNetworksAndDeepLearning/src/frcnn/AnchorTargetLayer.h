/*
 * AnchorTargetLayer.h
 *
 *  Created on: Nov 18, 2016
 *      Author: jkim
 */

#ifndef ANCHORTARGETLAYER_H_
#define ANCHORTARGETLAYER_H_




#include "common.h"
#include "Layer.h"


/**
 * Assign anchors to ground-truth targets. Produces anchor classification
 * labels and bounding-box regression targets.
 * 실제 input data에 대한 cls score, bbox pred를 계산, loss를 계산할 때 쓸 데이터를 생성한다.
 */
template <typename Dtype>
class AnchorTargetLayer : public Layer<Dtype> {
public:
	class Builder : public Layer<Dtype>::Builder {
	public:
		uint32_t _featStride;
		uint32_t _allowedBorder;
		std::vector<uint32_t> _scales;

		Builder() {
			this->type = Layer<Dtype>::AnchorTarget;
			this->_featStride = 16;
			this->_allowedBorder = 0;
			this->_scales = {8, 16, 32};
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
		virtual Builder* featStride(const uint32_t featStride) {
			this->_featStride = featStride;
			return this;
		}
		virtual Builder* allowedBorder(const uint32_t allowedBorder) {
			this->_allowedBorder = allowedBorder;
			return this;
		}
		virtual Builder* scales(const std::vector<uint32_t>& scales) {
			this->_scales = scales;
			return this;
		}
		Layer<Dtype>* build() {
			return new AnchorTargetLayer(this);
		}
	};

	AnchorTargetLayer(Builder* builder);
	virtual ~AnchorTargetLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();


protected:
	void initialize();


private:
	void _computeTargets(const std::vector<std::vector<float>>& exRois,
			const std::vector<std::vector<float>>& gtRois,
			std::vector<std::vector<float>>& bboxTargets);

	void _unmap(const std::vector<int>& data, const uint32_t count,
			const std::vector<uint32_t>& indsInside, const int fill,
			std::vector<int>& result);
	void _unmap(const std::vector<std::vector<float>>& data, const uint32_t count,
			const std::vector<uint32_t>& indsInside,
			std::vector<std::vector<float>>& result);


protected:
	uint32_t featStride;
	uint32_t allowedBorder;
	uint32_t numAnchors;
	std::vector<uint32_t> scales;
	std::vector<std::vector<float>> anchors;
};

#endif /* ANCHORTARGETLAYER_H_ */
































