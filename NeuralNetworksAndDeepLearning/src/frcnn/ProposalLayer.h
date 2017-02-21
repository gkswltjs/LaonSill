/*
 * ProposalLayer.h
 *
 *  Created on: Nov 29, 2016
 *      Author: jkim
 */

#ifndef PROPOSALLAYER_H_
#define PROPOSALLAYER_H_

#include <vector>

#include "common.h"
#include "Layer.h"

template <typename Dtype>
class ProposalLayer : public Layer<Dtype> {
public:
	class Builder : public Layer<Dtype>::Builder {
	public:
		uint32_t _featStride;
		std::vector<uint32_t> _scales;

		Builder() {
			this->type = Layer<Dtype>::Proposal;
			this->_featStride = 16;
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
		virtual Builder* scales(const std::vector<uint32_t>& scales) {
			this->_scales = scales;
			return this;
		}
		Layer<Dtype>* build() {
			return new ProposalLayer(this);
		}
	};


	ProposalLayer();
	ProposalLayer(Builder* builder);
	virtual ~ProposalLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:
	void initialize();
	void _filterBoxes(std::vector<std::vector<float>>& boxes,
			const float minSize, std::vector<uint32_t>& keep);

private:
	uint32_t featStride;
	uint32_t numAnchors;
	std::vector<uint32_t> scales;
	std::vector<std::vector<float>> anchors;
};

#endif /* PROPOSALLAYER_H_ */
