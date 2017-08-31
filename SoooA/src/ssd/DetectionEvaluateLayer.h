/*
 * DetectionEvaluateLayer.h
 *
 *  Created on: May 15, 2017
 *      Author: jkim
 */

#ifndef DETECTIONEVALUATELAYER_H_
#define DETECTIONEVALUATELAYER_H_

#include "common.h"
#include "BaseLayer.h"
#include "ssd_common.h"

/*
 * @brief Generate the detection evaluation based on DetectionOutputLayer and
 * ground truth bounding box labels.
 *
 * Intended for use with MultiBox detection method.
 */
template <typename Dtype>
class DetectionEvaluateLayer : public Layer<Dtype> {
public:
	/*
	class Builder : public Layer<Dtype>::Builder {
	public:
		int _numClasses;
		int _backgroundLabelId;
		float _overlapThreshold;
		bool _evaluateDifficultGt;
		std::string _nameSizeFile;

		Builder() {
			this->type = Layer<Dtype>::Concat;
			this->_numClasses = -1;
			this->_backgroundLabelId = 0;
			this->_overlapThreshold = 0.5f;
			this->_evaluateDifficultGt = true;
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
		virtual Builder* numClasses(const int numClasses) {
			this->_numClasses = numClasses;
			return this;
		}
		virtual Builder* backgroundLabelId(const int backgroundLabelId) {
			this->_backgroundLabelId = backgroundLabelId;
			return this;
		}
		virtual Builder* overlapThreshold(const float overlapThreshold) {
			this->_overlapThreshold = overlapThreshold;
			return this;
		}
		virtual Builder* evaluateDifficultGt(const bool evaluateDifficultGt) {
			this->_evaluateDifficultGt = evaluateDifficultGt;
			return this;
		}
		virtual Builder* nameSizeFile(const std::string& nameSizeFile) {
			this->_nameSizeFile = nameSizeFile;
			return this;
		}
		Layer<Dtype>* build() {
			return new DetectionEvaluateLayer(this);
		}
	};
	*/

	DetectionEvaluateLayer();
	virtual ~DetectionEvaluateLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:


private:
	/*
	int numClasses;
	int backgroundLabelId;
	float overlapThreshold;
	bool evaluateDifficultGt;
	std::string nameSizeFile;
	*/

	std::vector<std::pair<int, int>> sizes;
	int count;
	bool useNormalizedBBox;








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

#endif /* DETECTIONEVALUATELAYER_H_ */
