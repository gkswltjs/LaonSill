/*
 * ConcatLayer.h
 *
 *  Created on: Apr 26, 2017
 *      Author: jkim
 */

#ifndef CONCATLAYER_H_
#define CONCATLAYER_H_


#include "common.h"
#include "BaseLayer.h"

/*
 * @brief Takes at least two Datas and concatenates them along either the num
 *        or channel dimension, outputting the result.
 */
template <typename Dtype>
class ConcatLayer : public Layer<Dtype> {
public:
	/*
	class Builder : public Layer<Dtype>::Builder {
	public:
		int _axis;
		Builder() {
			this->type = Layer<Dtype>::Concat;
			this->_axis = -1;
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
		virtual Builder* axis(const int axis) {
			this->_axis = axis;
			return this;
		}
		Layer<Dtype>* build() {
			return new ConcatLayer(this);
		}
	};
	*/

	ConcatLayer();
	virtual ~ConcatLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

private:

private:
	int count;
	int numConcat;
	int concatInputSize;
	int concatAxis;

	int tempCount;


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

#endif /* CONCATLAYER_H_ */
