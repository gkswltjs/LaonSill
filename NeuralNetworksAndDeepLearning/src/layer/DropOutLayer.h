/**
 * @file DropOutLayer.h
 * @date 2017-04-19
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef DROPOUTLAYER_H
#define DROPOUTLAYER_H 

#include <memory>

#include "common.h"
#include "Layer.h"
#include "LayerConfig.h"
#include "SyncMem.h"

template<typename Dtype>
class DropOutLayer : public Layer<Dtype> {
public: 
	class Builder : public Layer<Dtype>::Builder {
	public:
        double  _probability;
        double  _scale;

		Builder() {
			this->type = Layer<Dtype>::DropOut;
            this->_scale = 0.0;
            this->_probability = 0.5;
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
        Builder* probability(double probability) {
            this->_probability = probability;
            return this;
        }
        Builder* scale(double scale) {
            this->_scale = scale;
            return this;
        }
		Layer<Dtype>* build() {
			return new DropOutLayer(this);
		}
	};

	DropOutLayer(Builder* builder);
    DropOutLayer(const std::string& name);
	virtual ~DropOutLayer();

	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

protected:
	void    initialize(double scale, double probability);
    void    doDropOutForward();
    void    doDropOutBackward();

    double          scale;
    double          probability;

    std::shared_ptr<SyncMem<Dtype>>  mask;

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
#endif /* DROPOUTLAYER_H */
