/**
 * @file Sigmoid.h
 * @date 2016/5/10
 * @author jhkim
 * @brief
 * @details
 */



#ifndef ACTIVATION_SIGMOID_H_
#define ACTIVATION_SIGMOID_H_

#include "../common.h"
#include "Activation.h"


/**
 * @brief Sigmoid Activation 구현 클래스.
 * @details Activation 클래스를 상속받아 Sigmoid 활성화를 구현.
 */
template <typename Dtype>
class Sigmoid : public Activation<Dtype> {
public:
	virtual ~Sigmoid() {}

#ifndef GPU_MODE
public:
	Sigmoid() {
		this->type = Activation<Dtype>::Sigmoid;
	}
	/*
	void initialize_weight(int n_in, rmat &weight) {
		weight.randn();
		weight *= 1.0/sqrt(n_in);				// initial point scaling
	}
	void initialize_weight(int n_in, rcube &weight) {
		weight.randn();
		weight *= 1.0/sqrt(n_in);				// initial point scaling
	}
	*/

	/*
	void initialize_weight(int filters, void *weight, int type) {
		if(type) {
			mat *w = (mat *)weight;
			w->randn();
			(*w) *= 1 / sqrt(w->n_rows);
		}
		else {
			cube *w = (cube *)weight;
			for(int i = 0; i < filters; i++) {
				w[i].randn();
				w[i] *= 1 / sqrt(w[i]);
			}
		}
	}
	*/
	void forward(const rcube &z, rcube &activation) {
		activation = 1.0 / (1.0 + exp(-1 * z));
	}
	void d_activate(const rcube &activation, rcube &da) {
		Util::printCube(activation, "activate-activation:");
		da = activation % (1.0 - activation);
		Util::printCube(da, "activate-da:");
	}
#else
public:
	Sigmoid() {
		this->type = Activation<Dtype>::Sigmoid;
		checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
		checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));
	}

	void forward(const cudnnTensorDescriptor_t& desc, const Dtype* x, Dtype* y) {
		checkCUDNN(cudnnActivationForward(Cuda::cudnnHandle, activationDesc,
				&Cuda::alpha, desc, x, &Cuda::beta, desc, y));
	}

	void backward(const cudnnTensorDescriptor_t& desc,  const Dtype* y, const Dtype* dy, const Dtype* x, Dtype* dx) {
		checkCUDNN(cudnnActivationBackward(Cuda::cudnnHandle, activationDesc,
				&Cuda::alpha, desc, y, desc, dy, desc, x,
				&Cuda::beta, desc, dx));
	}
#endif
private:
	cudnnActivationDescriptor_t activationDesc;			///< cudnn 활성화 관련 자료구조에 대한 포인터.

};


template class Sigmoid<float>;

#endif /* ACTIVATION_SIGMOID_H_ */
