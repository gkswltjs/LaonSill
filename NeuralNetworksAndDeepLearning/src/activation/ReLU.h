/**
 * @file ReLU.h
 * @date 2016/5/18
 * @author jhkim
 * @brief
 * @details
 */


#ifndef ACTIVATION_RELU_H_
#define ACTIVATION_RELU_H_

#include "../common.h"
#include "Activation.h"


/**
 * @brief ReLU(Rectified Linear Unit) Activation 구현 클래스.
 * @details Activation 클래스를 상속받아 ReLU 활성화를 구현.
 */
template <typename Dtype>
class ReLU : public Activation<Dtype> {
public:
	virtual ~ReLU() {}

#ifndef GPU_MODE
public:
	ReLU() {
		this->type = Activation<Dtype>::ReLU;
	}
	ReLU(io_dim activation_dim) {
		this->type = Activation<Dtype>::TypeReLU;
		zero.set_size(activation_dim.rows, activation_dim.cols, activation_dim.channels);
		zero.zeros();
	}
	/*
	void initialize_weight(int n_in, rmat &weight) {
		weight.randn();
		weight *= sqrt(2.0/n_in);				// initial point scaling
	}
	void initialize_weight(int n_in, rcube &weight) {
		weight.randn();
		weight *= sqrt(2.0/n_in);				// initial point scaling
	}
	*/
	void forward(const rcube &z, rcube &activation) {
		if(zero.n_elem <= 1) {
			zero.set_size(size(z));
			zero.zeros();
		}
		activation = arma::max(z, zero);
	}
	void d_activate(const rcube &activation, rcube &da) {
		Util::printCube(activation, "activate-activation:");
		da = conv_to<rcube>::from(activation > zero);
		Util::printCube(da, "activate-da:");
	}

private:
	rcube zero;

#else
public:
	ReLU() {
		this->type = Activation<Dtype>::ReLU;
		checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
		checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
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

private:
	cudnnActivationDescriptor_t activationDesc;			///< cudnn 활성화 관련 자료구조에 대한 포인터.

#endif

};


template class ReLU<float>;



#endif /* ACTIVATION_RELU_H_ */
