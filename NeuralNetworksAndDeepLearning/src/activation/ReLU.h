/**
 * @file ReLU.h
 * @date 2016/5/18
 * @author jhkim
 * @brief
 * @details
 */


#ifndef ACTIVATION_RELU_H_
#define ACTIVATION_RELU_H_

#include "Activation.h"



/**
 * @brief ReLU(Rectified Linear Unit) Activation 구현 클래스.
 * @details Activation 클래스를 상속받아 ReLU 활성화를 구현.
 */
class ReLU : public Activation {
public:
	virtual ~ReLU() {}

#ifndef GPU_MODE
public:
	ReLU() {
		this->type = ActivationType::ReLU;
	}
	ReLU(io_dim activation_dim) {
		this->type = ActivationType::ReLU;
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
	void activate(const rcube &z, rcube &activation) {
		if(zero.n_elem <= 1) {
			zero.set_size(size(z));
			zero.zeros();
		}
		activation = arma::max(z, zero);
	}
	void d_activate(const rcube &activation, rcube &da) {
		Util::printCube(activation, "d_activate-activation:");
		da = conv_to<rcube>::from(activation > zero);
		Util::printCube(da, "d_activate-da:");
	}

private:
	rcube zero;

#else
public:
	ReLU() {
		this->type = ActivationType::ReLU;
		checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
		checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
	}
	void activate(const DATATYPE *z, DATATYPE *activation, cudnnTensorDescriptor_t &tensorDesc) {
		//float alpha = 1.0f, beta = 0.0f;
		checkCUDNN(cudnnActivationForward(Cuda::cudnnHandle, activationDesc, &Cuda::alpha,
					tensorDesc, z, &Cuda::beta, tensorDesc, activation));
	}
	void d_activate(const DATATYPE *activation, const DATATYPE *deltaInput, const DATATYPE *z, DATATYPE *da,
			cudnnTensorDescriptor_t &tensorDesc) {
		//float alpha = 1.0f, beta = 0.0f;
		checkCUDNN(cudnnActivationBackward(Cuda::cudnnHandle, activationDesc, &Cuda::alpha,
				tensorDesc, activation, tensorDesc, deltaInput,
				tensorDesc, z, &Cuda::beta, tensorDesc, da));
	}

private:
	cudnnActivationDescriptor_t activationDesc;			///< cudnn 활성화 관련 자료구조에 대한 포인터.

#endif

};






#endif /* ACTIVATION_RELU_H_ */
