/*
 * Sigmoid.h
 *
 *  Created on: 2016. 5. 10.
 *      Author: jhkim
 */

#ifndef ACTIVATION_SIGMOID_H_
#define ACTIVATION_SIGMOID_H_

#include "Activation.h"



class Sigmoid : public Activation {
public:
	virtual ~Sigmoid() {}

#if CPU_MODE
public:
	Sigmoid() {
		this->type = ActivationType::Sigmoid;
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
	void activate(const rcube &z, rcube &activation) {
		activation = 1.0 / (1.0 + exp(-1 * z));
	}
	void d_activate(const rcube &activation, rcube &da) {
		Util::printCube(activation, "d_activate-activation:");
		da = activation % (1.0 - activation);
		Util::printCube(da, "d_activate-da:");
	}
#else
public:
	Sigmoid() {
		this->type = ActivationType::Sigmoid;
		checkCUDNN(cudnnCreateActivationDescriptor(&activationDesc));
		checkCUDNN(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));
	}
	void activate(const DATATYPE *z, DATATYPE *activation, cudnnTensorDescriptor_t &tensorDesc) {
		float alpha = 1.0f, beta = 0.0f;
		checkCUDNN(cudnnActivationForward(Cuda::cudnnHandle, activationDesc, &alpha,
				tensorDesc, z, &beta, tensorDesc, activation));
	}
	void d_activate(const DATATYPE *activation, const DATATYPE *deltaInput, const DATATYPE *z, DATATYPE *da,
				cudnnTensorDescriptor_t &tensorDesc) {
		float alpha = 1.0f, beta = 0.0f;
		checkCUDNN(cudnnActivationBackward(Cuda::cudnnHandle, activationDesc, &alpha,
				tensorDesc, activation, tensorDesc, deltaInput,
				tensorDesc, z, &beta, tensorDesc, da));
	}
#endif
private:
	cudnnActivationDescriptor_t activationDesc;

};

#endif /* ACTIVATION_SIGMOID_H_ */
