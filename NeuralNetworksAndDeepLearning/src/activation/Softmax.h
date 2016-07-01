/*
 * Softmax.h
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef ACTIVATION_SOFTMAX_H_
#define ACTIVATION_SOFTMAX_H_

#include "Activation.h"







class Softmax : public Activation {
public:
	virtual ~Softmax() {}

#if CPU_MODE
public:
	Softmax() {
		this->type = ActivationType::Softmax;
	}


	/*
	void initialize_weight(int n_in, rmat &weight) {
		// TODO for debug
		//weight.randn();
		//weight *= sqrt(1.0/n_in);
		weight.zeros();
	}
	void initialize_weight(int n_in, rcube &weight) {
		// TODO for debug
		//weight.randn();
		//weight *= sqrt(1.0/n_in);
		weight.zeros();
	}
	*/

	void activate(const rcube &z, rcube &activation) {
		// TODO softmax는 output layer only,
		// vector형태의 output을 전제
		//Util::printCube(z, "z:");
		//cout << "maxz: " << max(max(z.slice(0)));
		rcube temp = exp(z-max(max(z.slice(0))));
		//Util::printCube(temp, "temp:");
		activation = temp / accu(temp);
		//Util::printCube(activation, "activation:");
	}
	void d_activate(const rcube &activation, rcube &da) {
		//Util::printCube(activation, "activation:");
		da = activation % (1.0 - activation);
		//Util::printCube(da, "da:");
	}

#else
	Softmax() {
		this->type = ActivationType::Softmax;
	}

	void activate(const DATATYPE *z, DATATYPE *activation, cudnnTensorDescriptor_t &tensorDesc) {
		float alpha = 1.0f, beta = 0.0f;
		//checkCUDNN(cudnnSoftmaxForward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
		//		&alpha, tensorDesc, z, &beta, tensorDesc, activation));
		checkCUDNN(cudnnSoftmaxForward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
					&alpha, tensorDesc, z, &beta, tensorDesc, activation));
	}

	void d_activate(const DATATYPE *activation, const DATATYPE *deltaInput, const DATATYPE *z, DATATYPE *da,
				cudnnTensorDescriptor_t &tensorDesc) {
	}
#endif

};





#endif /* ACTIVATION_SOFTMAX_H_ */
