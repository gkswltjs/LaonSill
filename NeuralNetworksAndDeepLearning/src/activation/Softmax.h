/**
 * @file Softmax.h
 * @date 2016/5/12
 * @author jhkim
 * @brief
 * @details
 */

#ifndef ACTIVATION_SOFTMAX_H_
#define ACTIVATION_SOFTMAX_H_

#include "../common.h"
#include "Activation.h"

/**
 * @brief Softmax Activation 구현 클래스.
 * @details Activation 클래스를 상속받아 Softmax 활성화를 구현.
 */
template <typename Dtype>
class Softmax : public Activation<Dtype> {
public:
	virtual ~Softmax() {}

#ifndef GPU_MODE
public:
	Softmax() {
		this->type = Activation::Softmax;
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

	void forward(const rcube &z, rcube &activation) {
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
		this->type = Activation<Dtype>::Softmax;
	}

	void forward(const cudnnTensorDescriptor_t& desc, const Dtype* x, Dtype* y) {
		checkCUDNN(cudnnSoftmaxForward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
				&Cuda::alpha, desc, x,
				&Cuda::beta, desc, y));
	}

	void backward(const cudnnTensorDescriptor_t& desc,  const Dtype* y, const Dtype* dy, const Dtype* x, Dtype* dx) {
		checkCUDNN(cudnnSoftmaxBackward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
				&Cuda::alpha, desc, y, desc, dy,
				&Cuda::beta, desc, dx));
	}
#endif

};


template class Softmax<float>;


#endif /* ACTIVATION_SOFTMAX_H_ */
