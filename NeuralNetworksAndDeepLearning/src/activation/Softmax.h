/**
 * @file Softmax.h
 * @date 2016/5/12
 * @author jhkim
 * @brief
 * @details
 */

#ifndef ACTIVATION_SOFTMAX_H_
#define ACTIVATION_SOFTMAX_H_

#include "Activation.h"






/**
 * @brief Softmax Activation 구현 클래스.
 * @details Activation 클래스를 상속받아 Softmax 활성화를 구현.
 */
class Softmax : public Activation {
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
	Softmax() {// : num_label(10), batches(10) {
		this->type = Activation::Softmax;
		//h_z = new DATATYPE[num_label*batches];
		//h_activation = new DATATYPE[num_label*batches];
	}

	/*
	void forward(const DATATYPE *z, DATATYPE *activation, cudnnTensorDescriptor_t &tensorDesc) {
		checkCUDNN(cudnnSoftmaxForward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
			&Cuda::alpha, tensorDesc, z, &Cuda::beta, tensorDesc, activation));
	}
	*/

	void forward(const cudnnTensorDescriptor_t& desc, const DATATYPE* x, DATATYPE* y) {
		checkCUDNN(cudnnSoftmaxForward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
				&Cuda::alpha, desc, x, &Cuda::beta, desc, y));
	}

	/*
	void backward(const DATATYPE *activation, const DATATYPE *deltaInput, const DATATYPE *z, DATATYPE *da,
				cudnnTensorDescriptor_t &tensorDesc) {

		checkCUDNN(cudnnSoftmaxBackward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
				&Cuda::alpha, ))

	}
	*/

	void backward(const cudnnTensorDescriptor_t& desc,  const DATATYPE *y, const DATATYPE *dy, const DATATYPE *x, DATATYPE *dx) {
		checkCUDNN(cudnnSoftmaxBackward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
				&Cuda::alpha, desc, y, desc, dy,
				&Cuda::beta, desc, dx));
	}

	//DATATYPE *h_z;						///< (임시) 활성화 입력값 확인용 호스트 메모리 포인터.
	//DATATYPE *h_activation;				///< (임시) 활성화 출력값 확인용 호스트 메모리 포인터.
	//const int num_label;				///< (임시) 네트워크의 레이블 수.
	//const int batches;					///< (임시) 네트워크의 batch 사이즈.

#endif



};





#endif /* ACTIVATION_SOFTMAX_H_ */
