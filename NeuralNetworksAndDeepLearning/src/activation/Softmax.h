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
	Softmax() : num_label(1000), batches(50) {
		this->type = ActivationType::Softmax;
		h_z = new DATATYPE[num_label*batches];
		h_activation = new DATATYPE[num_label*batches];
	}

	void activate(const DATATYPE *z, DATATYPE *activation, cudnnTensorDescriptor_t &tensorDesc) {
		float alpha = 1.0f, beta = 0.0f;
		//checkCUDNN(cudnnSoftmaxForward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
		//		&alpha, tensorDesc, z, &beta, tensorDesc, activation));

		/*
		checkCudaErrors(cudaMemcpyAsync(this->h_z, z, sizeof(DATATYPE)*num_label*batches, cudaMemcpyDeviceToHost));
		for(int batch = 0; batch < batches; batch++) {
			DATATYPE max = -1000;
			for(int label = 0; label < num_label; label++) {
				int index = batch*num_label+label;
				if(h_z[index]>max) {
					max = h_z[index];
				}
			}
			//cout << "softmax: " << max << endl;

			DATATYPE sum = 0;
			for(int label = 0; label < num_label; label++) {
				int index = batch*num_label+label;
				h_activation[index] = std::exp(h_z[index]-max);
				sum += h_activation[index];
			}

			for(int label = 0; label < num_label; label++) {
				h_activation[batch*num_label+label] /= sum;
			}
		}
		checkCudaErrors(cudaMemcpyAsync(activation, h_activation, sizeof(DATATYPE)*num_label*batches, cudaMemcpyHostToDevice));


		//Util::setPrint(true);
		Util::printData(h_z, num_label, batches, 1, 1, string("/d_z:"));
		Util::printData(h_activation, num_label, batches, 1, 1, string("/d_output:"));
		Util::setPrint(false);

		*/


		checkCUDNN(cudnnSoftmaxForward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
			&alpha, tensorDesc, z, &beta, tensorDesc, activation));


		float result;
		//cout << "softmax result sum: " << endl;
		for(int i = 0; i < batches; i++) {
			checkCudaErrors(cublasSasum(Cuda::cublasHandle, num_label, activation+i*num_label, 1, &result));
			//cout << result << ", ";
			if(result > 1.005f || result < 0.995f) {
				cout << "!!!!!!!!!!!!!!!!!!!!!softmax sum abnormal: " << result << endl;
			}
		}
		//cout << endl;



	}

	void d_activate(const DATATYPE *activation, const DATATYPE *deltaInput, const DATATYPE *z, DATATYPE *da,
				cudnnTensorDescriptor_t &tensorDesc) {
	}

	DATATYPE *h_z;						///< (임시) 활성화 입력값 확인용 호스트 메모리 포인터.
	DATATYPE *h_activation;				///< (임시) 활성화 출력값 확인용 호스트 메모리 포인터.
	const int num_label;				///< (임시) 네트워크의 레이블 수.
	const int batches;					///< (임시) 네트워크의 batch 사이즈.

#endif



};





#endif /* ACTIVATION_SOFTMAX_H_ */
