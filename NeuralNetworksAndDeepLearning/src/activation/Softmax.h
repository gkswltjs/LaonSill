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
	Softmax() : num_label(100), batches(10) {
		this->type = ActivationType::Softmax;
		h_z = new DATATYPE[num_label*batches];
		h_activation = new DATATYPE[num_label*batches];
	}

	void activate(const DATATYPE *z, DATATYPE *activation, cudnnTensorDescriptor_t &tensorDesc) {
		float alpha = 1.0f, beta = 0.0f;
		//checkCUDNN(cudnnSoftmaxForward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
		//		&alpha, tensorDesc, z, &beta, tensorDesc, activation));


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



		//checkCUDNN(cudnnSoftmaxForward(Cuda::cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
		//	&alpha, tensorDesc, z, &beta, tensorDesc, activation));
	}

	void d_activate(const DATATYPE *activation, const DATATYPE *deltaInput, const DATATYPE *z, DATATYPE *da,
				cudnnTensorDescriptor_t &tensorDesc) {
	}

	DATATYPE *h_z;
	DATATYPE *h_activation;
	const int num_label;
	const int batches;

#endif



};





#endif /* ACTIVATION_SOFTMAX_H_ */
