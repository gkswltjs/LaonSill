/*
 * LearnableLayer.h
 *
 *  Created on: 2016. 8. 20.
 *      Author: jhkim
 */

#ifndef LEARNABLELAYER_H_
#define LEARNABLELAYER_H_


#include "../cuda/Cuda.h"
#include "../Data.h"

/**
 * @brief 학습하는 레이어에서 구현해야하는 베이스 추상 클래스,
 *        인터페이스 역할을 한다.
 */
template <typename Dtype>
class LearnableLayer {
public:
	virtual ~LearnableLayer() {}

	virtual const string getName() = 0;




	/**
	 * @details 학습한 파라미터 그레디언트를 파라미터에 업데이트한다.
	 */
	virtual void update() = 0;
	/**
	 * @details 파라미터들의 제곱의 합을 구한다.
	 * @return 파라미터들의 제곱의 합
	 */
	virtual double sumSquareParamsData() = 0;
	/**
	 * @details 파라미터 그레디언트들의 제곱의 합을 구한다.
	 * @return 파라미터 그레디언트들의 제곱의 합
	 */
	virtual double sumSquareParamsGrad() = 0;
	/**
	 * @details 파라미터 그레디언트를 스케일링한다.
	 * @param 파라미터 그레디언트를 스케일링할 스케일 값
	 */
	virtual void scaleParamsGrad(float scale) = 0;


	//virtual double testParamAbnormality() = 0;
	virtual uint32_t boundParams() = 0;

protected:
	void _updateParam(const uint32_t paramSize, const Dtype regScale, const Dtype learnScale, Data<Dtype>* dataHistory, Data<Dtype>* data) {
		Dtype normScale = 1.0/this->in_dim.batches;
		const Dtype momentum = this->networkConfig->_momentum;
		const Dtype negativeOne = -1.0;

		data->print_grad("paramGrad:");
		data->print_data("paramData:");
		dataHistory->print_grad("paramHistoryGrad:");

		Dtype* d_paramGrad = data->mutable_device_grad();
		Dtype* d_paramData = data->mutable_device_data();
		Dtype* d_paramHistoryData = dataHistory->mutable_device_data();

		checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(paramSize), &normScale, d_paramGrad, 1));							// normalized by batch size
		checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &regScale, d_paramData, 1, d_paramGrad, 1));			// regularize
		checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(paramSize), &momentum, d_paramHistoryData, 1));					//
		checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &learnScale, d_paramGrad, 1, d_paramHistoryData, 1));	// momentum
		checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(paramSize), &negativeOne, d_paramHistoryData, 1, d_paramData, 1));	// update
	}


};



#endif /* LEARNABLELAYER_H_ */
