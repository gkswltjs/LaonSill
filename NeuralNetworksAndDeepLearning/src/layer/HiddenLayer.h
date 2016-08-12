/**
 * @file	HiddenLayer.h
 * @date	2016/5/11
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_HIDDENLAYER_H_
#define LAYER_HIDDENLAYER_H_

#include "Layer.h"
#ifndef GPU_MODE
#include <armadillo>
#endif


#ifndef GPU_MODE
using namespace arma;
#endif



/**
 * @brief 히든 레이어 기본 추상 클래스
 * @details 기본 레이어의 클래스에 backpropagation, parameter update와 같은
 *          파라미터 학습 관련 기능을 추가한다.
 */
class HiddenLayer : public Layer {
public:
	HiddenLayer() {}
	HiddenLayer(const string name) : Layer(name) {}
#ifndef GPU_MODE
	HiddenLayer(const string name, int n_in, int n_out) : Layer(name, n_in, n_out) {}
	virtual ~HiddenLayer() {}
#else
	virtual ~HiddenLayer() {
		checkCudaErrors(cudaFree(d_delta_input));
		checkCudaErrors(cudaFree(d_delta_output));
	}
#endif

#ifndef GPU_MODE
	virtual rcube &getDeltaInput()=0;

	/**
	 * @details CPU MODE에서 누적된 학습 parameter값들을 reset한다.
	 *          현재 CPU_MODE의 경우 batch 단위의 학습을 한번에 수행하지 않고
	 *          각 데이터 별로 학습, 학습결과를 누적하여 batch 단위만큼 학습된 경우
	 *          누적된 학습결과를 갱신한다. 이 때 누적된 학습결과를 다음 batch 학습을 위해 reset하는 기능을 한다.
	 *          (GPU 연산에서는 현재 아무일도 하지 않고 있음)
	 * @param idx 현재 레이어에 연결된 다음 레이어의 순번 index
	 */
	virtual void reset_nabla(UINT idx)=0;
#else
	/**
	 * @details 네트워크 cost의 현재 레이어 입력에 관한 gradient 장치 메모리 포인터를 조회한다.
	 * @return 네트워크 cost의 현재 레이어 입력에 관한 gradient 장치 메모리 포인터
	 */
	virtual DATATYPE *getDeltaInput()=0;
	/**
	 * @details 네트워크 cost의 현재 레이어 입력에 관한 gradient값을 특정 값으로 설정한다.
	 *          deepdream application 수행을 위해 임시로 만들었다.
	 * @param delta_input 수정할 네트워크 cost의 현재 레이어 입력에 관한 gradient값
	 */
	void setDeltaInput(DATATYPE *delta_input) {
		checkCudaErrors(cudaMemcpyAsync(d_delta_input, delta_input, sizeof(DATATYPE)*in_dim.batchsize(), cudaMemcpyDeviceToDevice));
		Util::printDeviceData(delta_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "delta_input:");
		Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.cols, in_dim.channels, in_dim.batches, "d_delta_input:");
	}
#endif



	/**
	 * @details 네트워크 cost의 다음 레이어의 입력에 관한 gradient값을 전달 받아
	 *          현재 레이어의 parameter(parameter가 있는 경우), input에 관한 gradient를 계산하고
	 *          이전 레이어에 현재 레이어의 input에 관한 gradient값을 전달한다.
	 * @param idx 현재 레이어에 연결된 다음 레이어의 순번 index
	 * @param next_delta_input 네트워크 cost의 다음 레이어의 입력에 관한 gradient 장치 메모리 포인터
	 */
	virtual void backpropagation(UINT idx, DATATYPE *next_delta_input) {
		_deconcat(idx, next_delta_input);
		if (!isLastNextLayerRequest(idx)) return;

		_backpropagation();
		propBackpropagation();
	}






protected:
	virtual void _shape(bool recursive=true) {
		if(recursive) {
			Layer::_shape();
		}
		checkCudaErrors(Util::ucudaMalloc(&this->d_delta_input, sizeof(DATATYPE)*in_dim.batchsize()));
		checkCudaErrors(Util::ucudaMalloc(&this->d_delta_input, sizeof(DATATYPE)*out_dim.batchsize()));
	}
	virtual void _clearShape() {
		checkCudaErrors(cudaFree(d_delta_input));
		d_delta_input = NULL;
		checkCudaErrors(cudaFree(d_delta_output));
		d_delta_output = NULL;

		Layer::_clearShape();
	}
	/**
	 * @details 네트워크 cost의 다음 레이어의 입력에 관한 gradient값을 전달 받아
	 *          현재 레이어의 parameter(parameter가 있는 경우), input에 관한 gradient를 계산한다.
	 */
	virtual void _backpropagation() {
		return;
	}
	/**
	 * @details 복수의 '다음' 레이어로부터의 gradient를 조합한다.
	 *          기본 조합은 gradient의 합으로 한다.
	 * @param idx 현재 레이어에 연결된 다음 레이어의 순번 index
	 * @param next_delta_input 네트워크 cost의 다음 레이어의 입력에 관한 gradient 장치 메모리 포인터
	 */
	virtual void _deconcat(UINT idx, const DATATYPE* next_delta_input) {
		Util::printMessage("HiddenLayer::backpropagation()---"+string(name));

		Util::printDeviceData(next_delta_input, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "next_delta_input:");
		Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_output:");
		// 첫번째 branch로부터의 backpropagation, 그대로 copy
		if(idx == 0) {
			checkCudaErrors(cudaMemcpyAsync(d_delta_output, next_delta_input, sizeof(DATATYPE)*out_dim.batchsize(), cudaMemcpyDeviceToDevice));
		}
		// 첫번째 이후의 branch로부터의 backpropagation, accumulate gradient
		else {
			checkCudaErrors(cublasSaxpy(Cuda::cublasHandle, static_cast<int>(out_dim.batchsize()),
					&Cuda::alpha, next_delta_input, 1, d_delta_output, 1));
		}
		Util::printDeviceData(d_delta_output, out_dim.rows, out_dim.cols, out_dim.channels, out_dim.batches, "d_delta_output:");
	}

	/**
	 * @details 이전 레이어들에 대해 backpropagation() 메쏘드를 호출한다.
	 */
	void propBackpropagation() {
		HiddenLayer *hiddenLayer;
		for(UINT i = 0; i < prevLayers.size(); i++) {
			hiddenLayer = dynamic_cast<HiddenLayer *>(prevLayers[i].prev_layer);
			if(hiddenLayer) hiddenLayer->backpropagation(prevLayers[i].idx, this->getDeltaInput());
		}
	}




protected:
	DATATYPE *d_delta_input;			///< 네트워크 cost의 현재 레이어 입력에 관한 gradient 장치 메모리 포인터
	DATATYPE *d_delta_output;			///< 네트워크 cost의 현재 레이어 출력에 관한 gradient 장치 메모리 포인터, multi-branch concat memory.
};








#endif /* LAYER_HIDDENLAYER_H_ */



















