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
#include <armadillo>

using namespace arma;


/**
 * @brief 히든 레이어 기본 추상 클래스
 * @details 기본 레이어의 클래스에 backpropagation, parameter update와 같은
 *          파라미터 학습 관련 기능을 추가한다.
 */
class HiddenLayer : public Layer {
public:
	HiddenLayer() {}
	HiddenLayer(const char *name) : Layer(name) {}
	virtual ~HiddenLayer() {}

	/**
	 * @details 네트워크 cost의 다음 레이어의 입력에 관한 gradient값을 전달 받아
	 *          현재 레이어의 parameter(parameter가 있는 경우), input에 관한 gradient를 계산하고
	 *          이전 레이어에 현재 레이어의 input에 관한 gradient값을 전달한다.
	 * @param idx 현재 레이어에 연결된 다음 레이어의 순번 index
	 * @param next_delta_input 네트워크 cost의 다음 레이어의 입력에 관한 gradient 장치 메모리 포인터
	 */
	virtual void backpropagation(UINT idx, DATATYPE *next_delta_input) { propBackpropagation(); }

	/**
	 * @details CPU MODE에서 누적된 학습 parameter값들을 reset한다.
	 *          현재 CPU_MODE의 경우 batch 단위의 학습을 한번에 수행하지 않고
	 *          각 데이터 별로 학습, 학습결과를 누적하여 batch 단위만큼 학습된 경우
	 *          누적된 학습결과를 갱신한다. 이 때 누적된 학습결과를 다음 batch 학습을 위해 reset하는 기능을 한다.
	 *          (GPU 연산에서는 현재 아무일도 하지 않고 있음)
	 * @param idx 현재 레이어에 연결된 다음 레이어의 순번 index
	 */
	virtual void reset_nabla(UINT idx)=0;

	/**
	 * @details batch 단위의 학습을 종료한 후, 학습 파라미터들을 갱신한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 idx
	 * @param n 전체 학습 데이터의 수
	 * @param miniBatchSize batch 사이즈
	 */
	virtual void update(UINT idx, UINT n, UINT miniBatchSize)=0;


#ifndef GPU_MODE
public:
	HiddenLayer(const char *name, int n_in, int n_out) : Layer(name, n_in, n_out) {}
	virtual rcube &getDeltaInput()=0;
#else
public:
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



protected:

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

#ifndef GPU_MODE
protected:
#else
protected:
	virtual void _shape(bool recursive=true) {
		if(recursive) {
			Layer::_shape();
		}
	}
	virtual void _clearShape() {
		Layer::_clearShape();
	}

	DATATYPE *d_delta_input;			///< 네트워크 cost의 현재 레이어 입력에 관한 gradient 장치 메모리 포인터


#endif





};








#endif /* LAYER_HIDDENLAYER_H_ */



















