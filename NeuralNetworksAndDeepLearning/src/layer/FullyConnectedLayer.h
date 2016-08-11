/**
 * @file	FullyConnectedLayer.h
 * @date	2016/5/10
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_FULLYCONNECTEDLAYER_H_
#define LAYER_FULLYCONNECTEDLAYER_H_

#include "HiddenLayer.h"
#include "LayerConfig.h"
#include "../activation/Activation.h"
#include "../activation/ActivationFactory.h"
#include "../cost/Cost.h"





/**
 * @brief Fully Connected (Inner Product) 레이어
 * @details 이전 레이어와 현재 레이어의 모든 노드들에 대해 연결성이 있고
 *          연결성을 통해 weighted sum, activation을 수행 출력값을 계산하는 레이어이다.
 *          입력 레이어가 다차원인 경우(이미지의 경우 height x width x channel의 3차원) 1차원으로 flatten((height*width*channel) x 1 x 1)된다.
 *          출력 역시 1차원 flatten 결과이며 필요에 따라서 입력받는 레이어에서 다시 차원을 복구해야 한다.
 */
class FullyConnectedLayer : public HiddenLayer {
public:
	FullyConnectedLayer() { this->type = LayerType::FullyConnected; }
	/**
	 * @details FullyConnectedLayer 생성자
	 * @param name 레이어의 이름 문자열 포인터
	 * @param n_out 출력 노드의 수
	 * @param p_dropout dropout을 적용할 확율
	 * @param weight_update_param weight 갱신 관련 파라미터 구조체
	 * @param bias_update_param bias 갱신 관련 파라미터 구조체
	 * @param weight_filler weight 초기화 관련 파라미터 구조체
	 * @param bias_filler bias 초기화 관련 파라미터 구조체
	 * @param activationType weighted sum에 적용할 활성화 타입
	 */
	FullyConnectedLayer(const string name, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, ActivationType activationType=ActivationType::None);
	virtual ~FullyConnectedLayer();

	virtual void backpropagation(UINT idx, DATATYPE *next_delta_input);
	virtual void reset_nabla(UINT idx);
	virtual void update(UINT idx, UINT n, UINT miniBatchSize);
	virtual void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);
	virtual bool isLearnable() { return true; }

#ifndef GPU_MODE
public:
	FullyConnectedLayer(const string name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, ActivationType activationType=ActivationType::None);

	rmat &getWeight() { return this->weight; }
	rcube &getDeltaInput() { return this->delta_input; }

	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	virtual void feedforward(UINT idx, const rcube &input, const char *end=0);
#else
public:
	/**
	 * @details 레이어의 weight 장치 메모리 포인터를 조회한다.
	 * @return 레이어의 weight 장치 메모리 포인터
	 */
	DATATYPE *getWeight() { return this->d_weight; }
	DATATYPE *getDeltaInput() { return this->d_delta_input; }
	virtual void feedforward(UINT idx, const DATATYPE *input, const char *end=0);

#endif





private:
	void initialize(int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler, ActivationType activationType);

protected:
	virtual void _save(ofstream &ofs);
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();
	virtual DATATYPE _sumSquareParam();
	virtual DATATYPE _sumSquareParam2();
	virtual void _scaleParam(DATATYPE scale_factor);

	double p_dropout;						///< dropout을 적용할 확율

	update_param weight_update_param;		///< weight 갱신 관련 파라미터 구조체
	update_param bias_update_param;			///< bias 갱신 관련 파라미터 구조체

	param_filler weight_filler;				///< weight 초기화 관련 파라미터 구조체
	param_filler bias_filler;				///< bias 초기화 관련 파라미터 구조체

	Activation *activation_fn;				///< 활성화 객체

#ifndef GPU_MODE
protected:
	rmat weight;
	rvec bias;

	rvec nabla_b;
	rmat nabla_w;

	rcube z;
	rcube delta;
	rcube delta_input;
#else
protected:
	DATATYPE *weight;						///< weight 호스트 메모리 포인터 (초기화 및 읽기, 쓰기용)
	DATATYPE *bias;							///< bias 호스트 메모리 포인터 (초기화 및 읽기, 쓰기용)

	DATATYPE *d_weight;						///< weight 장치 메모리 포인터
	DATATYPE *d_bias;						///< bias 장치 메모리 포인터

	DATATYPE *d_z;							///< weighted sum 장치 메모리 포인터
	DATATYPE *d_delta;						///< 네트워크 cost의 z(weighted sum)에 관한 gradient 장치 메모리 포인터
	DATATYPE *d_delta_weight;				///< 네트워크 cost의 weight에 관한 gradient 장치 메모리 포인터
	DATATYPE *d_delta_weight_prev;			///< 이전 업데이트의 네트워크 cost의 weight에 관한 gradient 장치 메모리 포인터 (momentum 계산용)
	DATATYPE *d_delta_bias;					///< 네트워크 cost의 bias에 관한 gradient 장치 메모리 포인터
	DATATYPE *d_delta_bias_prev;			///< 이전 업데이트의 네트워크 cost의 bias에 관한 gradient 장치 메모리 포인터 (momentum 계산용)

	DATATYPE *d_onevec;						///< batch 사이즈의 1 벡터, bias를 weighted sum에 더해 줄 때 사용

	DATATYPE *mask;							///< dropout 마스크 호스트 메모리 포인터
	DATATYPE *d_mask;						///< dropout 마스크 장치 메모리 포인터
	DATATYPE scale;							///< dropout 스케일 팩터


#endif


};






#endif /* LAYER_FULLYCONNECTEDLAYER_H_ */
