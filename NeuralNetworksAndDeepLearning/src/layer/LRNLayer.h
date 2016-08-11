/**
 * @file	LRNLayer.h
 * @date	2016/5/25
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_LRNLAYER_H_
#define LAYER_LRNLAYER_H_

#include "HiddenLayer.h"
#include "LayerConfig.h"
#include "../exception/Exception.h"





/**
 * @brief Local Response Normalization 레이어
 * @details 입력값의 row x column 상의 값들에 대해 인접 채널의 동일 위치값들을 이용(ACROSS CHANNEL)하여 정규화하는 레이어
 *          (WITHIN CHANNEL과 같이 한 채널 내에서 정규화하는 방법도 있으나 아직 사용하지 않아 별도 파라미터로 기능을 제공하지 않음)
 *          'http://caffe.berkeleyvision.org/tutorial/layers.html'의 Local Response Normalization (LRN) 항목 참고
 *          (1+(α/n)∑ixi^2)^β의 수식으로 계산
 */
class LRNLayer : public HiddenLayer {
public:
	LRNLayer() { this->type = LayerType::LRN; }
	LRNLayer(const string name, lrn_dim lrn_d);
	virtual ~LRNLayer();

	void backpropagation(UINT idx, DATATYPE *next_delta_input);

	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);

#ifndef GPU_MODE
public:
	rcube &getDeltaInput() { return delta_input; }
	void _feedforward(UINT idx, const rcube &input, const char *end=0);
	// update할 weight, bias가 없기 때문에 아래의 method에서는 do nothing
	void reset_nabla(UINT idx) {
		if(!isLastPrevLayerRequest(idx)) throw Exception();
		propResetNParam();
	}
#else
public:
	DATATYPE *getDeltaInput() { return d_delta_input; }

#endif


protected:
	void initialize(lrn_dim lrn_d);

	virtual void _save(ofstream &ofs);
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();
	virtual void _feedforward(const DATATYPE *input, const char *end=0);

	lrn_dim lrn_d;					///< LRN 연산 관련 파라미터 구조체

#ifndef GPU_MODE
private:
	rcube delta_input;
	rcube z;	// beta powered 전의 weighted sum 상태의 norm term
#else
private:
	const float alpha=1.0f, beta=0.0f;			///< cudnn 함수에서 사용하는 scaling factor, 다른 곳으로 옮겨야 함.
	cudnnLRNDescriptor_t lrnDesc;				///< cudnn LRN 연산 정보 구조체
#endif

};




#endif /* LAYER_LRNLAYER_H_ */
