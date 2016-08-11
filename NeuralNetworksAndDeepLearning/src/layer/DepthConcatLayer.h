/**
 * @file	DepthConcatLayer.h
 * @date	2016/5/25
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_DEPTHCONCATLAYER_H_
#define LAYER_DEPTHCONCATLAYER_H_

#include "HiddenLayer.h"
#include "../exception/Exception.h"




/**
 * @brief Depth Concatenation 레이어
 * @details 복수의 입력 레이어로부터 전달된 결과값을 하나로 조합하는 레이어
 *          channel축을 기준으로 복수의 입력을 조합한다.
 *          input1: batch1-1(channel1-1-1/channel1-1-2)/batch1-2(channel1-2-1/channel1-2-2)
 *          input2: batch2-1(channel2-1-1/channel2-1-2)/batch2-2(channel2-2-1/channel2-2-2)
 *          output: batch1-1(channel1-1-1/channel1-1-2)/batch2-1(channel2-1-1/channel2-1-2)/batch1-2(channel1-2-1/channel1-2-2)/batch2-2(channel2-2-1/channel2-2-2)
 */
class DepthConcatLayer : public HiddenLayer {
public:
	DepthConcatLayer() { this->type = LayerType::DepthConcat; }
	DepthConcatLayer(const string name);
	virtual ~DepthConcatLayer();

	void backpropagation(UINT idx, DATATYPE *next_delta_input);

	void reset_nabla(UINT idx) {
		if(!isLastPrevLayerRequest(idx)) return;
		propResetNParam();
	}
	void update(UINT idx, UINT n, UINT miniBatchSize) {
		if(!isLastPrevLayerRequest(idx)) return;
		propUpdate(n, miniBatchSize);
	}

	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);

	virtual void shape(UINT idx, io_dim in_dim);
	virtual void reshape(UINT idx, io_dim in_dim);
	virtual void clearShape(UINT idx);

#ifndef GPU_MODE
public:
	DepthConcatLayer(const string name, int n_in);
	rcube &getDeltaInput();
	void feedforward(UINT idx, const rcube &input, const char *end=0);
#else
	/**
	 * @details 조합되어있는 입력에 관한 gradient를 getDeltaInput의 호출 순서에 따라 다시 deconcatenation하여 조회한다.
	 *          getDeltaInput() 호출 순서에 의존성이 있기 때문에 조합한 순서대로 다시 호출해야하므로 호출 순서를 변경할 경우 주의해야 한다.
	 *          feedforward()를 다시 수행한 경우 해당 순서가 reset된다.
	 * @return getDeltaInput() 순서에 따라 deconcate된 d_delta_input 장치 메모리 포인터
	 */
	DATATYPE *getDeltaInput();
	void feedforward(UINT idx, const DATATYPE *input, const char *end=0);
#endif


protected:
	void initialize();
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();


	int offsetIndex;			///< 입력에 관한 gradient 호출한 횟수 카운터, getDeltaInput() 호출마다 증가되고 feedforward()가 수행될 때 reset된다.

#ifndef GPU_MODE
protected:
	rcube delta_input;
	rcube delta_input_sub;

	vector<int> offsets;
#else
protected:
	const float alpha=1.0f, beta=0.0f;				///< cudnn 함수에서 사용하는 scaling factor, 다른 곳으로 옮겨야 함.
#endif



};



#endif /* LAYER_DEPTHCONCATLAYER_H_ */
