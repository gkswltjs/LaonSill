/**
 * @file PoolingLayer.h
 * @date 2016/5/23
 * @author jhkim
 * @brief
 * @details
 */


#ifndef LAYER_POOLINGLAYER_H_
#define LAYER_POOLINGLAYER_H_

#include "common.h"
#include "HiddenLayer.h"
#include "Pooling.h"
#include "PoolingFactory.h"
#include "Exception.h"

/**
 * @brief 풀링 레이어
 * @details Max 풀링, Average 풀링 제공
 *          padding에 관한 옵션을 제공하지 않고 있고
 *          GoogLeNet에 따라 Max 풀링의 경우 padding을 기본으로, Average 풀링의 경우 Non padding을 기본으로 하고 있다.
 */
template <typename Dtype>
class PoolingLayer : public HiddenLayer<Dtype> {
public:
	/**
	 * @brief 풀링 레이어 객체 빌더
	 * @details 풀링 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 풀링 레이어 객체를 생성한다.
	 */
	class Builder : public HiddenLayer<Dtype>::Builder {
	public:
		pool_dim _poolDim;								///< 풀링 파라미터
		typename Pooling<Dtype>::Type _poolingType;		///< 풀링 타입

		Builder() {
			this->type = Layer<Dtype>::Pool;
			_poolDim.cols = 0;
			_poolDim.rows = 0;
			_poolDim.pad = 0;
			_poolDim.stride = 0;
			_poolingType = Pooling<Dtype>::Max;
		}
		Builder* poolDim(uint32_t cols, uint32_t rows, uint32_t pad, uint32_t stride) {
			this->_poolDim.cols = cols;
			this->_poolDim.rows = rows;
			this->_poolDim.pad = pad;
			this->_poolDim.stride = stride;
			return this;
		}
		Builder* poolingType(typename Pooling<Dtype>::Type poolingType) {
			this->_poolingType = poolingType;
			return this;
		}
		virtual Builder* name(const std::string name) {
			HiddenLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			HiddenLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			HiddenLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			HiddenLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			HiddenLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		Layer<Dtype>* build() {
			return new PoolingLayer(this);
		}
	};

	/**
	 * @details PoolingLayer 기본 생성자
	 */
	PoolingLayer();
	PoolingLayer(Builder* builder);
	/**
	 * @details PoolingLayer 생성자
	 * @param name 레이어 이름 문자열 포인터
	 * @param pool_d 풀링 연산 관련 파라미터 구조체
	 */
	PoolingLayer(const std::string name, pool_dim pool_d, typename Pooling<Dtype>::Type poolingType);
	/**
	 * @details PoolingLayer 소멸자
	 */
	virtual ~PoolingLayer();


	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();


protected:
	void initialize(pool_dim pool_d, typename Pooling<Dtype>::Type poolingType);

	virtual void _clearShape();

protected:
	pool_dim pool_d;				///< 풀링 연산 관련 파라미터 구조체
	Pooling<Dtype> *pooling_fn;			///< 풀링 객체

#ifndef GPU_MODE
	ucube pool_map;
	rcube delta;
	rcube delta_input;
#else
	cudnnTensorDescriptor_t inputTensorDesc;			///< cudnn 입력 데이터(n-D 데이터셋) 구조 정보
	cudnnTensorDescriptor_t outputTensorDesc;			///< cudnn 출력 데이터(n-D 데이터셋) 구조 정보
#endif



};





#endif /* LAYER_POOLINGLAYER_H_ */
