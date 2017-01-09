/**
 * @file	LRNLayer.h
 * @date	2016/5/25
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_LRNLAYER_H_
#define LAYER_LRNLAYER_H_

#include "common.h"
#include "HiddenLayer.h"
#include "LayerConfig.h"
#include "Exception.h"





/**
 * @brief Local Response Normalization 레이어
 * @details 입력값의 row x column 상의 값들에 대해 인접 채널의 동일 위치값들을 이용(ACROSS CHANNEL)하여 정규화하는 레이어
 *          (WITHIN CHANNEL과 같이 한 채널 내에서 정규화하는 방법도 있으나 아직 사용하지 않아 별도 파라미터로 기능을 제공하지 않음)
 *          'http://caffe.berkeleyvision.org/tutorial/layers.html'의 Local Response Normalization (LRN) 항목 참고
 *          (1+(α/n)∑ixi^2)^β의 수식으로 계산
 */
template <typename Dtype>
class LRNLayer : public HiddenLayer<Dtype> {
public:
	/**
	 * @brief LRN 레이어 객체 빌더
	 * @details LRN 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 LRN 레이어 객체를 생성한다.
	 */
	class Builder : public HiddenLayer<Dtype>::Builder {
	public:
		lrn_dim _lrnDim;
		Builder() {
			this->type = Layer<Dtype>::LRN;
		}
		Builder* lrnDim(uint32_t local_size, double alpha, double beta, double k) {
			this->_lrnDim.local_size = local_size;
			this->_lrnDim.alpha = alpha;
			this->_lrnDim.beta = beta;
			this->_lrnDim.k = k;
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
			return new LRNLayer(this);
		}
	};
	/**
	 * @details LRNLayer 기본 생성자
	 */
	LRNLayer();
	LRNLayer(Builder* builder);
	/**
	 * @details LRNLayer 생성자
	 * @param 레이어 이름 문자열
	 * @param lrn_d LRN 연산 관련 파라미터 구조체
	 */
	LRNLayer(const std::string name, lrn_dim lrn_d);
	/**
	 * @details LRNLayer 소멸자
	 */
	virtual ~LRNLayer();


	virtual void reshape();
	virtual void feedforward();
	virtual void backpropagation();

protected:
	void initialize(lrn_dim lrn_d);

	virtual void _clearShape();


protected:
	lrn_dim lrn_d;								///< LRN 연산 관련 파라미터 구조체

#ifndef GPU_MODE
	rcube delta_input;
	rcube z;	// beta powered 전의 weighted sum 상태의 norm term
#else
	cudnnTensorDescriptor_t inputTensorDesc;			///< cudnn 입력 데이터(n-D 데이터셋) 구조 정보
	cudnnTensorDescriptor_t outputTensorDesc;			///< cudnn 출력 데이터(n-D 데이터셋) 구조 정보
	cudnnLRNDescriptor_t lrnDesc;				///< cudnn LRN 연산 정보 구조체
#endif

};


#endif /* LAYER_LRNLAYER_H_ */
