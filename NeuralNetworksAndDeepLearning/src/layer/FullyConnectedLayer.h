/**
 * @file	FullyConnectedLayer.h
 * @date	2016/5/10
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_FULLYCONNECTEDLAYER_H_
#define LAYER_FULLYCONNECTEDLAYER_H_

#include "common.h"
#include "Layer.h"
#include "LearnableLayer.h"
#include "LayerConfig.h"
#include "Activation.h"
#include "ActivationFactory.h"
#include "Cost.h"

/**
 * @brief Fully Connected (Inner Product) 레이어
 * @details 이전 레이어와 현재 레이어의 모든 노드들에 대해 연결성이 있고
 *          연결성을 통해 weighted sum, activation을 수행 출력값을 계산하는 레이어이다.
 *          입력 레이어가 다차원인 경우(이미지의 경우 height x width x channel의 3차원) 
 *          1차원으로 flatten((height*width*channel) x 1 x 1)된다.
 *          출력 역시 1차원 flatten 결과이며 필요에 따라서 입력받는 레이어에서 다시 차원을
 *          복구해야 한다.
 */
template <typename Dtype>
class FullyConnectedLayer : public LearnableLayer<Dtype> {
public:
	/**
	 * @brief Fully Connected 레이어 객체 빌더
	 * @details Fully Connected 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 Fully Connected 레이어 객체를 생성한다.
	 */
	class Builder : public LearnableLayer<Dtype>::Builder {
	public:
		uint32_t _nOut;										///< 출력 노드의 수
		double _pDropout;									///< dropout을 적용할 확율
		update_param _weightUpdateParam;					///< weight 갱신 관련 파라미터
		update_param _biasUpdateParam;						///< bias 갱신 관련 파라미터
		param_filler<Dtype> _weightFiller;					///< weight 초기화 관련 파라미터
		param_filler<Dtype> _biasFiller;					///< bias 초기화 관련 파라미터
		int _axis;
		//typename Activation<Dtype>::Type _activationType;	///< weighted sum에 적용할 활성화

		Builder() {
			this->type = Layer<Dtype>::FullyConnected;
			this->_nOut = 0;
			this->_pDropout = 0.0;
			this->_weightUpdateParam.lr_mult = 1.0;
			this->_weightUpdateParam.decay_mult = 0.0;
			this->_biasUpdateParam.lr_mult = 1.0;
			this->_biasUpdateParam.decay_mult = 0.0;
			this->_weightFiller.type = ParamFillerType::Constant;
			this->_weightFiller.value = 0.0f;
			this->_biasFiller.type = ParamFillerType::Constant;
			this->_biasFiller.value = 0.0f;
			this->_axis = 1;
			//_activationType = Activation<Dtype>::NoActivation;
            //
		}
		Builder* nOut(uint32_t nOut) {
			this->_nOut = nOut;
			return this;
		}
		Builder* pDropout(double pDropout) {
			this->_pDropout = pDropout;
			return this;
		}
		Builder* weightUpdateParam(double lr_mult, double decay_mult) {
			this->_weightUpdateParam.lr_mult = lr_mult;
			this->_weightUpdateParam.decay_mult = decay_mult;
			return this;
		}
		Builder* biasUpdateParam(double lr_mult, double decay_mult) {
			this->_biasUpdateParam.lr_mult = lr_mult;
			this->_biasUpdateParam.decay_mult = decay_mult;
			return this;
		}
		Builder* weightFiller(ParamFillerType weightFillerType, Dtype value) {
			this->_weightFiller.type = weightFillerType;
			this->_weightFiller.value = value;
			return this;
		}
		Builder* biasFiller(ParamFillerType paramFillerType, Dtype value) {
			this->_biasFiller.type = paramFillerType;
			this->_biasFiller.value = value;
			return this;
		}
		Builder* axis(int axis) {
			this->_axis = axis;
			return this;
		}
		virtual Builder* name(const std::string name) {
			LearnableLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			LearnableLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			LearnableLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			LearnableLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			LearnableLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		Layer<Dtype>* build() {
			return new FullyConnectedLayer(this);
		}
	};

	/**
	 * @details FullyConnectedLayer 기본 생성자
	 *          내부적으로 레이어 타입만 초기화한다.
	 */
	FullyConnectedLayer(Builder* builder);

    FullyConnectedLayer(const std::string& name);

	virtual ~FullyConnectedLayer();

	//////////////////////////////////////////
	// Learnable Layer Method
	//////////////////////////////////////////
	using Layer<Dtype>::getName;
	virtual const std::string getName() { return this->name; }
	virtual void update();
	//virtual double sumSquareParamsData();
	//virtual double sumSquareParamsGrad();
	//virtual void scaleParamsGrad(float scale);
	//virtual double testParamAbnormality();
	//virtual uint32_t boundParams();
	//virtual uint32_t numParams();
	//virtual void saveParams(std::ofstream& ofs);
	//virtual void loadParams(std::ifstream& ifs);
	//virtual void loadParams(std::map<std::string, Data<Dtype>*>& dataMap);
	//////////////////////////////////////////

	virtual void backpropagation();
	virtual void reshape();
	virtual void feedforward();

private:
	/**
	 * @details FullyConnectedLayer를 초기화한다.
	 * @param n_out 출력 노드의 수
	 * @param p_dropout dropout을 적용할 확율
	 * @param weight_update_param weight 갱신 관련 파라미터 구조체
	 * @param bias_update_param bias 갱신 관련 파라미터 구조체
	 * @param weight_filler weight 초기화 관련 파라미터 구조체
	 * @param bias_filler bias 초기화 관련 파라미터 구조체
	 * @param activationType weighted sum에 적용할 활성화 타입
     * @param useBatchNorm batch normalization 사용 여부
     * @param batchNormEpsilon epsilon value for batch normalization
	 */
	void initialize(int n_out, double p_dropout, int axis, update_param weight_update_param,
        update_param bias_update_param, param_filler<Dtype> weight_filler,
        param_filler<Dtype> bias_filler);
	void initialize();

    void applyChanges(LearnableLayer<Dtype> *targetLayer);
    void syncParams(LearnableLayer<Dtype> *targetLayer);

protected:



	void _computeWeightedData();
	void _computeWeightBiasedData();
	//void _computeActivatedData();

	//void _computePreActivationGrad();
	void _computeWeightGrad();
	void _computeBiasGrad();
	void _computeInputGrad();

    // FIXME: 파라미터가 너무 많다. 구조화해서 줄이자.
	void _updateParam(const uint32_t paramSize, const Dtype regScale, const Dtype learnScale,
        const Dtype epsilon, const Dtype decayRate, const Dtype beta1, const Dtype beta2,
        Data<Dtype>* dataHistory, Data<Dtype>* dataHistory2, Data<Dtype>* data);
	void _dropoutForward();
	void _dropoutBackward();

	enum ParamType {
		Weight = 0,
		Bias = 1
	};

protected:
	uint32_t n_out;
	double p_dropout;						///< dropout을 적용할 확율
	int axis;

	update_param weight_update_param;		///< weight 갱신 관련 파라미터 구조체
	update_param bias_update_param;			///< bias 갱신 관련 파라미터 구조체

	param_filler<Dtype> weight_filler;				///< weight 초기화 관련 파라미터 구조체
	param_filler<Dtype> bias_filler;				///< bias 초기화 관련 파라미터 구조체

	//Activation<Dtype> *activation_fn;				///< 활성화 객체

#ifndef GPU_MODE
	rmat weight;
	rvec bias;

	rvec nabla_b;
	rmat nabla_w;

	rcube z;
	rcube delta;
	rcube delta_input;
#else
	//cudnnTensorDescriptor_t inputTensorDesc;	///< cudnn 입력 데이터(n-D 데이터셋) 구조 정보
	//cudnnTensorDescriptor_t outputTensorDesc;   ///< cudnn 출력 데이터(n-D 데이터셋) 구조 정보

	Dtype* d_onevec;    ///< batch 사이즈의 1 벡터, bias를 weighted sum에 더해 줄 때 사용

	//Dtype* mask;	    ///< dropout 마스크 호스트 메모리 포인터
	//Dtype* d_mask;	///< dropout 마스크 장치 메모리 포인터
	SyncMem<Dtype> _mask;

	Dtype scale;		///< dropout 스케일 팩터
#endif


public:
    uint32_t batches;
    uint32_t in_rows;
    uint32_t out_rows;

public:
    void donateParam(FullyConnectedLayer<Dtype>* receiver);

public:
    /****************************************************************************
     * layer callback functions 
     ****************************************************************************/
    static void* initLayer();
    static void destroyLayer(void* instancePtr);
    static void setInOutTensor(void* instancePtr, void* tensorPtr, bool isInput, int index);
    static bool allocLayerTensors(void* instancePtr);
    static void forwardTensor(void* instancePtr, int miniBatchIndex);
    static void backwardTensor(void* instancePtr);
    static void learnTensor(void* instancePtr);
};

#endif /* LAYER_FULLYCONNECTEDLAYER_H_ */
