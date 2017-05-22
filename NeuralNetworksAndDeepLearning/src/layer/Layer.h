/**
 * @file	Layer.h
 * @date	2016/5/10
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_LAYER_H_
#define LAYER_LAYER_H_

#include <cudnn.h>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <memory>

#include "common.h"
#include "Data.h"
#include "LayerConfig.h"

template <typename Dtype> class NetworkConfig;

/**
 * @brief 레이어 베이스 추상 클래스, 모든 레이어는 이 클래스를 상속받아 구현한다.
 * @details
 * @todo  save/load/reshape 관련 method의 경우 아직 정리가 되지 않음,
 *        레이어 리팩토링이 어느정도 정리된 후에 작업할 예정임.
 *        template화하였지만 float에 대해서만 처리됨, double과 관련된 처리를 구현해야 함.
 */
template <typename Dtype>
class Layer {
public:
	/**
	 * @brief 레이어 타입 열거형
	 * @details	지원하는 레이어 타입 열거,
	 */
	enum LayerType : int {
		None = 0,
        AAA,                    // template class
		Input, 					// 입력 레이어
        NoiseInput,             // noise 입력 레이어
        ALEInput,               // ALE 입력 레이어
        CelebAInput,            // CelebA 입력 레이어
        KistiInput,             // Kisti project input layer. (FIXME: name;;;)
        VOCPascalInput,         // VOC PASCAL data
        ILSVRCInput,

		FullyConnected, 		// Fully Connected 레이어
		Conv, 					// 컨볼루션 레이어
		Pool, 					// 풀링 레이어
		DepthConcat,			// Depth Concat 레이어
		Inception, 				// 인셉션 레이어
		LRN,					// Local Response Normaization 레이어
		Sigmoid, 				// 시그모이드 레이어
		Softmax,				// 소프트맥스 레이어
		Split,					//
        DQNOutput,              // DQN Output 레이어

        Deconv,                 // deconvolution 레이어
                                // 엄밀히 말하면 transpose convolution layer 혹은
                                // fractionally strided convolution layer 입니다

        BatchNorm,              // Batch normalization 레이어
        BatchNorm2,             // CUDNN으로 만든 Batch normalization 레이어

		Reshape,				//
		SmoothL1Loss,
		SoftmaxWithLoss,
        CrossEntropyWithLoss,
        YoloLoss,

        Sigmoid2,               // 새로운 sigmoid layer. 구현 완료되면 기존 sigmoid layer를
                                // 대체할 예정.
        HyperTangent,

		AnchorTarget,			//
		Proposal,				//
		ProposalTarget,			//
		RoIPooling,				//
		RoIInput,
		RoITestInput,
		FrcnnTestOutput,

		Relu,
        DropOut,
        LayerTypeMax
	};


	/**
	 * @brief 레이어 객체 빌더
	 * @details 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 레이어 객체를 생성한다.
	 */
	class Builder {
	public:
		LayerType type;
        std::string _name;							///< 레이어의 이름
		uint32_t _id;								///< 레이어의 아이디

        std::vector<std::string> _inputs;
        std::vector<std::string> _outputs;

        std::vector<bool> _propDown;

        bool _isDonator;
        bool _isReceiver;
        uint32_t _donatorID;

		Builder() {
			type = Layer<Dtype>::None;
			_name = "";

            this->_isDonator = false;
            this->_isReceiver = false;
            this->_donatorID = 0;
		}
		virtual ~Builder() {}
		virtual Builder* name(const std::string name) {
			this->_name = name;
			return this;
		}
		virtual Builder* id(uint32_t id) {
			this->_id = id;
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			this->_inputs = inputs;
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			this->_outputs = outputs;
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			this->_propDown = propDown;
			return this;
		}
        Builder* donate() {
            this->_isDonator = true;
			return this;
        }
        Builder* receive(uint32_t donatorID) {
            this->_isReceiver = true;
            this->_donatorID = donatorID;
			return this;
        }

		virtual Layer<Dtype>* build() = 0;
	};

	////////////////////////////////////////////////////////////////////
	// CONSTRUCTOR & DESCTRUCTOR
	////////////////////////////////////////////////////////////////////

	/**
	 * @details 레이어 클래스 기본 생성자
	 */
	Layer() {}
	/**
	 * @details 레이어 클래스 빌더 생성자
	 * @param builder 레이어의 설정 정보를 담고 있는 객체
	 */
	Layer(Builder* builder);
	/**
	 * @details 레이어 클래스 생성자
	 * @param name 레이어 이름에 대한 문자열 포인터
	 */
	Layer(const std::string& name);
	/**
	 * @details 레이어 클래스 소멸자
	 */
	virtual ~Layer();

	////////////////////////////////////////////////////////////////////
	// GETTER & SETTER
	////////////////////////////////////////////////////////////////////

	/**
	 * @details 레이어에 부여된 유일한 id값을 조회한다.
	 * @return 레이어 id
	 */
	int getId() const { return id; }
	/**
	 * @details 레이어의 이름을 조회한다.
	 * @return 레이어 이름
	 */
	const std::string getName() const { return name; }
	/**
	 * @biref 레이어의 타입을 조회한다.
	 * @return 레이어 타입
	 */
	typename Layer<Dtype>::LayerType getType() const { return this->type; }
	/**
	 * @brief 레이어의 입력 데이터 이름 목록을 조회한다.
	 * @return 레이어 입력 데이터 이름 목록
	 */
	std::vector<std::string>& getInputs() { return this->_inputs; }
	std::vector<std::string>& getOutputs() { return this->_outputs; }
	uint32_t getInputsSize() const { return this->_inputs.size(); }
	uint32_t getOutputsSize() const { return this->_outputs.size(); }

	/**
	 * @brief 레이어의 입력 데이터 목록을 조회한다.
	 * @return 레이어 입력 데이터 목록
	 */
	std::vector<Data<Dtype>*>& getInputData() { return this->_inputData; }
	std::vector<Data<Dtype>*>& getOutputData() { return this->_outputData; }

	/**
	 * @details 레이어에 네트워크 설정값을 설정한다.
	 * @param networkConfig 네트워크 설정값 객체
	 */
	virtual void setNetworkConfig(NetworkConfig<Dtype>* networkConfig) { 
        this->networkConfig = networkConfig; 
    }

	/**
	 * @details 현재 레이어의 입력/출력 데이터 구조정보에 의존성이 있는 자료구조들을 구성하고
     *         초기화하고 다음 레이어들에 대해 shape()를 요청한다.
	 * @param idx 요청을 보낸 이전 레이어의 id
	 * @param in_dim 현재 레이어의 입력 데이터 구조정보
	 */
	virtual void reshape();
	/**
	 * @details 이미 shape가 구성된 레이어의 shape를 변경하고 다음 레이어들에 대해 reshape()를
     *         요청한다.
	 * @param idx 요청을 보낸 이전 레이어의 id
	 * @param in_dim 새롭게 변경할 현재 레이어의 입력 데이터 구조정보
	 */

#ifndef GPU_MODE
	/**
	 * @details batch단위로 누적된 gradient를 초기화하고 다음 레이어들에 대해 reset_nabla()를
     *          요청한다.
	 * @param idx 요청을 보낸 이전 레이어의 id
	 * @todo GPU_MODE에서 사용하지 않는다.
	 */
	virtual void reset_nabla(uint32_t idx);
#else
	/**
	 * @details 레이어 입력값을 전달받아 출력값을 계산하고 다음 레이어들에 대해
     *          feedforward()를 요청한다.
	 * @param idx 요청을 보낸 이전 레이어의 id
	 * @param input 현재 레이어에 전달된 레이어 입력값 장치 포인터
	 * @param end feedforward 종료 레이어 이름, 0인 경우 계속 진행
	 */
	virtual void feedforward();
	virtual void backpropagation();
#endif

protected:
	/**
	 * @details 레이어를 초기화한다.
	 * @param name 레이어의 이름 문자열 포인터
	 */
	void initialize(uint32_t id, const std::string name);

	bool _adjustInputShape();
	bool _isInputShapeChanged(uint32_t index);

public:
	std::vector<std::string> _inputs;					///< 레이어 입력 데이터 이름 목록 벡터
	std::vector<std::string> _outputs;					///< 레이어 출력 데이터 이름 목록 벡터

	std::vector<Data<Dtype>*> _inputData;				///< 레이어 입력 데이터 목록 벡터
	std::vector<Data<Dtype>*> _outputData;				///< 레이어 출력 데이터 목록 벡터


	Layer<Dtype>::LayerType type;					    ///< 레이어의 타입

	int id;												///< 레이어의 고유 아이디
	std::string name;									///< 레이어의 이름

    // FIXME: 디버깅 때문에 임시로 protected -> public으로 변수를 변경하였음..
    // inputShape는 input에 대한 메타이다.
    // 일반적으로 0번 인덱스에 해당하는 원소는 inputData에 대한 메타 값이 들어 있고, 
    // 1번 인덱스에 해당하는 원소는 라벨에 대한 메타 값이 들어 있다.
    // 각 원소의 메타 값은 { batch, channel, row, column } 크기가 들어 있다.
	std::vector<std::vector<uint32_t>> _inputShape;

protected:
	NetworkConfig<Dtype>* networkConfig;				///< 레이어가 속한 네트워크의 설정

	std::vector<bool> _propDown;

	static const int LAYER_NAME_LENGTH = 32;
};

#endif /* LAYER_LAYER_H_ */
