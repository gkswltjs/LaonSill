/**
 * @file	Activation.h
 * @date	2016/5/10
 * @author	jhkim
 * @brief	Activation 추상 클래스와 Activation 타입 열거형을 정의.
 * @details
 */


#ifndef ACTIVATION_ACTIVATION_H_
#define ACTIVATION_ACTIVATION_H_

#include "../layer/LayerConfig.h"
#include "../cuda/Cuda.h"
#include "../Util.h"






/**
 * @brief	Activation 구현 클래스의 베이스 추상 클래스.
 * @details	Activation 클래스를 상속받아 Activation을 구현하는 클래스를 생성할 수 있음.
 */
class Activation {
public:
	Activation() {};
	virtual ~Activation() {};

	/**
	 * @brief	Activation 타입 열거형
	 * @details	지원하는 Activation 타입 열거,
	 *          현재 Sigmoid, Softmax, ReLU 함수를 지원.
	 */
	enum Type {
		None = 0,		// Activation을 사용하지 않음, 입력값을 그대로 출력.
		Sigmoid = 1, 	// Activation에 Sigmoid 함수를 적용.
		Softmax = 2,	// Activation에 Softmax 함수를 적용.
		ReLU = 3		// Activation에 Rectified Linear Unit 함수를 적용.
	};

	Activation::Type getType() const { return type; }

	/**
	 * activation function에 따라 layer weight의 초기화하는 방법이 다름
	 */
	//virtual void initialize_weight(int n_in, rmat &weight)=0;
	//virtual void initialize_weight(int n_in, rcube &weight)=0;


#ifndef GPU_MODE
	/**
	 * activation function
	 */
	virtual void activate(const rcube &z, rcube &activation)=0;

	/**
	 * activation derivation
	 * 실제 weighted sum값을 이용하여 계산하여야 하지만
	 * 현재까지 activation으로도 계산이 가능하여 파라미터를 activation으로 지정
	 * weighted sum값이 필요한 케이스에 수정이 필요
	 */
	virtual void d_activate(const rcube &activation, rcube &da)=0;
#else
	/**
	 * @details 입력값에 대해 활성화 함수를 적용한 값을 반환.
	 * @param z 활성화 함수를 적용할 입력값을 담고 있는 장치 메모리 포인터.
	 * @param activation 활성화 함수를 적용한 출력값을 담고 있는 장치 메모리 포인터.
	 * @param tensorDesc 입력값 z의 데이터 구성을 설명하는 cudnnTensorDescriptor 포인터.
	 */
	virtual void activate(const DATATYPE *z, DATATYPE *activation, cudnnTensorDescriptor_t &tensorDesc)=0;

	/**
	 * @details 활성화 함수 출력값을 입력값에 대해 미분한 결과를 반환.
	 * @param activate activate()를 통해 활성화 함수를 적용했던 출력값을 담고 있는 장치 메모리 포인터.
	 * @param deltaInput Cost를 activate에 대해 미분한 결과를 담고 있는 장치 메모리 포인터.
	 * @param z activate()를 통해 활성화 함수를 적용하고자 한 입력값을 담고 있는 장치 메모리 포인터.
	 * @param da Cost를 z에 대해 미분한 결과를 담고 있는 장치 메모리 포인터.
	 * @param tensorDesc activate, deltaInput, z, da의 데이터 구성을 설명하는 cudnnTensorDescriptor 포인터.
	 */
	virtual void d_activate(const DATATYPE *activation, const DATATYPE *deltaInput, const DATATYPE *z, DATATYPE *da,
			cudnnTensorDescriptor_t &tensorDesc) = 0;

#endif


protected:
	Type type;	///< 현재 Activation 객체의 Activation 타입.

};



#endif /* ACTIVATION_ACTIVATION_H_ */
