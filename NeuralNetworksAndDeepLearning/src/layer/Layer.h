/**
 * @file	Layer.h
 * @date	2016/5/10
 * @author	jhkim
 * @brief
 * @details
 */





#ifndef LAYER_LAYER_H_
#define LAYER_LAYER_H_

#include "LayerConfig.h"
#include "../cuda/Cuda.h"
#include "../Util.h"
#include <armadillo>
#include <iostream>
#include <map>

using namespace arma;

const int LAYER_NAME_LENGTH = 32;

/**
 * @brief 레이어 타입 열거형
 * @details	지원하는 레이어 타입 열거,
 */
enum class LayerType {
	Input=0, 					// 입력 레이어
	FullyConnected=1, 			// Fully Connected 레이어
	Conv=2, 					// 컨볼루션 레이어
	Pooling=3, 					// 풀링 레이어
	DepthConcat=4,				// Depth Concat 레이어
	Inception=5, 				// 인셉션 레이어
	LRN=6,						// Local Response Normaization 레이어
	Sigmoid=7, 					// 시그모이드 레이어
	Softmax=8					// 소프트맥스 레이어
};


/**
 * @brief 레이어 베이스 추상 클래스, 모든 레이어는 이 클래스를 상속받아 구현한다.
 * @details
 */
class Layer {

public:
	/**
	 * @details 레이어 클래스 기본 생성자
	 */
	Layer() {}
	/**
	 * @details 레이어 클래스 생성자
	 * @param name 레이어 이름에 대한 문자열 포인터
	 */
	Layer(const string name);
	/**
	 * @details 레이어 클래스 소멸자
	 */
	virtual ~Layer();



	/**
	 * @details 레이어에 부여된 유일한 id값을 조회한다.
	 * @return 레이어 id
	 */
	int getId() const { return id; }
	/**
	 * @biref 레이어의 타입을 조회한다.
	 * @return 레이어 타입
	 */
	LayerType getType() const { return this->type; }
	/**
	 * @details 레이어에 연결된 다음 레이어 목록 벡터를 조회한다.
	 * @return 레이어에 연결된 다음 레이어 목록 벡터
	 */
	vector<next_layer_relation>& getNextLayers() { return this->nextLayers; }
	/**
	 * @details 레이어에 연결된 이전 레이어 목록 벡터를 조회한다.
	 * @return 레이어에 연결된 다음 레이어 목록 벡터
	 */
	vector<prev_layer_relation>& getPrevLayers() { return this->prevLayers; }
	/**
	 * @details 레이어에 연결된 다음 레이어의 수를 조회한다.
	 * @return 레이어에 연결된 다음 레이어의 수
	 */
	int getNextLayerSize() const { return this->nextLayers.size(); }
	/**
	 * @details 레이어에 연결된 이전 레이어의 수를 조회한다.
	 * @return 레이어에 연결된 이전 레이어의 수
	 */
	int getPrevLayerSize() const { return this->prevLayers.size(); }
	/**
	 * @details 레이어의 입력 데이터 구조정보를 담고 있는 구조체를 조회한다.
	 * @return 레이어의 입력 데이터 구조정보를 담고 있는 구조체
	 */
	io_dim getInDimension() const { return in_dim; }
	/**
	 * @details 레이어의 출력 데이터 구조정보를 담고 있는 구조체를 조회한다.
	 * @return 레이어의 출력 데이터 구조정보를 담고 있는 구조체
	 */
	io_dim getOutDimension() const { return out_dim; }
	/**
	 * @details 레이어에 이전 레이어를 추가한다.
	 * @param prevLayer 현재 레이어에 연결할 이전 레이어의 정보 구조체
	 */
	void addPrevLayer(prev_layer_relation prevLayer);
	/**
	 * @details 레이어에 다음 레이어를 추가한다.
	 * @param nextLayer 현재 레이어에 연결할 다음 레이어의 정보 구조체
	 */
	void addNextLayer(next_layer_relation nextLayer);
	/**
	 * @details 현재 레이어에 연결된 마지막 이전 레이어 여부를 조회한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @return 현재 레이어에 연결된 마지막 이전 레이어 여부
	 */
	bool isLastPrevLayerRequest(UINT idx);
	/**
	 * @details 현재 레이어에 연결된 마지막 다음 레이어 여부를 조회한다.
	 * @param idx 현재 레이어에 연결된 다음 레이어의 순번 index
	 * @return 현재 레이어에 연결된 마지막 다음 레이어 여부
	 */
	bool isLastNextLayerRequest(UINT idx);
	/**
	 * @details batch단위로 누적된 gradient를 초기화하고 다음 레이어들에 대해 reset_nabla()를 요청한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @todo GPU_MODE에서 사용하지 않는다.
	 */
	virtual void reset_nabla(UINT idx);
	/**
	 * @details 계산된 gradient를 각 학습레이어에서 갱신하고 다음 레이어들에 대해 update()를 요청한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @param n 학습 데이터의 수
	 * @param miniBatchSize 학습에 사용한 batch 사이즈
	 */
	virtual void update(UINT idx, UINT n, UINT miniBatchSize);



	/**
	 * @details 현재 레이어가 찾으려는 레이어인지 확인하고 아닌 경우 다음 레이어들에 대해 find()를 요청한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @param name 찾고자 하는 레이어의 이름
	 * @return 찾은 레이어에 대한 포인터, 해당하는 레이어가 없는 경우 0
	 */
	virtual Layer* find(UINT idx, const string name);
	/**
	 * @details 현재 레이어를 스트림에 쓰고 다음 레이어들에 대해 save()를 요청한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @param ofs 레이어를 쓸 출력 스트림
	 */
	virtual void save(UINT idx, ofstream &ofs);
	/**
	 * @details 현재 레이어의 메타정보를 스트림의 헤더에 쓰고 다음 레이어들에 대해 saveHeader()를 요청한다.
	 *          입력 레이어 또는 내부 레이어가 있는 레이어(e.g 인셉션레이어)에서 사용한다.
	 * @param  idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @param ofs 레이어를 쓸 출력 스트림
	 */
	virtual void saveHeader(UINT idx, ofstream &ofs);
	/**
	 * @details 현재 레이어를 스트림으로부터 읽어 들이고 다음 레이어들에 대해 load()를 요청한다.
	 * @param ifs 레이어를 읽어들일 입력 스트림
	 * @param layerMap
	 */
	virtual void load(ifstream &ifs, map<Layer *, Layer *> &layerMap);
	/**
	 * @details 현재 레이어의 입력/출력 데이터 구조정보에 의존성이 있는 자료구조들을 구성하고 초기화하고
	 *          다음 레이어들에 대해 shape()를 요청한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @param in_dim 현재 레이어의 입력 데이터 구조정보
	 */
	virtual void shape(UINT idx, io_dim in_dim);
	/**
	 * @details 이미 shape가 구성된 레이어의 shape를 변경하고 다음 레이어들에 대해 reshape()를 요청한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @param in_dim 새롭게 변경할 현재 레이어의 입력 데이터 구조정보
	 */
	virtual void reshape(UINT idx, io_dim in_dim);
	/**
	 * @details 입/출력 데이터 구조정보에 의존성이 있는 자료구조들을 clear하고 다음 레이어들에 대해 clearShape()를 요청한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 */
	virtual void clearShape(UINT idx);

	/**
	 * @details 학습하는 레이어인지 여부를 조회한다.
	 * @return 학습하는 레이어인지 여부
	 */
	virtual bool isLearnable() { return false; }
	/**
	 * @details 학습 파라미터(weight, bias등)의 gradient에 대한 square sum값을 구하고
	 *          다음 레이어들에 대해 sumSquareParam()을 요청한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @return 학습 파라미터 gradient에 대한 square sum값
	 */
	virtual DATATYPE sumSquareParam(UINT idx);
	/**
	 * @details 학습 파라미터(weight, bias등)에 대한 square sum값을 구하고 다음 레이어들에 대해 sumSquareParam2()를 요청한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @return 학습 파리미터에 대한 square sum값
	 */
	virtual DATATYPE sumSquareParam2(UINT idx);
	/**
	 * @details 학습 파라미터의 gradient를 스케일링하고 다음 레이어들에 대해 scaleParam()을 요청한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @param 학습 파라미터 스케일링 팩터
	 */
	virtual void scaleParam(UINT idx, DATATYPE scale_factor);

#ifndef GPU_MODE
public:
	Layer(const string name, int n_in, int n_out);

	rcube &getInput() { return this->input; }
	rcube &getOutput() { return this->output; }

	/**
	 * 주어진 입력 input에 대해 출력 activation을 계산
	 * @param input: 레이어 입력 데이터 (이전 레이어의 activation)
	 */
	// sub class에서 구현이 없을 때에만 참조, 구현이 있을 경우 prop*() 함수를 참조
	virtual void feedforward(UINT idx, const rcube &input, const char *end=0);
#else
public:
	/**
	 * @details 레이어의 입력값 장치 포인터를 조회한다.
	 * @return 레이어 입력값 장치 포인터
	 */
	const DATATYPE *getInput() { return this->d_input; }
	/**
	 * @details 레이어의 출력값 장치 포인터를 조회한다.
	 * @return 레이어 출력값 장치 포인터
	 */
	virtual DATATYPE *getOutput() { return this->d_output; }
	/**
	 * @details 레이어 입력값을 전달받아 출력값을 계산한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @param input 현재 레이어에 전달된 레이어 입력값 장치 포인터
	 * @param end feedforward 종료 레이어 이름, 0인 경우 계속 진행
	 */
	virtual void feedforward(UINT idx, const DATATYPE *input, const char *end=0);
#endif

protected:
	/**
	 * @details 레이어를 초기화한다.
	 * @param name 레이어의 이름 문자열 포인터
	 */
	void initialize(const string name);
	/**
	 * @details 레이어 메타정보로부터 레이어 구성을 로드한다.
	 * @param ifs 레이어를 읽어들일 입력 스트림
	 * @param layerMap 레이어 메타정보로부터 읽어 레이어 주소를 키, 레이어를 값으로 생성한 레이어 맵
	 */
	virtual void loadNetwork(ifstream &ifs, map<Layer *, Layer *> &layerMap);
	/**
	 * @details 현재 레이어 로드 후 이전/다음 레이어 포인터 벡터에 로드된 레이어 포인터 값을 키로하여
	 *          레이어맵의 실제 레이어 객체를 찾아서 이전/다음 레이어에 연결한다.
	 * @param layerMap save당시 레이어의 주소를 키, 해당 레이어를 값으로 하는 맵
	 */
	virtual void updateLayerRelation(map<Layer *, Layer *> &layerMap);
	/**
	 * @details 유일한 레이어 아이디를 생성한다.
	 * @return 생성된 레이어 아이디
	 */
	static int generateLayerId();

	/**
	 * @details 현재 레이어를 스트림에 쓴다.
	 * @param ofs 레이어를 쓸 출력 스트림
	 */
	virtual void _save(ofstream &ofs);
	/**
	 * @details 현재 레이어의 입/출력 데이터 구조정보에 의존성이 있는 자료구조들을 구성하고 초기화한다.
	 * @param recursive 상위 레이어에 대해서 _shape()를 재귀적으로 호출할 것인지 여부
	 */
	virtual void _shape(bool recursive=true);
	/**
	 * @details 이미 shape가 구성된 레이어의 shape를 변경한다.
	 */
	virtual void _reshape();
	/**
	 * @details 입/출력 데이터 구조정보에 의존성이 있는 자료구조들을 clear한다.
	 */
	virtual void _clearShape();
	/**
	 * @details 학습 파라미터(weight, bias등)의 gradient에 대한 square sum값을 구한다.
	 * @return 학습 파라미터의 gradient에 대한 square sum값
	 */
	virtual DATATYPE _sumSquareParam();
	/**
	 * @details 학습 파라미터(weight, bias등)에 대한 square sum값을 구한다.
	 * @return 학습 파라미터에 대한 square sum값
	 */
	virtual DATATYPE _sumSquareParam2();
	/**
	 * @details 학습 파라미터의 gradient를 스케일링한다.
	 * @param scale_factor 학습 파라미터의 gradient를 스케일할 팩터
	 */
	virtual void _scaleParam(DATATYPE scale_factor);

	/**
	 * @details 다음 레이어들에 대해 shape() 메쏘드를 호출한다.
	 */
	void propShape();
	/**
	 * @details 다음 레이어들에 대해 reshape() 메쏘드를 호출한다.
	 */
	void propReshape();
	/**
	 * @details 다음 레이어들에 대해 clearShape() 메쏘드를 호출한다.
	 */
	void propClearShape();
	/**
	 * @details 다음 레이어들에 대해 reset_nabla() 메쏘드를 호출한다.
	 */
	void propResetNParam();
	/**
	 * @details 다음 레이어들에 대해 update() 메쏘드를 호출한다.
	 */
	void propUpdate(UINT n, UINT miniBatchSize);
	/**
	 * @details 다음 레이어들에 대해 save() 메쏘드를 호출한다.
	 */
	void propSave(ofstream &ofs);
	/**
	 * @details 다음 레이어들에 대해 sumSquareParam() 메쏘드를 호출한다.
	 * @return 다음 레이어들로부터 계산된 square sum값의 합
	 */
	DATATYPE propSumSquareParam();
	/**
	 * @details 다음 레이어들에 대해 sumSquareParam2() 메쏘드를 호출한다.
	 * @return 다음 레이어들로부터 계산된 square sum값의 합
	 */
	DATATYPE propSumSquareParam2();
	/**
	 * @details 다음 레이어들에 대해 scaleParam() 메쏘드를 호출한다.
	 * @param scale_factor 학습 파라미터의 gradient를 스케일할 팩터
	 */
	void propScaleParam(DATATYPE scale_factor);


	LayerType type;				///< 레이어의 타입
	int id;						///< 레이어의 고유 아이디
	string name;			///< 레이어의 이름

	io_dim in_dim;				///< 레이어의 입력 데이터 구조 정보
	io_dim out_dim;				///< 레이어의 출력 데이터 구조 정보

	vector<prev_layer_relation> prevLayers;			///< 현재 레이어의 이전(입력) 레이어 목록 벡터
	vector<next_layer_relation> nextLayers;			///< 현재 레이어의 다음(출력) 레이어 목록 벡터

	static int layerCount;							///< 레이어의 고유 아이디 생성을 위한 레이어 카운터

#ifndef GPU_MODE
protected:
	void propFeedforward(const rcube output, const char *end=0);

	rcube input;
	rcube output;
#else
protected:
	/**
	 * @details 다음 레이어들에 대해 feedforward() 메쏘드를 호출한다.
	 * @param output 현재 레이어의 출력값 장치 메모리 포인터
	 * @param end feedforward 중단 레이어의 이름 (0인 경우 최종 output레이어가 중단 레이어)
	 */
	void propFeedforward(const DATATYPE *output, const char *end=0);

	const DATATYPE* d_input;			///< 현재 레이어의 입력값 장치 메모리 포인터 (이전 레이어의 출력값 메모리 포인터를 공유)
	DATATYPE *d_output;					///< 현재 레이어의 출력값 장치 메모리 포인터 (고유한 장치 할당 메모리 포인터)

	cudnnTensorDescriptor_t inputTensorDesc;			///< cudnn 입력 데이터(n-D 데이터셋) 구조 정보
	cudnnTensorDescriptor_t outputTensorDesc;			///< cudnn 출력 데이터(n-D 데이터셋) 구조 정보
#endif



};






#endif /* LAYER_LAYER_H_ */































