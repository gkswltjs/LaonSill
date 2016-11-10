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
//#define PRINT_CALLSTACK




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
	enum Type: int {
		None=0,
		Input, 					// 입력 레이어
		FullyConnected, 		// Fully Connected 레이어
		Conv, 					// 컨볼루션 레이어
		Pool, 					// 풀링 레이어
		DepthConcat,			// Depth Concat 레이어
		Inception, 				// 인셉션 레이어
		LRN,					// Local Response Normaization 레이어
		Sigmoid, 				// 시그모이드 레이어
		Softmax					// 소프트맥스 레이어
	};


	/**
	 * @brief 레이어 객체 빌더
	 * @details 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 레이어 객체를 생성한다.
	 */
	class Builder {
	public:
		Type type;
        std::string _name;							///< 레이어의 이름
		uint32_t _id;							///< 레이어의 아이디
        std::vector<uint32_t> _nextLayerIndices;		///< 레이어의 다음 레이어 아이디 목록

		Builder() {
			type = Layer<Dtype>::None;
			_name = "";
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
		virtual Builder* nextLayerIndices(const std::vector<uint32_t>& nextLayerIndices) {
			this->_nextLayerIndices = nextLayerIndices;
			return this;
		}
		virtual Layer<Dtype>* build() = 0;
		virtual void save(std::ofstream& ofs) {
			ofs.write((char*)&type, sizeof(uint32_t));

			size_t nameLength = _name.size();
			ofs.write((char*)&nameLength, sizeof(size_t));
			ofs.write((char*)_name.c_str(), nameLength);

			ofs.write((char*)&_id, sizeof(uint32_t));
			uint32_t numNextLayerIndices = _nextLayerIndices.size();
			ofs.write((char*)&numNextLayerIndices, sizeof(uint32_t));
			for(uint32_t i = 0; i < numNextLayerIndices; i++) {
				ofs.write((char*)&_nextLayerIndices[i], sizeof(uint32_t));
			}
		}
		virtual void load(std::ifstream& ifs) {

			size_t nameLength;
			ifs.read((char*)&nameLength, sizeof(size_t));
			char* name_c = new char[nameLength+1];
			ifs.read(name_c, nameLength);
			name_c[nameLength] = '\0';
			_name = name_c;
			delete [] name_c;

			ifs.read((char*)&_id, sizeof(uint32_t));
			uint32_t numNextLayersIndices;
			ifs.read((char*)&numNextLayersIndices, sizeof(uint32_t));
			for(uint32_t i = 0; i < numNextLayersIndices; i++) {
				uint32_t nextLayerIndice;
				ifs.read((char*)&nextLayerIndice, sizeof(uint32_t));
				_nextLayerIndices.push_back(nextLayerIndice);
			}
		}
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
	Layer(const std::string name);
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
	typename Layer<Dtype>::Type getType() const { return this->type; }
	/**
	 * @details 레이어에 연결된 다음 레이어 목록 벡터를 조회한다.
	 * @return 레이어에 연결된 다음 레이어 목록 벡터
	 */
    std::vector<Layer<Dtype>*>& getNextLayers() { return this->nextLayers; }
	/**
	 * @details 레이어에 연결된 이전 레이어 목록 벡터를 조회한다.
	 * @return 레이어에 연결된 다음 레이어 목록 벡터
	 */
    std::vector<Layer<Dtype>*>& getPrevLayers() { return this->prevLayers; }
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
	io_dim getInDimension() const { return this->in_dim; }
	/**
	 * @details 레이어의 출력 데이터 구조정보를 담고 있는 구조체를 조회한다.
	 * @return 레이어의 출력 데이터 구조정보를 담고 있는 구조체
	 */
	io_dim getOutDimension() const { return this->out_dim; }
	/**
	 * @details 레이어의 입력값 장치 포인터를 조회한다.
	 * @return 레이어 입력값 장치 포인터
	 */
	virtual Data<Dtype>* getInput() { return this->_input.get(); }
	/**
	 * @details 레이어의 출력값 장치 포인터를 조회한다.
	 * @return 레이어 출력값 장치 포인터
	 */
	virtual Data<Dtype>* getOutput() { return this->_output.get(); }

	/**
	 * @details 레이어에 네트워크 설정값을 설정한다.
	 * @param networkConfig 네트워크 설정값 객체
	 */
	virtual void setNetworkConfig(NetworkConfig<Dtype>* networkConfig) { this->networkConfig = networkConfig; }




	////////////////////////////////////////////////////////////////////
	//
	////////////////////////////////////////////////////////////////////
	/**
	 * @details 레이어에 이전 레이어를 추가한다.
	 * @param prevLayer 현재 레이어에 연결할 이전 레이어의 정보 구조체
	 */
	void addPrevLayer(Layer<Dtype>* prevLayer);
	/**
	 * @details 레이어에 다음 레이어를 추가한다.
	 * @param nextLayer 현재 레이어에 연결할 다음 레이어의 정보 구조체
	 */
	void addNextLayer(Layer<Dtype>* nextLayer);
	/**
	 * @details 현재 레이어에 연결된 첫 이전 레이어 여부를 조회한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어 아이디
	 * @return 현재 레이어에 연결된 첫 이전 레이어 여부
	 */
	bool isFirstPrevLayerRequest(uint32_t idx);
	/**
	 * @details 현재 레이어에 연결된 마지막 이전 레이어 여부를 조회한다.
	 * @param idx 요청을 보낸 이전 레이어의 id
	 * @return 현재 레이어에 연결된 마지막 이전 레이어 여부
	 */
	bool isLastPrevLayerRequest(uint32_t idx);
	/**
	 * @details isLastPrevLayerRequest()의 Logging버전 (임시 method)
	 *          어떤 레이어의 어떤 method에 대해 테스트하고 있는지,
	 *          테스트를 통과했는지 통과하지 못했는지를 화면에 출력한다.
	 * @param idx 요청을 보낸 이전 레이어의 id
	 * @param method 이 method를 호출한 method의 이름
	 * @return 현재 레이어에 연결된 마지막 이전 레이어 여부
	 */
	bool w_isLastPrevLayerRequest(uint32_t idx, const std::string method);
	/**
	 * @details 현재 레이어에 연결된 첫 다음 레이어 여부를 조회한다.
	 * @param idx 현재 레이어에 연결된 다음 레이어 아이디
	 * @return 현재 레이어에 연결된 첫 다음 레이어 여부
	 */
	bool isFirstNextLayerRequest(uint32_t idx);
	/**
	 * @details 현재 레이어에 연결된 마지막 다음 레이어 여부를 조회한다.
	 * @param idx 요청을 보낸 다음 레이어의 id
	 * @return 현재 레이어에 연결된 마지막 다음 레이어 여부
	 */
	bool isLastNextLayerRequest(uint32_t idx);
	/**
	 * @details isLastNextLayerRequest()의 Logging버전 (임시 method)
	 *          어떤 레이어의 어떤 method에 대해 테스트하고 있는지,
	 *          테스트를 통과했는지 통과하지 못했는지를 화면에 출력한다.
	 * @param idx 요청을 보낸 다음 레이어의 id
	 */
	bool w_isLastNextLayerRequest(uint32_t idx, const std::string method);

	/**
	 * @details 현재 레이어의 입력/출력 데이터 구조정보에 의존성이 있는 자료구조들을 구성하고 초기화하고
	 *          다음 레이어들에 대해 shape()를 요청한다.
	 * @param idx 요청을 보낸 이전 레이어의 id
	 * @param in_dim 현재 레이어의 입력 데이터 구조정보
	 */
	virtual void shape(uint32_t idx, io_dim in_dim, std::shared_ptr<Data<Dtype>>& prevLayerOutput);
	/**
	 * @details 이미 shape가 구성된 레이어의 shape를 변경하고 다음 레이어들에 대해 reshape()를 요청한다.
	 * @param idx 요청을 보낸 이전 레이어의 id
	 * @param in_dim 새롭게 변경할 현재 레이어의 입력 데이터 구조정보
	 */
	virtual void reshape(uint32_t idx, io_dim in_dim);
	/**
	 * @details 입/출력 데이터 구조정보에 의존성이 있는 자료구조들을 clear하고 다음 레이어들에 대해 clearShape()를 요청한다.
	 * @param idx 요청을 보낸 이전 레이어의 id
	 */
	virtual void clearShape(uint32_t idx);
	/**
	 * @details 현재 레이어를 스트림에 쓰고 다음 레이어들에 대해 save()를 요청한다.
	 * @param idx 요청을 보낸 이전 레이어의 id
	 * @param ofs 레이어를 쓸 출력 스트림
	 */
	//virtual void save(uint32_t idx, std:ofstream &ofs);
	/**
	 * @details 현재 레이어의 메타정보를 스트림의 헤더에 쓰고 다음 레이어들에 대해 saveHeader()를 요청한다.
	 *          입력 레이어 또는 내부 레이어가 있는 레이어(e.g 인셉션레이어)에서 사용한다.
	 * @param  idx 요청을 보낸 이전 레이어의 id
	 * @param ofs 레이어를 쓸 출력 스트림
	 */
	//virtual void saveHeader(uint32_t idx, std:ofstream &ofs);
	/**
	 * @details 현재 레이어를 스트림으로부터 읽어 들여 복구한다.
	 *          - 레이어의 상속받은 상위 클래스 영역 읽기 및 초기화 (_shape() 포함, Layer 클래스 제외)
	 *          - 현재 클래스 영역 읽기 및 초기화 (_shape() 포함)
	 *          읽기에 대해서 최초 레이어 (입력 레이어)에서 글로벌하게 진행.
	 * @param ifs 레이어를 읽어들일 입력 스트림
	 * @param layerMap
	 */
	//virtual void load(std::ifstream &ifs, std::map<Layer<Dtype>*, Layer<Dtype>*> &layerMap);

#ifndef GPU_MODE
	/**
	 * @details batch단위로 누적된 gradient를 초기화하고 다음 레이어들에 대해 reset_nabla()를 요청한다.
	 * @param idx 요청을 보낸 이전 레이어의 id
	 * @todo GPU_MODE에서 사용하지 않는다.
	 */
	virtual void reset_nabla(uint32_t idx);
#else
	/**
	 * @details 레이어 입력값을 전달받아 출력값을 계산하고 다음 레이어들에 대해 feedforward()를 요청한다.
	 * @param idx 요청을 보낸 이전 레이어의 id
	 * @param input 현재 레이어에 전달된 레이어 입력값 장치 포인터
	 * @param end feedforward 종료 레이어 이름, 0인 경우 계속 진행
	 */
	virtual void feedforward(uint32_t idx, Data<Dtype>* input, const char *end=0);
#endif





protected:
	/**
	 * @details 레이어를 초기화한다.
	 * @param name 레이어의 이름 문자열 포인터
	 */
	void initialize(uint32_t id, const std::string name);
	/**
	 * @details 레이어 메타정보로부터 레이어 구성을 로드한다.
	 * @param ifs 레이어를 읽어들일 입력 스트림
	 * @param layerMap 레이어 메타정보로부터 읽어 레이어 주소를 키, 레이어를 값으로 생성한 레이어 맵
	 */
	//virtual void loadNetwork(std::ifstream &ifs, std::map<Layer<Dtype>*, Layer<Dtype>*> &layerMap);
	/**
	 * @details 현재 레이어 로드 후 이전/다음 레이어 포인터 벡터에 로드된 레이어 포인터 값을 키로하여
	 *          레이어맵의 실제 레이어 객체를 찾아서 이전/다음 레이어에 연결한다.
	 * @param layerMap save당시 레이어의 주소를 키, 해당 레이어를 값으로 하는 맵
	 */
	//virtual void updateLayerRelation(std::map<Layer<Dtype>*, Layer<Dtype>*> &layerMap);



	bool isSharedInput();
	bool isSharedOutput();





	////////////////////////////////////////////////////////////////////
	// prop 계열의 method (레이어 연결을 따라 연쇄 호출되는 method) 들에 대해
	// 각 레이어의 실제 작업을 담당하는 method들
	////////////////////////////////////////////////////////////////////
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
	 * @details 현재 레이어를 스트림에 쓴다.
	 * @param ofs 레이어를 쓸 출력 스트림
	 */
	//virtual void _save(std:ofstream &ofs);
	/**
	 * @details 현재 레이어를 스트림으로부터 읽어 들여 복구한다.
	 *          - 레이어의 상속받은 상위 클래스 영역 읽기 및 초기화 (_shape() 포함, Layer 클래스 제외)
	 *          - 현재 클래스 영역 읽기 및 초기화 (_shape() 포함)
	 *          읽기에 대해서 최초 레이어 (입력 레이어)에서 글로벌하게 진행.
	 * @param ifs 레이어를 읽어들일 입력 스트림
	 * @param layerMap
	 */
	//virtual void _load(std::ifstream &ifs, std::map<Layer<Dtype>*, Layer<Dtype>*> &layerMap);
	/**
	 * @details 레이어 입력값을 전달받아 출력값을 계산한다.
	 */
	virtual void _feedforward();
	/**
	 * @details 복수의 '이전' 레이어로부터의 입력을 조합한다.
	 *          조합은 입력의 합으로 한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 idx
	 * @param input 현재 레이어에 전달된 레이어 입력값 장치 포인터
	 */
	virtual void _concat(uint32_t idx, Data<Dtype>* input);
	/**
	 * @details 복수의 '이전' 레이어로부터의 입력들에 대해 branch의 수 기준으로 스케일링한다.
	 *          _concat()이 입력 합산이 아닌 방식으로 구현된 경우 _scaleInput() 역시 적절히 재정의해야 한다.
	 */
	virtual void _scaleInput();









	////////////////////////////////////////////////////////////////////
	// 이전, 이후 레이어로의 method 호출을 담당하는 호출 propagation method들
	////////////////////////////////////////////////////////////////////

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
	 * @details 다음 레이어들에 대해 save() 메쏘드를 호출한다.
	 */
	//void propSave(std:ofstream &ofs);

#ifndef GPU_MODE
	/**
	 * @details 다음 레이어들에 대해 reset_nabla() 메쏘드를 호출한다.
	 */
	void propResetNParam();
#else
	/**
	 * @details 다음 레이어들에 대해 feedforward() 메쏘드를 호출한다.
	 * @param end feedforward 중단 레이어의 이름 (0인 경우 최종 output레이어가 중단 레이어)
	 */
	void propFeedforward(const char *end=0);
#endif



protected:
	Layer<Dtype>::Type type;							///< 레이어의 타입
	int id;												///< 레이어의 고유 아이디
    std::string name;										///< 레이어의 이름

	NetworkConfig<Dtype>* networkConfig;				///< 레이어가 속한 네트워크의 설정

	io_dim in_dim;										///< 레이어의 입력 데이터 구조 정보
	io_dim out_dim;										///< 레이어의 출력 데이터 구조 정보

    std::vector<Layer<Dtype>*> prevLayers;					///< 현재 레이어의 이전(입력) 레이어 목록 벡터
    std::vector<Layer<Dtype>*> nextLayers;					///< 현재 레이어의 다음(출력) 레이어 목록 벡터



#ifndef GPU_MODE
#else
	cudnnTensorDescriptor_t inputTensorDesc;			///< cudnn 입력 데이터(n-D 데이터셋) 구조 정보
	cudnnTensorDescriptor_t outputTensorDesc;			///< cudnn 출력 데이터(n-D 데이터셋) 구조 정보
#endif

	static const int LAYER_NAME_LENGTH = 32;






public:
	//Data<Dtype>* _input;								///< 레이어 입력 데이터 및 그레디언트
	//Data<Dtype>* _output;								///< 레이어 출력 데이터 및 그레디언트

    std::shared_ptr<Data<Dtype>> _input;
    std::shared_ptr<Data<Dtype>> _output;
};




#endif /* LAYER_LAYER_H_ */































