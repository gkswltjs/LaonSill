/**
 * @file Network.h
 * @date 2016/4/20
 * @author jhkim
 * @brief
 * @details
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include "common.h"
#include "Cost.h"
#include "Activation.h"
#include "NetworkListener.h"
#include "Layer.h"
#include "InputLayer.h"
#include "ALEInputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"
#include "LayerConfig.h"
#include "Evaluation.h"
#include "Worker.h"
#include "NetworkConfig.h"
#include "DQNImageLearner.h"

template <typename Dtype> class DataSet;
template <typename Dtype> class LayersConfig;
template <typename Dtype> class DQNImageLearner;

/**
 * @brief 네트워크 기본 클래스
 * @details 실제 작업이 일어나는 링크드 형식의 레이어 목록을 래핑하고 사용자 인터페이스를 
 * 제공한다.
 * @todo sgd 형태의 네트워크를 기본으로 하고 있으나 다양한 형태의 네트워크 최적화 기법을 
 * 적용할 수 있도록 수정되어야 한다.
 *       (SGD, AdaDelta, AdaGrad, Adam, NAG ... )
 *       개별 파라미터를 각각 전달하는 형태이나 Network Param 형태의 구조체를 만들어 한 번에 
 *       전달하도록 수정해야 한다.
 */
template <typename Dtype>
class Network {
public:
	//Network(NetworkParam& networkParam);
	Network(NetworkConfig<Dtype>* networkConfig);
	/**
	 * @details Network 생성자
	 * @param networkListener 네트워크 상태 리스너
	 */
	//Network(NetworkListener *networkListener=0);
	/**
	 * @details Network 생성자
	 * @param inputLayer 입력 레이어
	 * @param outputLayer 출력 레이어
	 * @param dataSet 학습 및 테스트용 데이터셋
	 * @param networkListener 네트워크 상태 리스너
	 */
	//Network(InputLayer *inputLayer, OutputLayer *outputLayer, DataSet *dataSet,
    //        NetworkListener *networkListener);
	/**
	 * @details Network 소멸자
	 */
	virtual ~Network();

    static void init();

	/**
	 * @details 네트워크에 설정된 입력 레이어를 조회한다.
	 * @return 네트워크에 설정된 입력 레이어
	 */
	InputLayer<Dtype> *getInputLayer();

    void setLayersConfig(LayersConfig<Dtype>* layersConfig);
	LayersConfig<Dtype>* getLayersConfig();

	/**
	 * @details sgd()를 수행한다. 시간을 측정하기 위한 임시 함수.
	 * @param epochs sgd를 수행할 최대 epoch
	 */
	void sgd_with_timer(int epochs);




	/**
	 * @details stochastic gradient descent를 수행한다.
	 * @param epochs sgd를 수행할 최대 epoch
	 */
	void sgd(int epochs);
	/**
	 * @details 네트워크의 주어진 테스트 데이터셋으로 네트워크 테스트를 수행한다.
	 */
	void test();


	/**
	 * @details 네트워크 쓰기관련 설정을 한다.
	 * @param savePrefix 네트워크 쓰기 파일의 경로의 prefix
	 */
	void saveConfig(const char *savePrefix);
	/**
	 * @details 네트워크를 파일에 쓴다.
	 * @param filename 네트워크를 쓸 파일의 경로
	 */
	void save();
	/**
	 * @details 네트워크를 파일로부터 읽는다.
	 * @param filename 네트워크를 읽을 파일의 경로
	 */
	void load(const char* filename);
	/**
	 * @details 네트워크의 입력 데이터 구조 정보를 설정한다.
	 * @param in_dim 네트워크의 입력 데이터 구조 정보 구조체
	 */
	//void shape(io_dim in_dim=io_dim(0,0,0,0));
	/**
	 * @details 네트워크가 이미 입력 데이터 구조 정보가 설정된 상태에서 이를 변경한다.
	 * @param in_dim 네트워크의 변경할 입력 데이터 구조 정보 구조체
	 */
	void reshape(io_dim in_dim=io_dim(0,0,0,0));
	/**
	 * @details 네트워크 내부의 레이어를 이름으로 찾는다.
	 * @param name 찾을 레이어의 이름
	 * @return 찾은 레이어에 대한 포인터
	 */
	Layer<Dtype>* findLayer(const std::string name);
	/**
	 * @details 네트워크에 등록된 데이터셋 특정 채널의 평균값을 조회한다.
	 * @param 데이터셋의 조회할 채널 index
	 * @return 조회된 특정 채널의 평균값
	 */
	float getDataSetMean(UINT channel);
	/**
	 * @details 네트워크에 등록된 데이터셋에 각 채널의 평균값을 설정한다.
	 * @param dataSetMean 채널의 평균값을 담고 있는 배열의 포인터
	 */
	void setDataSetMean(float *dataSetMean);

    /**
     * DQN related functions
     */
    std::vector<Data<Dtype>*>& feedForwardDQNNetwork(int batchCount,
        DQNImageLearner<Dtype> *learner);
    void backPropagateDQNNetwork(int batchCount);


protected:
	/**
	 * @details 배치단위의 학습이 종료된 후 학습된 내용을 적절한 정규화 과정을 거쳐 네트워크에
     *          반영한다.
	 * @param nthMiniBatch 한 epoch내에서 종료된 batch의 index
	 */
	void trainBatch(uint32_t batchIndex);
	/**
	 * @details 배치단위의 학습된 내용을 네트워크에 반영한다.
	 */
	void applyUpdate();
	/**
	 * @details 배치단위의 학습된 파라미터의 L2 norm을 설정된 값을 기준으로 스케일 다운한다.
	 *          gradient explode를 예방하는 역할을 한다.
	 */
	void clipGradients();

	double computeSumSquareParamsData();
	double computeSumSquareParamsGrad();
	void scaleParamsGrad(float scale);


	//double totalCost(const std::vector<const DataSample *> &dataSet, double lambda);
	//double accuracy(const std::vector<const DataSample *> &dataSet);
	/**
	 * @details 학습된 네트워크에 대해 전체 테스트셋으로 네트워크를 평가한다.
	 */
	double evaluateTestSet();



	//void checkAbnormalParam();
	//void checkLearnableParamIsNan();



	void _feedforward(uint32_t batchIndex);
	void _backpropagation(uint32_t batchIndex);



#ifndef GPU_MODE
	int testEvaluateResult(const rvec &output, const rvec &y);
#else
	/**
	 * @details 특정 테스트 데이터 하나에 대해 feedforward된 네트워크를 target값으로 평가한다.
	 * @param num_labels 데이터셋 레이블 크기 (카테고리의 수)
	 * @param output 데이터에 대한 네트워크의 출력 장치 메모리 포인터
	 * @param y 데이터의 정답 호스트 메모리 포인터
	 */
	//void evaluateTestData(const int num_labels, Data* output, const UINT *y);
	double evaluateTestData(uint32_t batchIndex);
#endif





public:
	NetworkConfig<Dtype>* config;



protected:

	//DataSet *dataSet;				///< 학습 및 테스트 데이터를 갖고 있는 데이터셋 객체

	//InputLayer *inputLayer;						///< 네트워크 입력 레이어 포인터
	//std::vector<OutputLayer*> outputLayers;		///< 네트워크 출력 레이어 포인터 목록 벡터
	//std::vector<Evaluation<Dtype>*> evaluations;		///< 네트워크 평가 객체 포인터 목록 벡터
	//std::vector<NetworkListener*> networkListeners;	///< 네트워크 이벤트 리스너 객체 포인터 
                                                         //목록 벡터

	//io_dim in_dim;								///< 네트워크 입력 데이터 구조 정보 구조체

	//char savePrefix[200];						///< 네트워크 파일 쓰기 경로 prefix
	//bool saveConfigured;						///< 네트워크 쓰기 설정 여부
	//double maxAccuracy;							///< 네트워크 평가 최대 정확도
	//double minCost;								///< 네트워크 평가 최소 cost
	//float dataSetMean[3];						///< 네트워크 데이터셋 평균 배열



	//uint32_t iterations;

public:
    int                                     getNetworkID() { return this->networkID; }
    static Network<Dtype>*                  getNetworkFromID(int networkID);

private:
    int                                     networkID;
    static std::atomic<int>                 networkIDGen;
    static std::map<int, Network<Dtype>*>   networkIDMap;
    static std::mutex                       networkIDMapMutex;
};


#endif /* NETWORK_H_ */
