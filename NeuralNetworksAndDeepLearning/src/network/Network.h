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
#include "NetworkListener.h"
#include "BaseLayer.h"
#include "InputLayer.h"
#include "ALEInputLayer.h"
#include "LayerConfig.h"
#include "Worker.h"
#include "DQNImageLearner.h"

template <typename Dtype> class DataSet;
//template <typename Dtype> class LayersConfig;
template <typename Dtype> class DQNImageLearner;

enum NetworkStatus : int {
	Train = 0,
	Test = 1
};

enum NetworkPhase : int {
	TrainPhase = 0,
	TestPhase = 1
};

enum LRPolicy : int {
	Fixed = 0,
	Step,
	Exp,
	Inv,
	Multistep,
	Poly
};

enum Optimizer : int {
    Momentum = 0,
    Vanilla,
    Nesterov,
    Adagrad,
    RMSprop,
    Adam
};

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
	Network();
	/**
	 * @details Network 생성자
	 * @param networkListener 네트워크 상태 리스너
	 */
	//Network(NetworkListener *networkListener=0);

	/**
	 * @details Network 소멸자
	 */
	virtual ~Network();

    static void init();

	/**
	 * @details run()을 수행한다. 시간을 측정하는 것이 목적.
	 * @param epochs run()을 수행할 최대 epoch
	 */
	void run_with_timer(int epochs);

	/**
	 * @details stochastic gradient descent를 수행한다.
	 * @param epochs run()을 수행할 최대 epoch
	 */
	void run(int epochs);


	/**
	 * @details 네트워크를 파일에 쓴다.
	 * @param filename 네트워크를 쓸 파일의 경로
	 */
	void save();
	/**
	 * @details 네트워크를 파일로부터 읽는다.
	 * @param filename 네트워크를 읽을 파일의 경로
	 */
	void load();

	/**
	 * @details 네트워크 내부의 레이어를 이름으로 찾는다.
	 * @param name 찾을 레이어의 이름
	 * @return 찾은 레이어에 대한 포인터
	 */
	Layer<Dtype>* findLayer(const std::string layerName);


public:
    int                                     getNetworkID() { return this->networkID; }
    static Network<Dtype>*                  getNetworkFromID(int networkID);
    void                                    setLoaded() { this->isLoaded = true; }

private:
    int                                     networkID;
    static std::atomic<int>                 networkIDGen;
    static std::map<int, Network<Dtype>*>   networkIDMap;
    static std::mutex                       networkIDMapMutex;
    bool                                    isLoaded;
};


#endif /* NETWORK_H_ */
