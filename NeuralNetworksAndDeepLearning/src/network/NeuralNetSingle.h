/**
 * @file NeuralNetSingle.h
 * @date 2016/6/2
 * @author jhkim
 * @brief
 * @details
 */

#ifndef NETWORK_NEURALNETSINGLE_H_
#define NETWORK_NEURALNETSINGLE_H_



/**
 * @brief 하나의 FullyConnectedLayer를 가진 Network를 구현한 클래스
 */
class NeuralNetSingle : public Network {
public:
	NeuralNetSingle(NetworkListener *networkListener=0, double lr_mult=0.1, double decay_mult=5.0) : Network(networkListener) {

		InputLayer *inputLayer = new InputLayer(
				"input"
				//io_dim(28, 28, 1, batchSize)
				);

		HiddenLayer *fc1Layer = new FullyConnectedLayer(
				"fc1",
				//io_dim(28*28*1, 1, 1, batchSize),
				//io_dim(100, 1, 1, batchSize),
				100,
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Xavier),
				param_filler(ParamFillerType::Gaussian, 1),
				ActivationType::ReLU
				);

		OutputLayer *softmaxLayer = new SoftmaxLayer(
				"softmax",
				//io_dim(100, 1, 1, batchSize),
				//io_dim(10, 1, 1, batchSize),
				10,
				0.5,
				update_param(lr_mult, decay_mult),
				update_param(lr_mult, decay_mult),
				param_filler(ParamFillerType::Constant, 0),
				param_filler(ParamFillerType::Gaussian, 1)
				);

		Network::addLayerRelation(inputLayer, fc1Layer);
		Network::addLayerRelation(fc1Layer, softmaxLayer);

		this->inputLayer = inputLayer;
		addOutputLayer(softmaxLayer);
	}
	virtual ~NeuralNetSingle() {}
};



#endif /* NETWORK_NEURALNETSINGLE_H_ */
