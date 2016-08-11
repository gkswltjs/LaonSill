/**
 * @file	InputLayer.h
 * @date	2016/5/11
 * @author	jhkim
 * @brief
 * @details
 */


#ifndef LAYER_INPUTLAYER_H_
#define LAYER_INPUTLAYER_H_

#include "Layer.h"
#include "LayerConfig.h"
#include "../Util.h"
#include "../exception/Exception.h"
#include <armadillo>

using namespace arma;


/**
 * @brief 입력 레이어 클래스
 * @details 입력 데이터를 그대로 출력 데이터로 전달하는 역할을 한다.
 *          특별한 기능을 담당하지 않고 입력 데이터를 한 레벨 추상화하고
 *          약간의 레어어 쓰기, 읽기 등의 부가 기능을 수행
 *          입력 레이어의 경우 자신의 레이어값 읽기, 쓰기뿐 아니라 최초의 레이어로써 뒤에 연결된 모든 레이어의
 *          메타 정보를 읽기, 쓰기를 수행한다.
 */
class InputLayer : public Layer {
public:
	/**
	 * @details InputLayer 기본 생성자
	 */
	InputLayer() { this->type = LayerType::Input; }
	/**
	 * @details InputLayer 생성자
	 * @param name 레이어의 이름 문자열 포인터
	 */
	InputLayer(const string name) : Layer(name) {
		initialize();
	}
	/**
	 * @details InputLayer 소멸자
	 */
	virtual ~InputLayer() {}

	/**
	 * @details batch size 1 기준의 입력 엘리먼트의 갯수를 조회한다.
	 * @return batch size 1 기준의 입력 엘리먼트의 갯수
	 */
	int getInputDimension() const { return in_dim.rows*in_dim.cols*in_dim.channels; }

	virtual void save(UINT idx, ofstream &ofs) {
		saveHeader(0, ofs);

		// header boundary (dummy layer)
		int type = 0;
		Layer *layer = 0;
		ofs.write((char *)&type, sizeof(int));
		ofs.write((char *)&layer, sizeof(Layer *));

		Layer::_save(ofs);
		propSave(ofs);
	}

	virtual void load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
		initialize();
		InputLayer::_shape(false);
		loadNetwork(ifs, layerMap);
	}

#ifndef GPU_MODE
public:
	InputLayer(const string name, int n_in) : Layer(name, n_in, n_in) {
		initialize();
	}
	/**
	 * Input 무조건 첫번째 layer,
	 * feedforward로 들어오는 input외의 input에 대해서는 고려하지 않음
	 */
	void feedforward(UINT idx, const rcube &input, const char *end=0) {
		//if(!isLastPrevLayerRequest(idx)) throw Exception();

		Util::convertCube(input, this->input);
		Util::convertCube(this->input, this->output);
		Util::printCube(input, "input:");
		Util::printCube(this->output, "output:");

		propFeedforward(this->output, end);
	}

#else
public:
	void feedforward(UINT idx, const DATATYPE *input, const char *end=0) {
		Util::printMessage("InputLayer::feedforward()---"+string(name));
		if(!isLastPrevLayerRequest(idx)) throw Exception();

		Cuda::refresh();

		this->d_input = input;
		//Util::printDeviceData(d_input, in_dim.rows, in_dim.batches, 1, 1, "d_input:");

		checkCudaErrors(cudaMemcpyAsync(this->d_output, this->d_input, sizeof(DATATYPE)*in_dim.batchsize(), cudaMemcpyDeviceToDevice));

		//float norm = 0.0f;
		//checkCudaErrors(cublasSnrm2(Cuda::cublasHandle, static_cast<int>(in_dim.batchsize()), this->d_output, 1, &norm));
		//norm = 1.0f/norm;
		//checkCudaErrors(cublasSscal(Cuda::cublasHandle, static_cast<int>(in_dim.batchsize()), &norm, this->d_output, 1));

		//exit(1);

		propFeedforward(this->d_output, end);
	}
#endif




protected:
	void initialize() {
		this->type = LayerType::Input;
	}

#ifndef GPU_MODE
protected:
#else
protected:
	virtual void _shape(bool recursive=true) {
		this->out_dim = in_dim;
		if(recursive) {
			Layer::_shape();
		}
	}

	virtual void _clearShape() {
		Layer::_clearShape();
	}

#endif




};





#endif /* LAYER_INPUTLAYER_H_ */
