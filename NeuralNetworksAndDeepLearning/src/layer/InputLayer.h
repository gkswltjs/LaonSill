/**
 * @file	InputLayer.h
 * @date	2016/5/11
 * @author	jhkim
 * @brief
 * @details
 */


#ifndef LAYER_INPUTLAYER_H_
#define LAYER_INPUTLAYER_H_

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "Layer.h"
#include "LayerConfig.h"

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
	class Builder : public Layer::Builder {
	public:
		Builder() {}
		virtual Builder* name(const string name) {
			Layer::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			Layer::Builder::id(id);
			return this;
		}
		virtual Builder* nextLayerIndices(const vector<uint32_t>& nextLayerIndices) {
			Layer::Builder::nextLayerIndices(nextLayerIndices);
			return this;
		}
		Layer* build() {
			return new InputLayer(this);
		}
	};


	/**
	 * @details InputLayer 기본 생성자
	 *          LayerFactory에서 객체 생성을 위해 name 파라미터가 없는 기본 생성자가 필요하다.
	 */
	InputLayer() {
		initialize();
	}
	/**
	 * @details InputLayer 생성자
	 * @param name 레이어의 이름 문자열
	 */
	InputLayer(const string name) : Layer(name) {
		initialize();
	}
	InputLayer(Builder* builder) : Layer(builder) {
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
	int getInputSize() const {
		return in_dim.rows*in_dim.cols*in_dim.channels;
	}

	using Layer::feedforward;
	void feedforward(const DATATYPE* input, const char* end=0) {
		//_input->set_data(input, Data::HostToDevice);
		_input->set_device_with_host_data(input);

		_feedforward();
		propFeedforward(end);
	}

protected:
	void initialize() {
		this->type = Layer::Input;
	}

	virtual void _shape(bool recursive=true) {
		this->out_dim = in_dim;
		if(recursive) {
			Layer::_shape();
		}
	}
	virtual void _clearShape() {
		Layer::_clearShape();
	}
	/**
	 * @details 현재 레이어를 스트림에 쓰고 다음 레이어들에 대해 save()를 요청한다.
	 *          입력 레이어의 경우 시작레이어이기 때문에 자신의 레이어를 쓸 뿐 아니라
	 *          연결된 이 후의 레이어들의 메타 정보를 기록하는 역할도 한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @param ofs 레이어를 쓸 출력 스트림
	 */
	virtual void _save(ofstream &ofs) {
		saveHeader(0, ofs);
		// header boundary (dummy layer)
		int type = 0;
		Layer *layer = 0;
		ofs.write((char *)&type, sizeof(int));
		ofs.write((char *)&layer, sizeof(Layer *));

		Layer::_save(ofs);
	}
	virtual void _load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
		initialize();
		InputLayer::_shape(false);
		loadNetwork(ifs, layerMap);
	}

};





#endif /* LAYER_INPUTLAYER_H_ */
