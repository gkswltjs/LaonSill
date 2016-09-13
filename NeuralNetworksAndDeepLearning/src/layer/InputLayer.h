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

#include "../Util.h"
#include "Layer.h"
#include "../dataset/DataSet.h"

/**
 * @brief 입력 레이어 클래스
 * @details 입력 데이터를 그대로 출력 데이터로 전달하는 역할을 한다.
 *          특별한 기능을 담당하지 않고 입력 데이터를 한 레벨 추상화하고
 *          약간의 레어어 쓰기, 읽기 등의 부가 기능을 수행
 *          입력 레이어의 경우 자신의 레이어값 읽기, 쓰기뿐 아니라 최초의 레이어로써 뒤에 연결된 모든 레이어의
 *          메타 정보를 읽기, 쓰기를 수행한다.
 */
template <typename Dtype>
class InputLayer : public Layer<Dtype> {
public:
	/**
	 * @brief 입력 레이어 객체 빌더
	 * @details 입력 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 레이어 입력 객체를 생성한다.
	 */
	class Builder : public Layer<Dtype>::Builder {
	public:
		Builder() {}
		virtual Builder* name(const string name) {
			Layer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			Layer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* nextLayerIndices(const vector<uint32_t>& nextLayerIndices) {
			Layer<Dtype>::Builder::nextLayerIndices(nextLayerIndices);
			return this;
		}
		Layer<Dtype>* build() {
			return new InputLayer(this);
		}
	};



	/**
	 * @details InputLayer 기본 생성자
	 *          LayerFactory에서 객체 생성을 위해 name 파라미터가 없는 기본 생성자가 필요하다.
	 */
	InputLayer();
	/**
	 * @details InputLayer 생성자
	 * @param name 레이어의 이름 문자열
	 */
	InputLayer(const string name);
	InputLayer(Builder* builder);
	/**
	 * @details InputLayer 소멸자
	 */
	virtual ~InputLayer();


	/**
	 * @details batch size 1 기준의 입력 엘리먼트의 갯수를 조회한다.
	 * @return batch size 1 기준의 입력 엘리먼트의 갯수
	 */
	int getInputSize() const;

	/*
	using Layer<Dtype>::feedforward;
	void feedforward(const Dtype* input, const char* end=0) {
		//_input->set_data(input, Data::HostToDevice);
		this->_input->set_device_with_host_data(input);

		this->_feedforward();
		this->propFeedforward(end);
	}
	*/

	using Layer<Dtype>::feedforward;
	void feedforward(DataSet<Dtype>* dataSet, const uint32_t baseIndex, const char* end=0);

protected:
	void initialize();
	virtual void _shape(bool recursive=true);
	virtual void _clearShape();
	/**
	 * @details 현재 레이어를 스트림에 쓰고 다음 레이어들에 대해 save()를 요청한다.
	 *          입력 레이어의 경우 시작레이어이기 때문에 자신의 레이어를 쓸 뿐 아니라
	 *          연결된 이 후의 레이어들의 메타 정보를 기록하는 역할도 한다.
	 * @param idx 현재 레이어에 연결된 이전 레이어의 순번 index
	 * @param ofs 레이어를 쓸 출력 스트림
	 */
	virtual void _save(ofstream &ofs);
	virtual void _load(ifstream& ifs, map<Layer<Dtype>*, Layer<Dtype>*>& layerMap);

};





#endif /* LAYER_INPUTLAYER_H_ */
