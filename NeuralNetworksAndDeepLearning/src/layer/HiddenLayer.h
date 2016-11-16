/**
 * @file	HiddenLayer.h
 * @date	2016/5/11
 * @author	jhkim
 * @brief
 * @details
 */

#ifndef LAYER_HIDDENLAYER_H_
#define LAYER_HIDDENLAYER_H_

#include "common.h"
#include "Layer.h"
//#include "../network/NetworkConfig.h"


/**
 * @brief 히든 레이어 기본 추상 클래스
 * @details 기본 레이어의 클래스에 backpropagation, parameter update와 같은
 *          파라미터 학습 관련 기능을 추가한다.
 */
template <typename Dtype>
class HiddenLayer : public Layer<Dtype> {
public:
	/**
	 * @brief 히든 레이어 객체 빌더
	 * @details 히든 레이어를 생성할 때 필요한 파라미터들을 설정하고 build()를 통해
	 *          해당 파라미터를 만족하는 히든 레이어 객체를 생성한다.
	 */
	class Builder : public Layer<Dtype>::Builder {
	public:
        std::vector<uint32_t> _prevLayerIndices;

		Builder() {}
		virtual Builder* name(const std::string name) {
			Layer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			Layer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			Layer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			Layer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		Layer<Dtype>* build() = 0;
		virtual void save(std::ofstream& ofs) {
			Layer<Dtype>::Builder::save(ofs);
			uint32_t numPrevLayerIndices = _prevLayerIndices.size();
			ofs.write((char*)&numPrevLayerIndices, sizeof(uint32_t));
			for(uint32_t i = 0; i < numPrevLayerIndices; i++) {
				ofs.write((char*)&_prevLayerIndices[i], sizeof(uint32_t));
			}
		}
		virtual void load(std::ifstream& ifs) {
			Layer<Dtype>::Builder::load(ifs);
			uint32_t numPrevLayerIndices;
			ifs.read((char*)&numPrevLayerIndices, sizeof(uint32_t));
			for(uint32_t i = 0; i < numPrevLayerIndices; i++) {
				uint32_t prevLayerIndice;
				ifs.read((char*)&prevLayerIndice, sizeof(uint32_t));
				_prevLayerIndices.push_back(prevLayerIndice);
			}
		}
	};


	HiddenLayer();
	HiddenLayer(Builder* builder);
	HiddenLayer(const std::string& name);
	virtual ~HiddenLayer();


	/**
	 * @details 네트워크 cost의 다음 레이어의 입력에 관한 gradient값을 전달 받아
	 *          현재 레이어의 parameter(parameter가 있는 경우), input에 관한 gradient를 계산하고
	 *          이전 레이어에 현재 레이어의 input에 관한 gradient값을 전달한다.
	 * @param idx 현재 레이어에 연결된 다음 레이어의 순번 index
	 * @param next_delta_input 네트워크 cost의 다음 레이어의 입력에 관한 gradient 장치 메모리 포인터
	 */
	//virtual void backpropagation(uint32_t idx, Data<Dtype>* next_input, uint32_t offset);
	virtual void backpropagation();

	virtual void _backpropagation();

	virtual void shape();

protected:
	/**
	 * @details 네트워크 cost의 다음 레이어의 입력에 관한 gradient값을 전달 받아
	 *          현재 레이어의 parameter(parameter가 있는 경우), input에 관한 gradient를 계산한다.
	 */


	//virtual void _shape(bool recursive=true);
	virtual void _clearShape();

	/**
	 * @details 복수의 '다음' 레이어로부터의 gradient를 조합한다.
	 *          기본 조합은 gradient의 합으로 한다.
	 * @param idx 현재 레이어에 연결된 다음 레이어의 순번 index
	 * @param next_delta_input 네트워크 cost의 다음 레이어의 입력에 관한 gradient 장치 메모리 포인터
	 */
	//virtual void _deconcat(uint32_t idx, Data<Dtype>* next_delta_input, uint32_t offset);

	/**
	 * @details 복수의 '다음' 레이어로부터의 gradient들에 대해 branch의 수 기준으로 스케일링한다.
	 *          _deconcat()이 gradient합산이 아닌 방식으로 구현된 경우 _scaleGradient() 역시 적절히 재정의해야 한다.
	 */
	//virtual void _scaleGradient();

	/**
	 * @details 이전 레이어들에 대해 backpropagation() 메쏘드를 호출한다.
	 */
	//virtual void propBackpropagation();
};



#endif /* LAYER_HIDDENLAYER_H_ */



















