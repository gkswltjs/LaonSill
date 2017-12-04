/*
 * LearnableLayer.cpp
 *
 *  Created on: 2017. 2. 21.
 *      Author: jhkim
 */

#include "LearnableLayer.h"
#include "SysLog.h"

using namespace std;

template<typename Dtype>
LearnableLayer<Dtype>::LearnableLayer() : Layer<Dtype>() {
}

template <typename Dtype>
double LearnableLayer<Dtype>::sumSquareParamsData() {
	double result = 0.0;
	for(uint32_t i = 0; i < this->_params.size(); i++) {
		result += this->_params[i]->sumsq_device_data();
	}
	return result;
}

template <typename Dtype>
double LearnableLayer<Dtype>::sumSquareParamsGrad() {
	double result = 0.0;
	for(uint32_t i = 0; i < this->_params.size(); i++) {
		result += this->_params[i]->sumsq_device_grad();
	}
	return result;
}

template <typename Dtype>
void LearnableLayer<Dtype>::scaleParamsGrad(float scale) {
	for(uint32_t i = 0; i < this->_params.size(); i++) {
		this->_params[i]->scale_device_grad(scale);
	}
}

template <typename Dtype>
uint32_t LearnableLayer<Dtype>::boundParams() {
	uint32_t updateCount = 0;
	for (uint32_t i = 0; i < this->_params.size(); i++) {
		updateCount += this->_params[i]->bound_grad();
	}
	return updateCount;
}

template <typename Dtype>
uint32_t LearnableLayer<Dtype>::numParams() {
	return this->_params.size();
}

template <typename Dtype>
void LearnableLayer<Dtype>::saveParams(ofstream& ofs) {
	uint32_t numParams = _params.size();
	//ofs.write((char*)&numParams, sizeof(uint32_t));
	for(uint32_t i = 0; i < numParams; i++) {
		_params[i]->save(ofs);
	}
}

template <typename Dtype>
void LearnableLayer<Dtype>::loadParams(ifstream& ifs) {
	uint32_t numParams;
	ifs.read((char*)&numParams, sizeof(uint32_t));
	for(uint32_t i = 0; i < numParams; i++) {
		_params[i]->load(ifs);
	}
}

template <typename Dtype>
void LearnableLayer<Dtype>::loadParams(map<string, Data<Dtype>*>& dataMap) {
	typename map<string, Data<Dtype>*>::iterator it;

	//char tempName[80];
	for (uint32_t i = 0; i < this->_params.size(); i++) {

		// XXX: so temporal ~~~
		//Util::refineParamName(this->_params[i]->_name.c_str(), tempName);
		//string refinedName(tempName);
		//cout << "refineName: " << refinedName << ", ";

		cout << "looking for " << this->_params[i]->_name;
		it = dataMap.find(this->_params[i]->_name.c_str());
		if (it == dataMap.end()) {
			cout << " ... could not find ... " << endl;
			continue;
		}
		cout << " ... found ... " << endl;

		this->_params[i]->reshapeLike(it->second);
		this->_params[i]->set_device_with_host_data(it->second->host_data());
		this->_paramsInitialized[i] = true;
	}
}

template<typename Dtype>
void LearnableLayer<Dtype>::donateParam(LearnableLayer<Dtype>* receiver) {
    receiver->_params.clear();
    receiver->_paramsHistory.clear();
    receiver->_paramsHistory2.clear();
    receiver->_paramsInitialized.clear();
    receiver->updateParams.clear();

    for (int i = 0; i < this->_params.size(); i++) {
        receiver->_params.push_back(this->_params[i]);
    }

    SASSERT0(this->_paramsHistory.size() == this->_paramsHistory2.size());

    for (int i = 0; i < this->_paramsHistory.size(); i++) {
        receiver->_paramsHistory.push_back(this->_paramsHistory[i]);
        receiver->_paramsHistory2.push_back(this->_paramsHistory2[i]);
    }

    for (int i = 0; i < this->_paramsInitialized.size(); i++) {
        receiver->_paramsInitialized.push_back(this->_paramsInitialized[i]);
    }

    for (int i = 0; i < this->updateParams.size(); i++) {
        receiver->updateParams.push_back(this->updateParams[i]);
    }
}

template class LearnableLayer<float>;
