/*
 * SoftmaxLayer.h
 *
 * C = -ln ay
 * dai/dzj =
 * 			ai*(1-ai) 	: when i = j
 * 			-ai*aj		: when i != j
 *
 * dC/dz = dC/da * da/dz
 * dC/da = -1/a
 *
 * dC/dz = -1/a *
 * 			ai*(1-ai) = ai-1	: when i = j
 * 			-ai*aj = aj-0		: when i != j
 *
 * dC/dz = a - y
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#ifndef LAYER_SOFTMAXLAYER_H_
#define LAYER_SOFTMAXLAYER_H_

#include "OutputLayer.h"
#include "../cost/LogLikelihoodCost.h"
#include "../activation/Softmax.h"
#include "../exception/Exception.h"
#include <armadillo>

using namespace arma;






class SoftmaxLayer : public OutputLayer {
public:
	SoftmaxLayer() { this->type = LayerType::Softmax; }
	SoftmaxLayer(const char *name, io_dim in_dim, io_dim out_dim, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler)
		: OutputLayer(name, in_dim, out_dim, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler,
				ActivationType::Softmax, CostType::LogLikelihood) {
		initialize();
	}
	virtual ~SoftmaxLayer() {}

	void save(UINT idx, ofstream &ofs) {
		if(!isLastPrevLayerRequest(idx)) throw Exception();
		OutputLayer::save(ofs);
		propSave(ofs);
	}

	void load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
		OutputLayer::load(ifs, layerMap);
		initialize();
	}

#if CPU_MODE
public:
	SoftmaxLayer(const char *name, int n_in, int n_out, double p_dropout, update_param weight_update_param, update_param bias_update_param,
			param_filler weight_filler, param_filler bias_filler)
		: OutputLayer(name, n_in, n_out, p_dropout, weight_update_param, bias_update_param, weight_filler, bias_filler,
				ActivationType::Softmax, CostType::LogLikelihood) {
		initialize();
	}

	void cost(const rvec &target) {
		// delta
		cost_fn->d_cost(z, output, target, delta);
		Util::printVec(nabla_b, "bias:");
		Util::printMat(nabla_w, "weight");
		Util::printCube(delta, "delta:");
		Util::printCube(input, "input:");
		nabla_b += delta.slice(0);
		// delta weight
		nabla_w += delta.slice(0)*input.slice(0).t();

		// delta input
		delta_input.slice(0) = weight.t()*delta.slice(0);

		propBackpropagation();
	}



private:
	void initialize() {
		this->type = LayerType::Softmax;
		this->id = Layer::generateLayerId();

		//this->cost_fn = CostFactory::create(CostType::LogLikelihood);
		//this->activation_fn = ActivationFactory::create(ActivationType::Softmax);
		//this->activation_fn->initialize_weight(in_dim.size(), weight);

		//weight.zeros();
		//bias.zeros();
	}

#else
public:
	void cost(const UINT *target) {
		Util::printMessage("SoftmaxLayer::cost()---"+string(name));
		Cuda::refresh();

		cost_fn->d_cost(d_z, d_output, target, d_delta, out_dim.rows, out_dim.batches);
		Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "d_delta:");
		// Accounting for batch size in SGD
		// checkCudaErrors(cublasSscal(cublasHandle, ref_fc2.outputs * m_batchSize, &scalVal, dloss_data, 1));

		float alpha = 1.0f, beta = 0.0f;
		Util::printDeviceData(d_input, in_dim.rows, in_dim.batches, 1, 1, "d_input:");
		checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, out_dim.rows, in_dim.rows, out_dim.batches,
				&alpha, d_delta, out_dim.rows, d_input, in_dim.rows, &beta, d_delta_weight, out_dim.rows));
		Util::printDeviceData(d_delta_weight, out_dim.rows, in_dim.rows, 1, 1, "d_delta_weight:");

		checkCudaErrors(cublasSgemv(Cuda::cublasHandle, CUBLAS_OP_N, out_dim.rows, out_dim.batches,
				&alpha, d_delta, out_dim.rows, d_onevec, 1, &beta, d_delta_bias, 1));
		Util::printDeviceData(d_delta_bias, out_dim.rows, 1, 1, 1, "d_delta_bias:");

		Util::printDeviceData(d_weight, out_dim.rows, in_dim.rows, 1, 1, "d_weight:");
		Util::printDeviceData(d_delta, out_dim.rows, out_dim.batches, 1, 1, "d_delta:");
		checkCudaErrors(cublasSgemm(Cuda::cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, in_dim.rows, out_dim.batches, out_dim.rows,
				&alpha, d_weight, out_dim.rows, d_delta, out_dim.rows, &beta, d_delta_input, in_dim.rows));
		Util::printDeviceData(d_delta_input, in_dim.rows, in_dim.batches, 1, 1, "d_delta_input:");

		propBackpropagation();
	}



private:
	void initialize() {
		this->type = LayerType::Softmax;
		this->id = Layer::generateLayerId();

		//this->cost_fn = CostFactory::create(CostType::LogLikelihood);
		//this->activation_fn = ActivationFactory::create(ActivationType::Softmax);
		//this->activation_fn->initialize_weight(in_dim.size(), weight);

		//weight.zeros();
		//bias.zeros();
	}
#endif

};







#endif /* LAYER_SOFTMAXLAYER_H_ */
















