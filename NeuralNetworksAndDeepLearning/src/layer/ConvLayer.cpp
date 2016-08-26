/*
 * ConvLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */


#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "../Util.h"
#include "../exception/Exception.h"
#include "../network/NetworkConfig.h"


ConvLayer::ConvLayer() {
	this->type = Layer::Conv;
}

ConvLayer::ConvLayer(Builder* builder)
	: HiddenLayer(builder) {
	initialize(builder->_filterDim, builder->_weightUpdateParam, builder->_biasUpdateParam,
			builder->_weightFiller, builder->_biasFiller, builder->_activationType);
}

ConvLayer::ConvLayer(const string name, filter_dim filter_d, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, Activation::Type activationType)
	: HiddenLayer(name) {
	initialize(filter_d, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);
}



double ConvLayer::sumSquareParamsData() {
	double result = 0.0;
	for(uint32_t i = 0; i < _params.size(); i++) {
		result += _params[i]->sumsq_device_data();
	}
	return result;
}

double ConvLayer::sumSquareParamsGrad() {
	double result = 0.0;
	for(uint32_t i = 0; i < _params.size(); i++) {
		result += _params[i]->sumsq_device_grad();
	}
	return result;
}

void ConvLayer::scaleParamsGrad(DATATYPE scale) {
	for(uint32_t i = 0; i < _params.size(); i++) {
		_params[i]->scale_device_grad(scale);
	}
}








#ifndef GPU_MODE
void convolution(const rmat &x, const rmat &w, rmat &result, int stride);
void dw_convolution(const rmat &d, const rmat &x, rmat &result);
void dx_convolution(const rmat &d, const rmat &w, rmat &result);


ConvLayer::~ConvLayer() {
	ActivationFactory::destory(activation_fn);
	if(filters) {
		delete [] filters;
		filters = NULL;
	}
	if(nabla_w) {
		delete [] nabla_w;
		nabla_w = NULL;
	}
}

void ConvLayer::_load(ifstream &ifs, map<Layer *, Layer *> &layerMap) {
	HiddenLayer::_load(ifs, layerMap);

	filter_dim filter_d;
	ifs.read((char *)&filter_d, sizeof(filter_dim));

	Activation::Type activationType;
	ifs.read((char *)&activationType, sizeof(int));

	update_param weight_update_param;
	ifs.read((char *)&weight_update_param, sizeof(update_param));

	update_param bias_update_param;
	ifs.read((char *)&bias_update_param, sizeof(update_param));

	param_filler weight_filler;
	ifs.read((char *)&weight_filler, sizeof(param_filler));

	param_filler bias_filler;
	ifs.read((char *)&bias_filler, sizeof(param_filler));

	initialize(filter_d, weight_update_param, bias_update_param, weight_filler, bias_filler, activationType);

	// initialize() 내부에서 weight, bias를 초기화하므로 initialize() 후에 weight, bias load를 수행해야 함
	for(UINT i = 0; i < filter_d.filters; i++) {
		filters[i].load(ifs, file_type::arma_binary);
	}
	biases.load(ifs, file_type::arma_binary);
}

void ConvLayer::initialize(filter_dim filter_d, update_param weight_update_param, update_param bias_update_param,
		param_filler weight_filler, param_filler bias_filler, Activation::Type activationType) {

	this->type = Layer::Conv;

	//this->in_dim = in_dim;
	this->filter_d = filter_d;

	this->weight_update_param = weight_update_param;
	this->bias_update_param = bias_update_param;
	this->weight_filler = weight_filler;
	this->bias_filler = bias_filler;

	// determine output dimension by in_dim, filter_dim, pool_dim
	//this->out_dim.rows = (in_dim.rows-filter_d.rows+1)/pool_d.rows;
	//this->out_dim.cols = (in_dim.cols-filter_d.cols+1)/pool_d.cols;
	this->out_dim.rows = in_dim.rows/filter_d.stride;
	this->out_dim.cols = in_dim.cols/filter_d.stride;
	this->out_dim.channels = filter_d.filters;

	this->delta_input.set_size(in_dim.rows, in_dim.cols, in_dim.channels);


	filters = new rcube[filter_d.filters];
	nabla_w = new rcube[filter_d.filters];

	for(UINT i = 0; i < filter_d.filters; i++) {
		filters[i].set_size(filter_d.rows, filter_d.cols, filter_d.channels);
		//filters[i].randn();
		this->weight_filler.fill(filters[i], in_dim.size());

		nabla_w[i].set_size(filter_d.rows, filter_d.cols, filter_d.channels);
		nabla_w[i].zeros();
	}

	biases.set_size(filter_d.filters);
	this->bias_filler.fill(biases, in_dim.size());

	nabla_b.set_size(filter_d.filters);
	nabla_b.zeros();


	//z.set_size(in_dim.rows-filter_d.rows+1, in_dim.cols-filter_d.cols+1, filter_d.filters);
	z.set_size(out_dim.rows, out_dim.cols, out_dim.channels);
	output.set_size(size(z));

	this->activation_fn = ActivationFactory::create(activationType);
	//if(this->activation_fn) this->activation_fn->initialize_weight();
	//int n_out = filter_d.filters*filter_d.rows*filter_d.cols/9;
	//if(this->activation_fn) {
	//	for(UINT i = 0; i < filter_d.filters; i++) {
	//		 this->activation_fn->initialize_weight(in_dim.size(), filters[i]);
	//	}
	//}

	delta.set_size(size(z));
}

void ConvLayer::_feedforward() {

	// 현재 CONV 레이어의 경우 여러 레이어로 값이 전달되지 않기 때문에 무의미하다.
	// 다만 backpropagation에서 delta값을 합으로 할당하기 때문에 어쨌든 0로 init은 반드시 해야 함.
	// delta.zeros();

	Util::printCube(input, "input:");
	Util::convertCube(input, this->input);

	z.zeros();
	rmat conv(size(z.slice(0)));

	// 1. CONVOLUTION
	// for i, features (about output)
	for(UINT i = 0; i < filter_d.filters; i++) {
		// for j, channels (about input)
		for(UINT j = 0; j < filter_d.channels; j++) {
			Util::printMat(this->input.slice(j), "input:");
			Util::printMat(filters[i].slice(j), "filter:");
			convolution(this->input.slice(j), filters[i].slice(j), conv, filter_d.stride);
			Util::printMat(conv, "conv:");
			Util::printCube(z, "z:");
			z.slice(i) += conv;
			Util::printCube(z, "z:");
		}
		//Util::printCube(z, "z:");
		//Util::printVec(biases, "biases:");
		z.slice(i) += biases(i, 0);
		//Util::printCube(z, "z:");
	}

	Util::printCube(z, "z:");


	// 2. ACTIVATION
	activation_fn->forward(z, output);
	Util::printCube(output, "output:");

	propFeedforward(this->output, end);
}

void ConvLayer::backpropagation(UINT idx, HiddenLayer *next_layer) {
	// 여러 source로부터 delta값이 모두 모이면 dw, dx 계산
	if(!isLastNextLayerRequest(idx)) throw Exception();

	rcube da;
	activation_fn->backward(output, da);

	rcube dp;
	// 두 레이어를 연결하는 Weight가 있는 경우 (현재 다음 레이어가 FC인 케이스 only)
	// next_w->()*next_delta: 다음 FC의 delta를 현재 CONV max pool의 delta로 dimension 변환
	// max pool의 delta를 d_pool을 통해 upsample
	rcube w_next_delta(size(output));
	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);


	delta = w_next_delta % da;		//delta conv

	Util::printCube(da, "da:");
	Util::printCube(w_next_delta, "w_next_delta:");
	Util::printCube(delta, "delta:");

	// dw
	rmat conv(filter_d.rows, filter_d.cols);
	for(UINT i = 0; i < filter_d.filters; i++) {
		for(UINT j = 0; j < filter_d.channels; j++) {
			dw_convolution(delta.slice(i), input.slice(j), conv);
			Util::printMat(conv, "conv:");

			Util::printMat(nabla_w[i].slice(j), "nabla_w:");
			nabla_w[i].slice(j) += conv;
			Util::printMat(nabla_w[i].slice(j), "nabla_w after:");
		}
		nabla_b(i) += accu(delta.slice(i));
	}

	// dx
	rmat dconv(size(input.slice(0)));
	delta_input.zeros();
	for(UINT i = 0; i < filter_d.channels; i++) {
		for(UINT j = 0; j < filter_d.filters; j++) {
			Util::printMat(filters[j].slice(i), "filter:");
			Util::printMat(flipud(fliplr(filters[j].slice(i))), "filp:");
			dx_convolution(delta.slice(j), flipud(fliplr(filters[j].slice(i))), dconv);
			//d_convolution(conv_layer->getDelta().slice(j), conv_layer->getWeight()[j].slice(i), dconv);
			delta_input.slice(i) += dconv;
		}
	}

	propBackpropagation();
}

void ConvLayer::convolution(const rmat &x, const rmat &w, rmat &result, int stride) {
	UINT i, j, k, m;

	int top_pad = (w.n_cols-1)/2;
	int left_pad = (w.n_rows-1)/2;
	int in_image_row_idx;
	int in_image_col_idx;
	double conv;

	for(i = 0; i < x.n_rows; i+=stride) {
		for(j = 0; j < x.n_cols; j+=stride) {
			conv = 0;
			for(k = 0; k < w.n_rows; k++) {
				for(m = 0; m < w.n_cols; m++) {
					in_image_row_idx = i-left_pad+k;
					in_image_col_idx = j-top_pad+m;

					if((in_image_row_idx >= 0 && (UINT)in_image_row_idx < x.n_rows)
							&& (in_image_col_idx >=0 && (UINT)in_image_col_idx < x.n_cols)) {
						//conv += x.mem[in_image_row_idx+(in_image_col_idx)*x.n_cols]*w.mem[k+m*w.n_cols];
						//try {
						conv += M_MEM(x, in_image_row_idx, in_image_col_idx)*M_MEM(w, k, m);
						//} catch (...) {
						//	cout << "i:" << i << ", j:" << j << ", k:" << k << ", m:" << m << endl;
						//}
					}
				}
			}
			//try {
			result(i/stride, j/stride) = conv;
			//} catch (...) {
			//	cout << "i:" << i << ", j:" << j << endl;
			//}
		}
	}
}

// Yn,m = Sigma(i for 0~filter_size-1)Sigma(j for 0~filter_size-1) Wi,j * Xstride*n-filter_size/2+i, stride*m-filter_size/2+j
// dC/dWi,j	= dC/dY * dY/dWi,j
// 			= Sigma(n)Sigma(m) delta n,m * Xstride*n-filter_size/2+i, stride*m-filter_size/2+j)
void ConvLayer::dw_convolution(const rmat &d, const rmat &x, rmat &result) {

	UINT i, j, k, l;

	int top_pad = (filter_d.cols-1)/2;
	int left_pad = (filter_d.rows-1)/2;
	int in_image_row_idx;
	int in_image_col_idx;
	double dconv = 0.0;

	result.zeros();

	for(i = 0; i < filter_d.rows; i++) {
		for(j = 0; j < filter_d.cols; j++) {

			dconv = 0.0;
			for(k = 0; k < d.n_rows; k++) {
				for(l = 0; l < d.n_cols; l++) {
					in_image_row_idx = filter_d.stride*k-left_pad+i;
					in_image_col_idx = filter_d.stride*l-top_pad+j;

					if((in_image_row_idx >= 0 && (UINT)in_image_row_idx < x.n_rows)
							&& (in_image_col_idx >= 0 && (UINT)in_image_col_idx < x.n_cols)) {
						//dconv += d(k, l)*x(in_image_row_idx, in_image_col_idx);
						dconv += M_MEM(d, k, l)*M_MEM(x, in_image_row_idx, in_image_col_idx);
						//dconv += d.mem[in_image_row_idx+in_image_col_idx*d.n_cols]*x.mem[k+l*x.n_cols];
					}
				}
			}
			result(i, j) = dconv;
		}
	}

	Util::printMat(d, "d:");
	Util::printMat(x, "x:");
	Util::printMat(result, "result:");
}

void ConvLayer::dx_convolution(const rmat &d, const rmat &w, rmat &result) {
	UINT i, j;

	rmat d_ex(filter_d.stride*d.n_rows, filter_d.stride*d.n_cols);
	d_ex.zeros();

	for(i = 0; i < d.n_rows; i++) {
		for(j = 0; j < d.n_cols; j++) {
			//d_ex.mem[filter_d.stride*i+filter_d.stride*j*d_ex.n_cols] = d.mem[i+j*d.n_cols];
			//d_ex(filter_d.stride*i, filter_d.stride*j) = d.mem[i+j*d.n_cols];
			M_MEMPTR(d_ex, filter_d.stride*i, filter_d.stride*j) = M_MEM(d, i, j);
		}
	}

	Util::printMat(d, "d:");
	Util::printMat(d_ex, "d_ex:");


	convolution(d_ex, w, result, 1);

	Util::printMat(d, "d:");
	Util::printMat(d_ex, "d_ex:");
	Util::printMat(w, "w:");
	Util::printMat(result, "result:");
}

void ConvLayer::reset_nabla(UINT idx) {
	// 한번만 초기화하기 위해 마지막 prev layer의 초기화 요청에 대해서만 처리하고
	// next layer들에 대해서도 초기화 요청한다.
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	for(UINT i = 0; i < filter_d.filters; i++) nabla_w[i].zeros();
	nabla_b.zeros();

	propResetNParam();
}

void ConvLayer::update(UINT idx, UINT n, UINT miniBatchSize) {
	if(!isLastPrevLayerRequest(idx)) throw Exception();

	//for(UINT i = 0; i < filter_d.filters; i++) {
	//	filters[i] = (1-eta*lambda/n)*filters[i] - (eta/miniBatchSize)*nabla_w[i];
	//}
	//biases -= eta/miniBatchSize*nabla_b;

	for(UINT i = 0; i < filter_d.filters; i++) {
		filters[i] = (1-weight_update_param.lr_mult*weight_update_param.decay_mult/n)*filters[i] - (weight_update_param.lr_mult/miniBatchSize)*nabla_w[i];
	}
	biases -= bias_update_param.lr_mult/miniBatchSize*nabla_b;

	propUpdate(n, miniBatchSize);
}

void ConvLayer::_save(ofstream &ofs) {
	HiddenLayer::_save(ofs);

	int activationType = (int)activation_fn->getType();

	ofs.write((char *)&filter_d, sizeof(filter_dim));
	ofs.write((char *)&activationType, sizeof(int));
	ofs.write((char *)&weight_update_param, sizeof(update_param));
	ofs.write((char *)&bias_update_param, sizeof(update_param));
	ofs.write((char *)&weight_filler, sizeof(param_filler));
	ofs.write((char *)&bias_filler, sizeof(param_filler));
	//ofs.write((char *)&weight, sizeof(rmat));
	//ofs.write((char *)&bias, sizeof(rvec));

	for(UINT i = 0; i < filter_d.filters; i++) {
		filters[i].save(ofs, file_type::arma_binary);
	}
	biases.save(ofs, file_type::arma_binary);
}
#endif
