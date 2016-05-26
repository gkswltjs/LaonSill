/*
 * ConvLayer.cpp
 *
 *  Created on: 2016. 5. 23.
 *      Author: jhkim
 */


#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "../Util.h"

ConvLayer::ConvLayer(io_dim in_dim, filter_dim filter_d, Activation *activation_fn)
	: HiddenLayer(in_dim, in_dim) {
	//this->in_dim = in_dim;
	this->filter_d = filter_d;

	// determine output dimension by in_dim, filter_dim, pool_dim
	//this->out_dim.rows = (in_dim.rows-filter_d.rows+1)/pool_d.rows;
	//this->out_dim.cols = (in_dim.cols-filter_d.cols+1)/pool_d.cols;
	this->out_dim.rows = in_dim.rows/filter_d.stride;
	this->out_dim.cols = in_dim.cols/filter_d.stride;
	this->out_dim.channels = filter_d.filters;

	this->delta_input.set_size(in_dim.rows, in_dim.cols, in_dim.channels);


	filters = new cube[filter_d.filters];
	nabla_w = new cube[filter_d.filters];
	for(int i = 0; i < filter_d.filters; i++) {
		filters[i].set_size(filter_d.rows, filter_d.cols, filter_d.channels);
		filters[i].randn();
		nabla_w[i].set_size(filter_d.rows, filter_d.cols, filter_d.channels);
		nabla_w[i].zeros();
	}

	biases.set_size(filter_d.filters);
	biases.randn();
	nabla_b.set_size(filter_d.filters);
	nabla_b.randn();


	//z.set_size(in_dim.rows-filter_d.rows+1, in_dim.cols-filter_d.cols+1, filter_d.filters);
	z.set_size(out_dim.rows, out_dim.cols, out_dim.channels);
	output.set_size(size(z));

	// TODO activation에 따라 weight 초기화 하도록 해야 함.
	this->activation_fn = activation_fn;
	//if(this->activation_fn) this->activation_fn->initialize_weight();
	int n_out = filter_d.filters*filter_d.rows*filter_d.cols/9;
	for(int i = 0; i < filter_d.filters; i++) {
		//filters[i].randn();
		//Util::printCube(filters[i], "filter:");
		filters[i] *= 1 / sqrt(n_out);
		//Util::printCube(filters[i], "filter:");
	}
}

ConvLayer::~ConvLayer() {
	if(filters) delete filters;
	if(nabla_w) delete nabla_w;
}


void ConvLayer::feedforward(const cube &input) {
	//Util::printCube(input, "input:");
	Util::convertCube(input, this->input);

	z.zeros();
	mat conv(size(z.slice(0)));

	// 1. CONVOLUTION
	// for i, features (about output)
	for(int i = 0; i < filter_d.filters; i++) {
		// for j, channels (about input)
		for(int j = 0; j < filter_d.channels; j++) {
			Util::printMat(this->input.slice(j), "input:");
			Util::printMat(filters[i].slice(j), "filter:");
			convolution(this->input.slice(j), filters[i].slice(j), conv, filter_d.stride);
			Util::printMat(conv, "conv:");
			z.slice(i) += conv;
			//Util::printCube(z, "z:");
		}
	}
	Util::printCube(z, "z:");

	// 2. ACTIVATION
	activation_fn->activate(z, output);
	Util::printCube(output, "output:");

	Layer::feedforward(this->output);

}





void ConvLayer::backpropagation(HiddenLayer *next_layer) {
	cube da;
	activation_fn->d_activate(output, da);

	cube dp;
	// 두 레이어를 연결하는 Weight가 있는 경우 (현재 다음 레이어가 FC인 케이스 only)
	// next_w->()*next_delta: 다음 FC의 delta를 현재 CONV max pool의 delta로 dimension 변환
	// max pool의 delta를 d_pool을 통해 upsample
	cube w_next_delta(size(output));
	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);

	delta = w_next_delta % da;		//delta conv

	Util::printCube(da, "da:");
	Util::printCube(w_next_delta, "w_next_delta:");
	Util::printCube(delta, "delta:");

	// dw
	mat conv(filter_d.rows, filter_d.cols);
	for(int i = 0; i < filter_d.filters; i++) {
		for(int j = 0; j < filter_d.channels; j++) {
			dw_convolution(delta.slice(i), input.slice(j), conv);
			Util::printMat(conv, "conv:");

			Util::printMat(nabla_w[i].slice(j), "nabla_w:");
			nabla_w[i].slice(j) += conv;
			Util::printMat(nabla_w[i].slice(j), "nabla_w after:");
		}
		nabla_b(i) += accu(delta.slice(i));
	}


	// dx
	mat dconv(size(input.slice(0)));
	delta_input.fill(0.0);
	for(int i = 0; i < filter_d.channels; i++) {
		for(int j = 0; j < filter_d.filters; j++) {
			Util::printMat(filters[j].slice(i), "filter:");
			Util::printMat(flipud(fliplr(filters[j].slice(i))), "filp:");
			dx_convolution(delta.slice(j), flipud(fliplr(filters[j].slice(i))), dconv);
			//d_convolution(conv_layer->getDelta().slice(j), conv_layer->getWeight()[j].slice(i), dconv);
			delta_input.slice(i) += dconv;
		}
	}


	HiddenLayer::backpropagation(this);

}


void ConvLayer::convolution(const mat &x, const mat &w, mat &result, int stride) {
	int i, j, k, m;

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

					if((in_image_row_idx >= 0 && in_image_row_idx < x.n_rows)
							&& (in_image_col_idx >=0 && in_image_col_idx < x.n_cols)) {
						conv += x.mem[in_image_row_idx+(in_image_col_idx)*x.n_cols]*w.mem[k+m*w.n_cols];
					}
				}
			}
			result(i, j) = conv;
		}
	}
}




// Yn,m = Sigma(i for 0~filter_size-1)Sigma(j for 0~filter_size-1) Wi,j * Xstride*n-filter_size/2+i, stride*m-filter_size/2+j
// dC/dWi,j	= dC/dY * dY/dWi,j
// 			= Sigma(n)Sigma(m) delta n,m * Xstride*n-filter_size/2+i, stride*m-filter_size/2+j)

void ConvLayer::dw_convolution(const mat &d, const mat &x, mat &result) {

	int i, j, k, l;

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

					if((in_image_row_idx >= 0 && in_image_row_idx < x.n_rows)
							&& (in_image_col_idx >= 0 && in_image_col_idx < x.n_cols)) {
						dconv += d(k, l)*x(in_image_row_idx, in_image_col_idx);
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


void ConvLayer::dx_convolution(const mat &d, const mat &w, mat &result) {
	int i, j;

	mat d_ex(filter_d.stride*d.n_rows, filter_d.stride*d.n_cols);
	d_ex.zeros();

	for(i = 0; i < d.n_rows; i++) {
		for(j = 0; j < d.n_cols; j++) {
			//d_ex.mem[filter_d.stride*i+filter_d.stride*j*d_ex.n_cols] = d.mem[i+j*d.n_cols];
			d_ex(filter_d.stride*i, filter_d.stride*j) = d.mem[i+j*d.n_cols];
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






void ConvLayer::reset_nabla() {
	for(int i = 0; i < filter_d.filters; i++) nabla_w[i].fill(0.0);
	nabla_b.fill(0.0);

	Layer::reset_nabla();
}


void ConvLayer::update(double eta, double lambda, int n, int miniBatchSize) {
	for(int i = 0; i < filter_d.filters; i++) {
		filters[i] = (1-eta*lambda/n)*filters[i] - (eta/miniBatchSize)*nabla_w[i];
	}
	biases -= eta/miniBatchSize*nabla_b;

	Layer::update(eta, lambda, n, miniBatchSize);
}











