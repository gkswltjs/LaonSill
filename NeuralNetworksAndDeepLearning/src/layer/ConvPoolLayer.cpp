/*
 * ConvPoolLayer.cpp
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#include "ConvPoolLayer.h"
#include "FullyConnectedLayer.h"
#include "../Util.h"

ConvPoolLayer::ConvPoolLayer(io_dim in_dim, filter_dim filter_d, pool_dim pool_d, Activation *activation_fn, Pooling *pooling_fn)
	: HiddenLayer(in_dim, in_dim) {
	this->in_dim = in_dim;
	this->filter_d = filter_d;
	this->pool_d = pool_d;

	// determine output dimension by in_dim, filter_dim, pool_dim
	this->out_dim.rows = (in_dim.rows-filter_d.rows+1)/pool_d.rows;
	this->out_dim.cols = (in_dim.cols-filter_d.cols+1)/pool_d.cols;
	this->out_dim.channels = filter_d.filters;

	filters = new cube[filter_d.filters];
	nabla_w = new cube[filter_d.filters];
	for(int i = 0; i < filter_d.filters; i++) {
		filters[i].set_size(filter_d.rows, filter_d.cols, filter_d.channels);
		filters[i].randn();
		nabla_w[i].set_size(filter_d.rows, filter_d.cols, filter_d.channels);
		nabla_w[i].fill(0.0);
	}

	biases.set_size(filter_d.filters);
	biases.randn();
	nabla_b.set_size(filter_d.filters);
	nabla_b.randn();

	// TODO z 크기 conv2 결과에 맞춰 조정해야 함
	z.set_size(in_dim.rows-filter_d.rows+1, in_dim.cols-filter_d.cols+1, filter_d.filters);
	activated.set_size(in_dim.rows-filter_d.rows+1, in_dim.cols-filter_d.cols+1, filter_d.filters);
	pool_map.set_size(in_dim.rows-filter_d.rows+1, in_dim.cols-filter_d.cols+1, filter_d.filters);
	//poold.set_size(activated.n_rows/pool_d.rows, activated.n_cols/pool_d.cols, filter_d.filters);
	output.set_size(out_dim.rows, out_dim.cols, out_dim.channels);




	// TODO activation에 따라 weight 초기화 하도록 해야 함.
	this->activation_fn = activation_fn;
	//if(this->activation_fn) this->activation_fn->initialize_weight();
	int n_out = filter_d.filters*filter_d.rows*filter_d.cols/(pool_d.rows*pool_d.cols);
	for(int i = 0; i < filter_d.filters; i++) {
		//filters[i].randn();
		filters[i] *= 1 / sqrt(n_out);
	}

	this->pooling_fn = pooling_fn;
}

ConvPoolLayer::~ConvPoolLayer() {
	if(filters) delete filters;
	if(nabla_w) delete nabla_w;
}


void ConvPoolLayer::feedforward(const cube &input) {
	Util::printCube(input, "input:");
	Util::convertCube(input, this->input);

	z.fill(0.0);


	mat conv(this->input.n_rows-filter_d.rows+1, this->input.n_cols-filter_d.cols+1);



	// 1. CONVOLUTION
	// for i, features (about output)
	for(int i = 0; i < filter_d.filters; i++) {
		// for j, channels (about input)
		for(int j = 0; j < filter_d.channels; j++) {
			//z.slice(i) += conv2(input.slice(j), filters[i].slice(j), "same");
			convolution(this->input.slice(j), filters[i].slice(j), conv);
			z.slice(i) += conv;
		}
		//z.slice(i) += biases(i, 0);
		//z(i) += biases.row(i);
		//z.slice(i) = z.slice(i).submat(0, 0, z.slice(i).n_rows-2, z.slice(i).n_cols-2);
		//z(i) = z(i).submat(0, 0, z(i).n_rows-2, z(i).n_cols-2);
	}

	Util::printCube(z, "z:");


	// 2. ACTIVATION
	activation_fn->activate(z, activated);
	Util::printCube(activated, "activated:");

	// 3. MAX-POOLING
	pooling_fn->pool(pool_d, activated, pool_map, output);
	Util::printCube(output, "output:");
	//pool_map.print("pool_map:");

}





void ConvPoolLayer::backpropagation(HiddenLayer *next_layer) {
	cube da;
	activation_fn->d_activate(activated, da);

	cube dp;
	// 두 레이어를 연결하는 Weight가 있는 경우 (현재 다음 레이어가 FC인 케이스 only)
	// next_w->()*next_delta: 다음 FC의 delta를 현재 CONV max pool의 delta로 dimension 변환
	// max pool의 delta를 d_pool을 통해 upsample
	cube w_next_delta(size(output));
	FullyConnectedLayer *fc_layer = dynamic_cast<FullyConnectedLayer *>(next_layer);
	if(fc_layer) {
		cube temp(output.size(), 1, 1);

		Util::printMat(fc_layer->getWeight(), "weight:");
		Util::printMat(fc_layer->getDelta().slice(0), "delta:");

		temp.slice(0) = fc_layer->getWeight().t()*fc_layer->getDelta().slice(0);
		// output dim 기준으로 w_next_delta를 변환
		Util::convertCube(temp, w_next_delta);

		Util::printCube(temp, "temp:");
		Util::printCube(w_next_delta, "w_next_delta:");
	}
	// 두 레이어를 연결하는 Weight가 없는 경우 (현재 다음 레이어가 CONV인 케이스)
	else {
		ConvPoolLayer *conv_layer = dynamic_cast<ConvPoolLayer *>(next_layer);

		mat dconv(size(output.slice(0)));
		w_next_delta.fill(0.0);
		for(int i = 0; i < conv_layer->get_filter_dim().channels; i++) {
			for(int j = 0; j < conv_layer->get_filter_dim().filters; j++) {
				d_convolution(conv_layer->getDelta().slice(j), conv_layer->getWeight()[j].slice(i), dconv);
				w_next_delta.slice(i) += dconv;
			}
		}
	}
	pooling_fn->d_pool(pool_d, w_next_delta, pool_map, dp);
	delta =  dp % da;		//delta conv
	Util::printCube(delta, "delta:");

	mat conv(filter_d.rows, filter_d.cols);
	for(int i = 0; i < filter_d.filters; i++) {
		for(int j = 0; j < filter_d.channels; j++) {
			convolution(input.slice(j), delta.slice(i), conv);
			Util::printMat(conv, "conv:");

			Util::printMat(nabla_w[i].slice(j), "nabla_w:");
			nabla_w[i].slice(j) += conv;
			Util::printMat(nabla_w[i].slice(j), "nabla_w after:");
		}
		nabla_b(i) += accu(delta.slice(i));
	}



}





void ConvPoolLayer::convolution(const mat &image, const mat &filter, mat &result) {
	unsigned int i, j, k, m;
	unsigned int image_max_row_index = image.n_rows-filter.n_rows+1;
	unsigned int image_max_col_index = image.n_cols-filter.n_cols+1;
	double conv;
	for(i = 0; i < image_max_row_index; i++) {
		for(j = 0; j < image_max_col_index; j++) {
			conv = 0;
			for(k = 0; k < filter.n_rows; k++) {
				for(m = 0; m < filter.n_cols; m++) {
					conv += image(i+k, j+m)*filter(k, m);
				}
			}
			result(i, j) = conv;
		}
	}
}

void ConvPoolLayer::d_convolution(const mat &conv, const mat &filter, mat &result) {
	int i, j, k, m;

	mat filter_filp = flipud(fliplr(filter));

	int filter_slide_min_row_index = -filter_filp.n_rows+1;		// inclusive
	int filter_slide_min_col_index = -filter_filp.n_cols+1;		// inclusive
	int filter_slide_max_row_index = conv.n_rows;				// exclusive
	int filter_slide_max_col_index = conv.n_cols;				// exclusive
	double dconv;

	// filter slide 범위
	for(i = filter_slide_min_row_index; i < filter_slide_max_row_index; i++) {
		for(j = filter_slide_min_col_index; j < filter_slide_max_col_index; j++) {
			dconv = 0;
			for(k = 0; k < filter_filp.n_rows; k++) {
				for(m = 0; m < filter_filp.n_cols; m++) {
					if((i+k >= 0 && i+k < conv.n_rows) && (j+m >=0 && j+m < conv.n_cols)) {
						 dconv += conv(i+k, j+m)*filter_filp(k, m);
					}
				}
			}
			result(i-filter_slide_min_row_index, j-filter_slide_min_col_index) = dconv;
		}
	}
}



void ConvPoolLayer::reset_nabla() {
	for(int i = 0; i < filter_d.filters; i++) nabla_w[i].fill(0.0);
	nabla_b.fill(0.0);
}


void ConvPoolLayer::update(double eta, double lambda, int n, int miniBatchSize) {
	for(int i = 0; i < filter_d.filters; i++) {
		filters[i] = (1-eta*lambda/n)*filters[i] - (eta/miniBatchSize)*nabla_w[i];
	}
	biases -= eta/miniBatchSize*nabla_b;
}
































