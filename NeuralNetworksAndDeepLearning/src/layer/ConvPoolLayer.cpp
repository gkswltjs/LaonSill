/*
 * ConvPoolLayer.cpp
 *
 *  Created on: 2016. 5. 12.
 *      Author: jhkim
 */

#include "ConvPoolLayer.h"
#include "FullyConnectedLayer.h"
#include "../Util.h"

ConvPoolLayer::ConvPoolLayer(string name, io_dim in_dim, filter_dim filter_d, pool_dim pool_d, Activation *activation_fn, Pooling *pooling_fn)
	: HiddenLayer(name, in_dim, in_dim) {
	//this->in_dim = in_dim;
	this->filter_d = filter_d;
	this->pool_d = pool_d;

	// determine output dimension by in_dim, filter_dim, pool_dim
	//this->out_dim.rows = (in_dim.rows-filter_d.rows+1)/pool_d.rows;
	//this->out_dim.cols = (in_dim.cols-filter_d.cols+1)/pool_d.cols;
	this->out_dim.rows = in_dim.rows/pool_d.rows;
	this->out_dim.cols = in_dim.cols/pool_d.cols;
	this->out_dim.channels = filter_d.filters;

	this->delta_input.set_size(in_dim.rows, in_dim.cols, in_dim.channels);



	filters = new rcube[filter_d.filters];
	nabla_w = new rcube[filter_d.filters];
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
	z.set_size(in_dim.rows, in_dim.cols, filter_d.filters);
	activated.set_size(size(z));
	pool_map.set_size(size(z));
	//poold.set_size(activated.n_rows/pool_d.rows, activated.n_cols/pool_d.cols, filter_d.filters);
	output.set_size(out_dim.rows, out_dim.cols, out_dim.channels);


	// TODO activation에 따라 weight 초기화 하도록 해야 함.
	this->activation_fn = activation_fn;
	//if(this->activation_fn) this->activation_fn->initialize_weight();
	int n_out = filter_d.filters*filter_d.rows*filter_d.cols/(pool_d.rows*pool_d.cols);
	for(int i = 0; i < filter_d.filters; i++) {
		//filters[i].randn();
		//Util::printCube(filters[i], "filter:");
		filters[i] *= 1 / sqrt(n_out);
		//Util::printCube(filters[i], "filter:");
	}

	this->pooling_fn = pooling_fn;
}

ConvPoolLayer::~ConvPoolLayer() {
	if(filters) delete filters;
	if(nabla_w) delete nabla_w;
}


void ConvPoolLayer::feedforward(int idx, const rcube &input) {
	//Util::printCube(input, "input:");
	Util::convertCube(input, this->input);

	z.zeros();
	rmat conv(size(z.slice(0)));

	// 1. CONVOLUTION
	// for i, features (about output)
	for(int i = 0; i < filter_d.filters; i++) {
		// for j, channels (about input)
		for(int j = 0; j < filter_d.channels; j++) {
			//Util::printMat(this->input.slice(j), "input:");
			//Util::printMat(filters[i].slice(j), "filter:");
			convolution(this->input.slice(j), filters[i].slice(j), conv);
			//Util::printMat(conv, "conv:");
			z.slice(i) += conv;
			//Util::printCube(z, "z:");
		}
	}
	//Util::printCube(z, "z:");

	// 2. ACTIVATION
	activation_fn->activate(z, activated);
	//Util::printCube(activated, "activated:");

	// 3. MAX-POOLING
	pooling_fn->pool(pool_d, activated, pool_map, output);
	//Util::printCube(output, "output:");
}





void ConvPoolLayer::backpropagation(int idx, HiddenLayer *next_layer) {
	rcube da;
	activation_fn->d_activate(activated, da);

	rcube dp;
	// 두 레이어를 연결하는 Weight가 있는 경우 (현재 다음 레이어가 FC인 케이스 only)
	// next_w->()*next_delta: 다음 FC의 delta를 현재 CONV max pool의 delta로 dimension 변환
	// max pool의 delta를 d_pool을 통해 upsample
	rcube w_next_delta(size(output));


	/*
	FullyConnectedLayer *fc_layer = dynamic_cast<FullyConnectedLayer *>(next_layer);
	if(fc_layer) {
		cube temp(output.size(), 1, 1);
		//temp.slice(0) = fc_layer->getWeight().t()*fc_layer->getDelta().slice(0);
		temp.slice(0) = fc_layer->getDelta().slice(0);

		// output dim 기준으로 w_next_delta를 변환
		Util::convertCube(temp, w_next_delta);
	}
	// 두 레이어를 연결하는 Weight가 없는 경우 (현재 다음 레이어가 CONV인 케이스)
	else {
		ConvPoolLayer *conv_layer = dynamic_cast<ConvPoolLayer *>(next_layer);

		w_next_delta = conv_layer->getDelta();

		//rmat dconv(size(output.slice(0)));
		//w_next_delta.zeros();
		//for(int i = 0; i < conv_layer->get_filter_dim().channels; i++) {
		//	for(int j = 0; j < conv_layer->get_filter_dim().filters; j++) {
		//		convolution(conv_layer->getDelta().slice(j), flipud(fliplr(conv_layer->getWeight()[j].slice(i))), dconv);
		//		//d_convolution(conv_layer->getDelta().slice(j), conv_layer->getWeight()[j].slice(i), dconv);
		//		w_next_delta.slice(i) += dconv;
		//	}
		//}
	}
	*/
	Util::convertCube(next_layer->getDeltaInput(), w_next_delta);

	pooling_fn->d_pool(pool_d, w_next_delta, pool_map, dp);
	delta =  dp % da;		//delta conv
	Util::printCube(delta, "delta:");

	// dC/dw
	rmat conv(filter_d.rows, filter_d.cols);
	for(int i = 0; i < filter_d.filters; i++) {
		for(int j = 0; j < filter_d.channels; j++) {
			d_convolution(input.slice(j), delta.slice(i), conv);
			Util::printMat(conv, "conv:");

			Util::printMat(nabla_w[i].slice(j), "nabla_w:");
			nabla_w[i].slice(j) += conv;
			Util::printMat(nabla_w[i].slice(j), "nabla_w after:");
		}
		nabla_b(i) += accu(delta.slice(i));
	}



	// dC/dx
	rmat dconv(size(input.slice(0)));
	delta_input.zeros();
	for(int i = 0; i < filter_d.channels; i++) {
		for(int j = 0; j < filter_d.filters; j++) {
			convolution(delta.slice(j), flipud(fliplr(filters[j].slice(i))), dconv);
			//d_convolution(conv_layer->getDelta().slice(j), conv_layer->getWeight()[j].slice(i), dconv);
			delta_input.slice(i) += dconv;
		}
	}

}





void ConvPoolLayer::convolution(const rmat &image, const rmat &filter, rmat &result) {
	int i, j, k, m;

	int top_pad = (filter.n_cols-1)/2;
	int left_pad = (filter.n_rows-1)/2;
	int in_image_row_idx;
	int in_image_col_idx;
	double conv;

	for(i = 0; i < image.n_rows; i++) {
		for(j = 0; j < image.n_cols; j++) {
			conv = 0;
			for(k = 0; k < filter.n_rows; k++) {
				for(m = 0; m < filter.n_cols; m++) {
					in_image_row_idx = i-left_pad+k;
					in_image_col_idx = j-top_pad+m;
					if((in_image_row_idx >= 0 && in_image_row_idx < image.n_rows)
							&& (in_image_col_idx >=0 && in_image_col_idx < image.n_cols)) {
						//conv += image.mem[in_image_row_idx+(in_image_col_idx)*image.n_cols]*filter.mem[k+m*filter.n_cols];
						conv += M_MEM(image, in_image_row_idx, in_image_col_idx)*M_MEM(filter, k, m);
					}
				}
			}
			result(i, j) = conv;
		}
	}
}



void ConvPoolLayer::d_convolution(const rmat &conv, const rmat &filter, rmat &result) {

	int i, j, k, m;
	double dconv;

	int top_pad = (result.n_rows-1)/2;
	int left_pad = (result.n_cols-1)/2;
	int x_row_start_idx, x_col_start_idx;
	int y_row_start_idx, y_col_start_idx;
	int row_overlapped;
	int col_overlapped;

	for(i = -left_pad; i < (int)result.n_rows-left_pad; i++) {
		for(j = -top_pad; j < (int)result.n_cols-top_pad; j++) {
			x_row_start_idx = max(i, 0);
			x_col_start_idx = max(j, 0);
			y_row_start_idx = max(-i, 0);
			y_col_start_idx = max(-j, 0);

			row_overlapped = min(conv.n_rows, conv.n_rows+i)-x_row_start_idx;
			col_overlapped = min(conv.n_cols, conv.n_cols+j)-x_col_start_idx;

			dconv = 0;

			//cout << "(" << i << ", " << j << "): " << endl;

			for(k = 0; k < row_overlapped; k++) {
				for(m = 0; m < col_overlapped; m++) {
					//cout << "(" << x_row_start_idx+k << ", " << x_col_start_idx+m << ") * (" << y_row_start_idx+k << ", " << y_col_start_idx+m << ")" << endl;
					//dconv += conv.mem[x_row_start_idx+k+(x_col_start_idx+m)*conv.n_cols]
					//				  *filter((y_row_start_idx+k)+(y_col_start_idx+m)*filter.n_cols);
					dconv += M_MEM(conv, x_row_start_idx+k, x_col_start_idx+m)*M_MEM(filter, y_row_start_idx+k, y_col_start_idx+m);
				}
			}
			result(left_pad+i, top_pad+j) = dconv;
		}
	}








	/*
	int i, j, k, m;
	rmat filter_flip = flipud(fliplr(filter));

	int filter_slide_min_row_index = -filter_flip.n_rows+1;		// inclusive
	int filter_slide_min_col_index = -filter_flip.n_cols+1;		// inclusive
	int filter_slide_max_row_index = conv.n_rows;				// exclusive
	int filter_slide_max_col_index = conv.n_cols;				// exclusive
	double dconv;

	// filter slide 범위
	for(i = filter_slide_min_row_index; i < filter_slide_max_row_index; i++) {
		for(j = filter_slide_min_col_index; j < filter_slide_max_col_index; j++) {
			dconv = 0;
			for(k = 0; k < filter_flip.n_rows; k++) {
				for(m = 0; m < filter_flip.n_cols; m++) {
					if((i+k >= 0 && i+k < conv.n_rows) && (j+m >=0 && j+m < conv.n_cols)) {
						//dconv += conv(i+k, j+m)*filter_flip(k, m);
						dconv += conv.mem[i+k+(j+m)*conv.n_cols]*filter_flip(k+m*filter.n_cols);
					}
				}
			}
			result(i-filter_slide_min_row_index, j-filter_slide_min_col_index) = dconv;
		}
	}
	*/
}




void ConvPoolLayer::reset_nabla(int idx) {
	for(int i = 0; i < filter_d.filters; i++) nabla_w[i].zeros();
	nabla_b.zeros();
}


void ConvPoolLayer::update(int idx, double eta, double lambda, int n, int miniBatchSize) {
	for(int i = 0; i < filter_d.filters; i++) {
		filters[i] = (1-eta*lambda/n)*filters[i] - (eta/miniBatchSize)*nabla_w[i];
	}
	biases -= eta/miniBatchSize*nabla_b;
}
































