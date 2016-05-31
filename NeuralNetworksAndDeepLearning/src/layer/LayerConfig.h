/*
 * LayerConfig.h
 *
 *  Created on: 2016. 5. 14.
 *      Author: jhkim
 */

#ifndef LAYERCONFIG_H_
#define LAYERCONFIG_H_

class Layer;
class HiddenLayer;


typedef struct io_dim {
    int rows;
    int cols;
    int channels;

    io_dim() {
    	this->rows = 1;
    	this->cols = 1;
    	this->channels = 1;
    }
    io_dim(int rows, int cols, int channels) {
    	this->rows = rows;
    	this->cols = cols;
    	this->channels = channels;
    }
    int size() const { return rows*cols*channels; }
} io_dim;

typedef struct filter_dim : public io_dim {
	int filters;
	int stride;

	filter_dim() {
		this->filters = 1;
		this->stride = 1;
	}
	filter_dim(int rows, int cols, int channels, int filters, int stride) : io_dim(rows, cols, channels) {
		this->filters = filters;
		this->stride = stride;
	}
} filter_dim;

typedef struct pool_dim {
	int rows;
	int cols;
	int stride;

	pool_dim() {
		this->rows = 1;
		this->cols = 1;
		this->stride = 1;
	}
	pool_dim(int rows, int cols, int stride) {
		this->rows = rows;
		this->cols = cols;
		this->stride = stride;
	}
} pool_dim;


typedef struct lrn_dim {
	int local_size;
	double alpha;
	double beta;

	lrn_dim() {
		this->local_size = 5;
		this->alpha = 1;
		this->beta = 5;
	}
	lrn_dim(int local_size, double alpha, double beta) {
		this->local_size = local_size;
		this->alpha = alpha;
		this->beta = beta;
	}
} lrn_dim;




typedef struct next_layer_relation {
	Layer *next_layer;
	int idx;

	next_layer_relation() {}
	next_layer_relation(Layer *next_layer, int idx) {
		this->next_layer = next_layer;
		this->idx = idx;
	}
} next_layer_relation;

typedef struct prev_layer_relation {
	HiddenLayer *prev_layer;
	int idx;

	prev_layer_relation() {}
	prev_layer_relation(HiddenLayer *prev_layer, int idx) {
		this->prev_layer = prev_layer;
		this->idx = idx;
	}
} prev_layer_relation;






#endif /* LAYERCONFIG_H_ */





























