/*
 * LayerConfig.h
 *
 *  Created on: 2016. 5. 14.
 *      Author: jhkim
 */

#ifndef LAYERCONFIG_H_
#define LAYERCONFIG_H_

#include "../Util.h"
#include <armadillo>


//typedef arma::fvec rvec;
//typedef arma::fmat rmat;
//typedef arma::fcube rcube;
//typedef unsigned int UINT;


class Layer;





enum class ParamFillerType {
	None, Constant, Xavier, Gaussian
};



typedef struct io_dim {
    UINT rows;
    UINT cols;
    UINT channels;

    io_dim() {
    	this->rows = 1;
    	this->cols = 1;
    	this->channels = 1;
    }
    io_dim(UINT rows, UINT cols, UINT channels) {
    	this->rows = rows;
    	this->cols = cols;
    	this->channels = channels;
    }
    int size() const { return rows*cols*channels; }
} io_dim;

typedef struct filter_dim : public io_dim {
	UINT filters;
	UINT stride;

	filter_dim() {
		this->filters = 1;
		this->stride = 1;
	}
	filter_dim(UINT rows, UINT cols, UINT channels, UINT filters, UINT stride) : io_dim(rows, cols, channels) {
		this->filters = filters;
		this->stride = stride;
	}
} filter_dim;

typedef struct pool_dim {
	UINT rows;
	UINT cols;
	UINT stride;

	pool_dim() {
		this->rows = 1;
		this->cols = 1;
		this->stride = 1;
	}
	pool_dim(UINT rows, UINT cols, UINT stride) {
		this->rows = rows;
		this->cols = cols;
		this->stride = stride;
	}
} pool_dim;


typedef struct lrn_dim {
	UINT local_size;
	double alpha;
	double beta;

	lrn_dim() {
		this->local_size = 5;
		this->alpha = 1;
		this->beta = 5;
	}
	lrn_dim(UINT local_size, double alpha, double beta) {
		this->local_size = local_size;
		this->alpha = alpha;
		this->beta = beta;
	}
} lrn_dim;


typedef struct update_param {
	double lr_mult;
	double decay_mult;

	update_param() {}
	update_param(double lr_mult, double decay_mult) {
		this->lr_mult = lr_mult;
		this->decay_mult = decay_mult;
	}
} update_param;

typedef struct param_filler {
	ParamFillerType type;
	double value;

	param_filler() {}
	param_filler(ParamFillerType type, double value=0) {
		this->type = type;
		this->value = value;
	}

	void fill(rvec &param, int n_in) {
		switch(type) {
		case ParamFillerType::Constant: param.fill(value); break;
		case ParamFillerType::Xavier:
			param.randn();
			param *= sqrt(3.0/n_in);
			break;
		case ParamFillerType::Gaussian:
			param.randn();
			break;
		case ParamFillerType::None:
		default:
			break;
		}
	}

	void fill(rmat &param, int n_in) {
		switch(type) {
		case ParamFillerType::Constant:
			param.fill(value);
			break;
		case ParamFillerType::Xavier:
			param.randn();
			param *= sqrt(3.0/n_in);				// initial point scaling
			break;
		case ParamFillerType::Gaussian:
			param.randn();
			break;
		case ParamFillerType::None:
		default:
			break;
		}
	}

	void fill(rcube &param, int n_in) {
		switch(type) {
		case ParamFillerType::Constant:
			param.fill(value);
			break;
		case ParamFillerType::Xavier:
			param.randn();
			param *= sqrt(3.0/n_in);				// initial point scaling
			break;
		case ParamFillerType::Gaussian:
			param.randn();
			break;
		case ParamFillerType::None:
		default:
			break;
		}
	}

} param_filler;



typedef struct next_layer_relation {
	Layer *next_layer;
	UINT idx;

	next_layer_relation() {}
	next_layer_relation(Layer *next_layer, UINT idx) {
		this->next_layer = next_layer;
		this->idx = idx;
	}
} next_layer_relation;

typedef struct prev_layer_relation {
	Layer *prev_layer;
	UINT idx;

	prev_layer_relation() {}
	prev_layer_relation(Layer *prev_layer, UINT idx) {
		this->prev_layer = prev_layer;
		this->idx = idx;
	}
} prev_layer_relation;









#endif /* LAYERCONFIG_H_ */





























