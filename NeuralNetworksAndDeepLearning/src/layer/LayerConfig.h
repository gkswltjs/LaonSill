/*
 * LayerConfig.h
 *
 *  Created on: 2016. 5. 14.
 *      Author: jhkim
 */

#ifndef LAYERCONFIG_H_
#define LAYERCONFIG_H_

#include <cmath>
#include <chrono>
#include <random>

#include "../Util.h"


//typedef arma::fvec rvec;
//typedef arma::fmat rmat;
//typedef arma::fcube rcube;
//typedef unsigned int UINT;


class Layer;





enum class ParamFillerType {
	None, Constant, Xavier, Gaussian
};



struct io_dim {
    UINT rows;
    UINT cols;
    UINT channels;
    UINT batches;

    //io_dim(UINT rows=1, UINT cols=1, UINT channels=1, UINT batches=1) {
    io_dim() {

    }
    io_dim(UINT rows, UINT cols, UINT channels, UINT batches) {
    	this->rows = rows;
    	this->cols = cols;
    	this->channels = channels;
    	this->batches = batches;
    }
    //int size() const { return rows*cols*channels*batches; }
    int unitsize() const { return rows*cols*channels; }
    int batchsize() const { return rows*cols*channels*batches; }
};

struct filter_dim : public io_dim {
	UINT filters;
	UINT stride;

	//filter_dim(UINT rows=1, UINT cols=1, UINT channels=1, UINT filters=1, UINT stride=1) : io_dim(rows, cols, channels) {
	filter_dim() {}
	filter_dim(UINT rows, UINT cols, UINT channels, UINT filters, UINT stride) : io_dim(rows, cols, channels, 1) {
		this->filters = filters;
		this->stride = stride;
	}
	int size() const { return rows*cols*channels*filters; }
};

struct pool_dim {
	UINT rows;
	UINT cols;
	UINT stride;

	//pool_dim(UINT rows=1, UINT cols=1, UINT stride=1) {
	pool_dim() {}
	pool_dim(UINT rows, UINT cols, UINT stride) {
		this->rows = rows;
		this->cols = cols;
		this->stride = stride;
	}
};


struct lrn_dim {
	UINT local_size;
	double alpha;
	double beta;
	double k;

	//lrn_dim(UINT local_size=5, double alpha=1, double beta=5) {
	lrn_dim(UINT local_size=5, double alpha=0.0001, double beta=0.75, double k=2.0) {
		this->local_size = local_size;
		this->alpha = alpha;
		this->beta = beta;
		this->k = k;
	}
};


struct update_param {
	double lr_mult;
	double decay_mult;

	update_param() {}
	update_param(double lr_mult, double decay_mult) {
		this->lr_mult = lr_mult;
		this->decay_mult = decay_mult;
	}
};

struct param_filler {
	ParamFillerType type;
	double value;

	param_filler() {}
	param_filler(ParamFillerType type, double value=0) {
		this->type = type;
		this->value = value;
	}

#if CPU_MODE
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

#else
	void fill(DATATYPE *param, int size, int n_in, int n_out) {
		UINT i;
		switch(type) {
		case ParamFillerType::Constant:
		{
			//memset(param, value, size);
			for(int i = 0; i < size; i++) param[i] = value;
		}
			break;

		// ret = Nd4j.randn(order, shape).divi(FastMath.sqrt(shape[0] + shape[1]));
		// N(0, 1), {channel in, channel out, kernel x, kernel y},
		case ParamFillerType::Xavier:
		{
			std::random_device rd_xavier;
			std::mt19937 gen_xavier(rd_xavier());
			std::normal_distribution<DATATYPE> normal_dist(0.0, 1.0);
			float sd_xavier = sqrt(1.0f / (n_in+n_out));
			for(i = 0; i < size; i++) param[i] = normal_dist(gen_xavier)*sd_xavier;
		}
			break;
		case ParamFillerType::Gaussian:
		{
			std::random_device rd_gaussian;
			std::mt19937 gen_gaussian(rd_gaussian());
			std::normal_distribution<DATATYPE> normal_dist(0.0, 1.0);
			float sd_gaussian = 1.0f/n_out;
			for(i = 0; i < size; i++) param[i] = (normal_dist(gen_gaussian)-0.5f)*sd_gaussian;
		}
			break;
		case ParamFillerType::None:
		default:
			break;
		}
	}
#endif

};



struct next_layer_relation {
	Layer *next_layer;
	UINT idx;

	next_layer_relation() {}
	next_layer_relation(Layer *next_layer, UINT idx) {
		this->next_layer = next_layer;
		this->idx = idx;
	}
};

struct prev_layer_relation {
	Layer *prev_layer;
	UINT idx;

	prev_layer_relation() {}
	prev_layer_relation(Layer *prev_layer, UINT idx) {
		this->prev_layer = prev_layer;
		this->idx = idx;
	}
};









#endif /* LAYERCONFIG_H_ */





























