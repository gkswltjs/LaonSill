/*
 * LayerConfig.h
 *
 *  Created on: 2016. 5. 14.
 *      Author: jhkim
 */

#ifndef LAYERCONFIG_H_
#define LAYERCONFIG_H_


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

	filter_dim() {
		this->filters = 1;
	}
	filter_dim(int rows, int cols, int channels, int filters) : io_dim(rows, cols, channels) {
		this->filters = filters;
	}
} filter_dim;

typedef struct pool_dim {
	int rows;
	int cols;

	pool_dim() {
		this->rows = 1;
		this->cols = 1;
	}
	pool_dim(int rows, int cols) {
		this->rows = rows;
		this->cols = cols;
	}
} pool_dim;




#endif /* LAYERCONFIG_H_ */





























