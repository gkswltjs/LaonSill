/*
 * MaxPooling.h
 *
 *  Created on: 2016. 5. 16.
 *      Author: jhkim
 */

#ifndef POOLING_MAXPOOLING_H_
#define POOLING_MAXPOOLING_H_

#include "Pooling.h"
#include <limits>

using namespace std;


class MaxPooling : public Pooling {
public:
	MaxPooling() {}
	virtual ~MaxPooling() {}

	void pool(const pool_dim &pool_d, const cube &input, ucube &pool_map, cube &output) {
		output.set_size(input.n_rows/pool_d.rows, input.n_cols/pool_d.cols, input.n_slices);

		unsigned int i, j, k, l, m;
		unsigned int pool_max_row_index = 0, pool_max_col_index = 0;
		double max = 0;
		unsigned int input_max_row_index = input.n_rows-pool_d.rows+1;
		unsigned int input_max_col_index = input.n_cols-pool_d.cols+1;

		// 전체 feature에 대해
		for(i = 0; i < input.n_slices; i++) {
			pool_map.slice(i).fill(0);
			// input의 row에 대해
			for(j = 0; j < input_max_row_index; j+=pool_d.rows) {
				// input의 col에 대해
				for(k = 0; k < input_max_col_index; k+=pool_d.cols) {
					max = numeric_limits<double>::min();
					pool_max_row_index = 0;
					pool_max_col_index = 0;
					for(l = 0; l < pool_d.rows; l++) {
						for(m = 0; m < pool_d.cols; m++) {
							if(input.slice(i)(j+l, k+m) > max) {
								max = input.slice(i)(j+l, k+m);

								pool_max_row_index = l;
								pool_max_col_index = m;
							}
						}
					}
					output.slice(i)(j/pool_d.rows, k/pool_d.cols) = max;
					pool_map.slice(i)(j+pool_max_row_index, k+pool_max_col_index) = 1;
				}
			}
		}
	}

	void d_pool(const pool_dim &pool_d, const cube &input, ucube &pool_map, cube &output) {
		output.set_size(size(pool_map));

		int i, j, k, l, m;
		for(i = 0; i < input.n_slices; i++) {
			for(j = 0; j < input.n_rows; j++) {
				for(k = 0; k < input.n_cols; k++) {
					for(l = 0; l < pool_d.rows; l++) {
						for(m = 0; m < pool_d.cols; m++) {
							output.slice(i)(j*pool_d.rows+l, k*pool_d.cols+m) = input.slice(i)(j, k)*pool_map.slice(i)(j*pool_d.rows+l, k*pool_d.cols+m);
						}
					}
				}
			}
		}

		Util::printCube(input, "d_pool-input:");
		//Util::printCube(pool_map, "d_pool-pool_map:");
		Util::printCube(output, "d_pool-output:");

	}

};

#endif /* POOLING_MAXPOOLING_H_ */























