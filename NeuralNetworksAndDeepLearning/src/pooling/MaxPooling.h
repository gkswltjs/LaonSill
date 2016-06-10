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
	MaxPooling() {
		this->type = PoolingType::Max;
	}
	virtual ~MaxPooling() {}

	void pool(const pool_dim &pool_d, const rcube &input, ucube &pool_map, rcube &output) {
		UINT i, j, k, l, m;

		int left_pad = (pool_d.rows-1)/2;
		int top_pad = (pool_d.cols-1)/2;
		int in_input_row_idx;
		int in_input_col_idx;
		int pool_max_idx;
		double max;

		pool_map.zeros();

		// input image에 대해
		for(i = 0; i < input.n_slices; i++) {
			//for(j = 0; j < input.n_rows; j+=pool_d.stride) {
			//	for(k = 0; k < input.n_cols; k+=pool_d.stride) {
			for(j = 0; j/pool_d.stride < output.n_rows; j+=pool_d.stride) {
				for(k = 0; k/pool_d.stride < output.n_cols; k+=pool_d.stride) {
					max = numeric_limits<double>::min();
					pool_max_idx = 4;

					// input image의 i, j를 center로 pool 영역만큼 최대값과 위치 찾기
					for(l = 0; l < pool_d.rows; l++) {
						for(m = 0; m < pool_d.cols; m++) {
							in_input_row_idx = j-left_pad+l;
							in_input_col_idx = k-top_pad+m;

							if((in_input_row_idx >= 0 && (UINT)in_input_row_idx < input.n_rows)
									&&(in_input_col_idx >= 0 && (UINT)in_input_col_idx < input.n_cols)) {

								//if(input.slice(i)(in_input_row_idx, in_input_col_idx) > max) {
								if(C_MEM(input, in_input_row_idx, in_input_col_idx, i) > max) {
									//max = input.slice(i)(in_input_row_idx, in_input_col_idx);
									max = C_MEM(input, in_input_row_idx, in_input_col_idx, i);
									pool_max_idx = l*pool_d.cols + m;
								}
							}
						}
					}

					//output.slice(i)(j/pool_d.stride, k/pool_d.stride) = max;
					C_MEMPTR(output, j/pool_d.stride, k/pool_d.stride, i) = max;
					//pool_map.slice(i)(j/pool_d.stride, k/pool_d.stride) = pool_max_idx;
					C_MEMPTR(pool_map, j/pool_d.stride, k/pool_d.stride, i) = pool_max_idx;
				}
			}
		}

		Util::printCube(input, "input:");
		Util::printUCube(pool_map, "pool_map:");
		Util::printCube(output, "output:");
	}

	void d_pool(const pool_dim &pool_d, const rcube &input, ucube &pool_map, rcube &output) {

		UINT i, j, k;
		int pool_max_idx;
		int left_pad = (pool_d.rows-1)/2;
		int top_pad = (pool_d.cols-1)/2;

		//output.set_size(input.n_rows*pool_d.stride, input.n_cols*pool_d.stride, input.n_slices);
		output.zeros();

		for(i = 0; i < input.n_slices; i++) {
			for(j = 0; j < input.n_rows; j++) {
				for(k = 0; k < input.n_cols; k++) {
					//pool_max_idx = pool_map.slice(i)(j, k);
					pool_max_idx = C_MEM(pool_map, j, k, i);
					//output.slice(i)(j*pool_d.stride + pool_max_idx/pool_d.cols-left_pad, k*pool_d.stride + pool_max_idx%pool_d.cols-top_pad)
					//		+= input.slice(i)(j, k);
					//C_MEMPTR(output, (int)(j*pool_d.stride + pool_max_idx/pool_d.cols-left_pad), (int)(k*pool_d.stride + pool_max_idx%pool_d.cols-top_pad), i)
					//		+= C_MEM(input, j, k, i);
					C_MEMPTR(output, (int)(j*pool_d.stride + pool_max_idx/pool_d.cols-left_pad), (int)(k*pool_d.stride + pool_max_idx%pool_d.cols-top_pad), i)
					= C_MEM(output, (int)(j*pool_d.stride + pool_max_idx/pool_d.cols-left_pad), (int)(k*pool_d.stride + pool_max_idx%pool_d.cols-top_pad), i) + C_MEM(input, j, k, i);
				}
			}
		}

		Util::printCube(input, "input:");
		Util::printUCube(pool_map, "pool_map:");
		Util::printCube(output, "output:");


	}

};

#endif /* POOLING_MAXPOOLING_H_ */























