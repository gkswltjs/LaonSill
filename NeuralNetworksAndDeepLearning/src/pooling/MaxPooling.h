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
		int i, j, k, l, m;

		int left_pad = (pool_d.rows-1)/2;
		int top_pad = (pool_d.cols-1)/2;
		int in_input_row_idx;
		int in_input_col_idx;
		int pool_max_idx;
		//int num_input_channel_elem = input.n_rows*input.n_cols;
		//int num_poolmap_channel_elem = pool_map.n_rows*pool_map.n_cols;
		double max;

		//output.set_size(input.n_rows/pool_d.stride, input.n_cols/pool_d.stride, input.n_slices);
		//int num_output_channel_elem = output.n_rows*output.n_cols;

		pool_map.zeros();

		// input image에 대해
		for(i = 0; i < input.n_slices; i++) {
			for(j = 0; j < input.n_rows; j+=pool_d.stride) {
				for(k = 0; k < input.n_cols; k+=pool_d.stride) {
					max = numeric_limits<double>::min();

					// input image의 i, j를 center로 pool 영역만큼 최대값과 위치 찾기
					for(l = 0; l < pool_d.rows; l++) {
						for(m = 0; m < pool_d.cols; m++) {
							in_input_row_idx = j-left_pad+l;
							in_input_col_idx = k-top_pad+m;

							if((in_input_row_idx >= 0 && in_input_row_idx < input.n_rows)
									&&(in_input_col_idx >= 0 && in_input_col_idx < input.n_cols)) {

								if(input.slice(i)(in_input_row_idx, in_input_col_idx) > max) {
									max = input.slice(i)(in_input_row_idx, in_input_col_idx);
									pool_max_idx = l*pool_d.cols + m;
								}

							}

						}
					}

					output.slice(i)(j/pool_d.stride, k/pool_d.stride) = max;
					pool_map.slice(i)(j/pool_d.stride, k/pool_d.stride) = pool_max_idx;
					//output.mem[j/pool_d.stride+(k/pool_d.stride)*output.n_cols+i*num_output_channel_elem] = max;
					//pool_map.mem[j+pool_max_row_idx + (k+pool_max_col_idx)*pool_map.n_cols + i*num_poolmap_channel_elem] = 1;
				}
			}
		}

		Util::printCube(input, "input:");
		//pool_map.print("pool_map:");
		Util::printCube(output, "output:");




		/*
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
		*/
	}

	void d_pool(const pool_dim &pool_d, const cube &input, ucube &pool_map, cube &output) {

		int i, j, k;
		int pool_max_idx;
		int left_pad = (pool_d.rows-1)/2;
		int top_pad = (pool_d.cols-1)/2;

		output.set_size(input.n_rows*pool_d.stride, input.n_cols*pool_d.stride, input.n_slices);
		output.zeros();

		for(i = 0; i < input.n_slices; i++) {
			for(j = 0; j < input.n_rows; j++) {
				for(k = 0; k < input.n_cols; k++) {
					pool_max_idx = pool_map.slice(i)(j, k);
					output.slice(i)(j*pool_d.stride + pool_max_idx/pool_d.cols-left_pad, k*pool_d.stride + pool_max_idx%pool_d.cols-top_pad)
							+= input.slice(i)(j, k);
				}
			}
		}

		Util::printCube(input, "input:");
		//pool_map.print("pool_map:");
		Util::printCube(output, "output:");





		/*
		int left_pad = (pool_d.rows-1)/2;
		int top_pad = (pool_d.cols-1)/2;
		int num_input_elem = input.n_rows*input.n_cols;
		int num_output_elem = pool_map.n_rows*pool_map.n_cols;
		int in_output_row_idx, in_output_col_idx;

		output.set_size(size(pool_map));
		output.zeros();

		int i, j, k, l, m;
		for(i = 0; i < input.n_slices; i++) {

			for(j = 0; j < input.n_rows; j++) {
				for(k = 0; k < input.n_cols; k++) {

					for(l = 0; l < pool_d.rows; l++) {
						for(m = 0; m < pool_d.cols; m++) {

							in_output_row_idx = j*pool_d.rows-left_pad+l;
							in_output_col_idx = k*pool_d.cols-top_pad+m;

							if((in_output_row_idx >= 0 && in_output_row_idx < output.n_rows)
									&& (in_output_col_idx >= 0 && in_output_col_idx < output.n_cols)) {

								output.mem[in_output_row_idx + in_output_col_idx*output.n_cols + i*num_output_elem] =
										input.mem[j + k*input.n_cols + i*num_input_elem] *
										pool_map.mem[in_output_row_idx + in_output_col_idx*output.n_cols + i*num_output_elem];
							}
							//output.slice(i)(j*pool_d.rows+l, k*pool_d.cols+m) = input.slice(i)(j, k)*pool_map.slice(i)(j*pool_d.rows+l, k*pool_d.cols+m);
						}
					}
				}
			}
		}

		//Util::printCube(input, "d_pool-input:");
		//Util::printCube(pool_map, "d_pool-pool_map:");
		//Util::printCube(output, "d_pool-output:");
		 */

	}

};

#endif /* POOLING_MAXPOOLING_H_ */























