/*
 * AvgPooling.h
 *
 *  Created on: 2016. 5. 24.
 *      Author: jhkim
 *
 *
 *
 * input의 좌상단을 pool 영역의 좌상단에 맞춰 pooling
 * overlap되지 않게 pooling
 * input.n_rows / pool_d.rows 만큼 down sample
 *
 */

#ifndef POOLING_AVGPOOLING_H_
#define POOLING_AVGPOOLING_H_

#include "Pooling.h"

using namespace std;


class AvgPooling : public Pooling {
public:
	AvgPooling() {}
	virtual ~AvgPooling() {}

	void pool(const pool_dim &pool_d, const cube &input, ucube &pool_map, cube &output) {
		int i, j, k, l, m;

		int left_pad = (pool_d.rows-1)/2;
		int top_pad = (pool_d.cols-1)/2;
		int in_input_row_idx;
		int in_input_col_idx;
		int num_pool_elem = pool_d.rows*pool_d.cols;
		double sum;

		//output.set_size(input.n_rows/pool_d.stride, input.n_cols/pool_d.stride, input.n_slices);
		int num_output_channel_elem = output.n_rows*output.n_cols;

		//pool_map.zeros();
		Util::printCube(input, "input:");


		// GoogLeNet에서 Average Pooling의 경우 image의 첫 위치가 아닌
		// left, top pad를 포함하여 offset된 위치에서 시작되도록 계산하는 게 맞아 보여 일단 그렇게 작성
		// (stride도 left, top pad를 따르나 일단 사용자로부터 입력받도록 둠
		// 아주 일반적이지 않을 수 있음, 추후의 케이스에 수정이 필요할 수 있음.

		// input image에 대해
		for(i = 0; i < input.n_slices; i++) {
			for(j = left_pad; j < input.n_rows; j+=pool_d.rows) {
				for(k = top_pad; k < input.n_cols; k+=pool_d.cols) {
					sum = 0.0;

					// input image의 i, j를 center로 pool 영역만큼 최대값과 위치 찾기
					for(l = 0; l < pool_d.rows; l++) {
						for(m = 0; m < pool_d.cols; m++) {
							in_input_row_idx = j-left_pad+l;
							in_input_col_idx = k-top_pad+m;

							if((in_input_row_idx >= 0 && in_input_row_idx < input.n_rows)
								&&(in_input_col_idx >= 0 && in_input_col_idx < input.n_cols)) {
								sum += input.slice(i)(in_input_row_idx, in_input_col_idx);
							}
						}
					}
					sum /= num_pool_elem;
					output.slice(i)(j/pool_d.rows, k/pool_d.cols) = sum;
				}
			}
			Util::printMat(output.slice(i), "output:");
		}
	}

	void d_pool(const pool_dim &pool_d, const cube &input, ucube &pool_map, cube &output) {
		int i, j, k, l, m;
		double num_pool_elem_factor = 1.0/(pool_d.rows*pool_d.cols);

		output.set_size(input.n_rows*pool_d.rows, input.n_cols*pool_d.cols, input.n_slices);
		//output.zeros();

		Util::printCube(input, "input:");


		for(i = 0; i < input.n_slices; i++) {

			// for down-sampled delta map
			for(j = 0; j < input.n_rows; j++) {
				for(k = 0; k < input.n_cols; k++) {

					// for up-sample map
					for(l = j*pool_d.rows; l < (j+1)*pool_d.rows; l++) {
						for(m = k*pool_d.cols; m < (k+1)*pool_d.cols; m++) {
							output.slice(i)(l, m) = num_pool_elem_factor*input.slice(i)(j, k);
						}
					}
				}
			}
		}

		Util::printCube(output, "output:");
	}
};

#endif /* POOLING_AVGPOOLING_H_ */
