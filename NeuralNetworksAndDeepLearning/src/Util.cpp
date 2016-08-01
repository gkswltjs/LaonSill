/*
 * Util.cpp
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#include "Util.h"
#include "exception/Exception.h"

#include <cstdlib>
#include <stdarg.h>

int Util::page = 0;
int Util::start_page = 1000;
int Util::end_page = 1500;
bool Util::temp_flag = false;
bool Util::print = true;
size_t Util::cuda_mem = 0;
int Util::alloc_cnt = 0;
ostream *Util::outstream = &cout;


static const char *LEVEL_LABEL[4] = {"DEBUG","INFO ", "WARN ", "ERROR"};


void log_print(FILE *fp, int log_level, const char* filename, const int line, const char *func, char *fmt, ...) {
	if(fp != NULL) {
		char log[1024];
		int log_index = 0;
		time_t t;
		va_list list;

		time(&t);

		sprintf(log, "[%s", ctime(&t));
		log_index = strlen(log)-1;

		sprintf(&log[log_index], "] (%s) %s:%d %s(): ", LEVEL_LABEL[log_level], filename, line, func);
		log_index = strlen(log);

		va_start(list, fmt);
		vsprintf(&log[log_index], fmt, list );
		va_end(list);

		fprintf(fp, log);
		fputc('\n', fp);

		fflush(fp);
	}
}





int Util::random(int min, int max)
{
	return (int)rand() * (max-min) + min;
}


int Util::pack4BytesToInt(unsigned char *buffer)
{
	int result = 0;
	for(int i = 0; i < 4; i++) {
		result += buffer[i] << 8*(3-i);
	}
	return result;
}

void Util::printVec(const rvec &vector, string name) {
	if(Util::print) {
		(*outstream) << "-------------------------------------" << endl;
		(*outstream) << "name: " << name << endl;
		(*outstream) << "address: " << &vector << endl;
		(*outstream) << "rows x cols: " << vector.n_rows << " x " << vector.n_cols << endl;
		vector.print("vec values: ");
		(*outstream) << endl;
		(*outstream) << "-------------------------------------" << endl;
	}
}

void Util::printMat(const rmat &matrix, string name) {
	if(Util::print) {
		(*outstream) << "-------------------------------------" << endl;
		(*outstream) << "name: " << name << endl;
		(*outstream) << "address: " << &matrix << endl;
		(*outstream) << "rows x cols: " << matrix.n_rows << " x " << matrix.n_cols << endl;
		matrix.raw_print((*outstream), "mat values: ");
		(*outstream) << endl;
		(*outstream) << "-------------------------------------" << endl;
	}
}

void Util::printCube(const rcube &c, string name) {
	if(Util::print) {
		(*outstream) << "-------------------------------------" << endl;
		(*outstream) << "name: " << name << endl;
		(*outstream) << "address: " << &c << endl;
		(*outstream) << "rows x cols x slices: " << c.n_rows << " x " << c.n_cols << " x " << c.n_slices << endl;
		c.raw_print((*outstream), "cube values: ");
		(*outstream) << endl;
		(*outstream) << "-------------------------------------" << endl;
	}
}

void Util::printUCube(const ucube &c, string name) {
	if(Util::print) {
		(*outstream) << "-------------------------------------" << endl;
		(*outstream) << "name: " << name << endl;
		(*outstream) << "address: " << &c << endl;
		(*outstream) << "rows x cols x slices: " << c.n_rows << " x " << c.n_cols << " x " << c.n_slices << endl;
		c.raw_print((*outstream), "cube values: ");
		(*outstream) << endl;
		(*outstream) << "-------------------------------------" << endl;
	}
}


void Util::printData(const DATATYPE *data, UINT rows, UINT cols, UINT channels, UINT batches, string name) {
	if(Util::print && data) {
		UINT i,j,k,l;

		(*outstream) << "-------------------------------------" << endl;
		(*outstream) << "name: " << name << endl;
		(*outstream) << "rows x cols x channels x batches: " << rows << " x " << cols << " x " << channels << " x " << batches << endl;

		UINT batchElem = rows*cols*channels;
		UINT channelElem = rows*cols;
		for(i = 0; i < batches; i++) {
			for(j = 0; j < channels; j++) {
				for(k = 0; k < rows; k++) {
					for(l = 0; l < cols; l++) {
				//for(k = 0; k < std::min(10, (int)rows); k++) {
				//	for(l = 0; l < std::min(10, (int)cols); l++) {
						(*outstream) << data[i*batchElem + j*channelElem + l*rows + k] << ", ";
					}
					(*outstream) << endl;
				}
				(*outstream) << endl;
			}
			(*outstream) << endl;
		}

		//for(i = 0; i < std::min(10, (int)(rows*cols*channels*batches)); i++) {
		//	(*outstream) << data[i] << ",";
		//}
		//(*outstream) << endl;

		(*outstream) << "-------------------------------------" << endl;
	}
}


void Util::printDeviceData(const DATATYPE *d_data, UINT rows, UINT cols, UINT channels, UINT batches, string name) {
	if(Util::print) {
		Cuda::refresh();

		DATATYPE *data = new DATATYPE[rows*cols*channels*batches];
		checkCudaErrors(cudaMemcpyAsync(data, d_data, sizeof(DATATYPE)*rows*cols*channels*batches, cudaMemcpyDeviceToHost));
		Util::printData(data, rows, cols, channels, batches, name);
		// TODO 메모리 해제시 print가 제대로 동작하지 않음
		//if(data) delete [] data;
	}
}


void Util::printMessage(string message) {
	//if(true || Util::print) {
	if(Util::print) {
		(*outstream) << message << endl;
	}
}



/*
void Util::convertCubeToVec(const io_dim &cube_dim, const cube &c, vec &v) {
	if(cube_dim.channels == 1) {
		if(cube_dim.cols == 1) v = c.slice(0);
		else v = vectorise(c.slice(0).t());
	} else if(cube_dim.channels > 1) {
		mat temp;
		temp = join_cols(c.slice(0), c.slice(1));
		for(unsigned int i = 2; i < c.n_slices; i++) {
			temp = join_cols(temp, c.slice(i));
		}
		v = vectorise(temp.t());
	}
}
*/


void Util::convertCube(const rcube &input, rcube &output) {
	// input, output의 dim이 동일한 경우, 변환이 필요없음, input을 output으로 그대로 전달
	if(size(input) == size(output)) {
		output = input;
		return;
	}

	// 두 매트릭스의 elem의 수가 같아야 함
	// 둘 중 하나는 vector여야 함 (vector로, 또는 vector로부터의 변환만 현재 지원)
	if(input.size() != output.size() ||
			!((input.n_cols==1&&input.n_slices==1)||(output.n_cols==1&&output.n_slices==1))) {
		throw Exception();
	}



	// output이 vector인 경우
	if(output.n_cols==1&&output.n_slices==1) {
		output = reshape(input, output.size(), 1, 1, 1);
		return;
	}

	// input이 vector인 경우
	if(input.n_cols==1&&input.n_slices==1) {
		rcube temp = reshape(input, output.n_cols, output.n_rows, output.n_slices);
		for(unsigned int i = 0; i < output.n_slices; i++) {
			output.slice(i) = temp.slice(i).t();
		}
		return;
	}
}


void Util::dropoutLayer(rcube &input, double p_dropout) {
	rcube p = randu<rcube>(size(input));
	//Util::printCube(p, "p:");
	//Util::printCube(input, "input:");

	UINT slice, row, col;
	for(slice = 0; slice < input.n_slices; slice++) {
		for(row = 0; row < input.n_rows; row++) {
			for(col = 0; col < input.n_cols; col++) {
				//if(p.slice(slice)(row, col) < p_dropout) input.slice(slice)(row, col) = 0;
				if(C_MEM(p, row, col, slice) < p_dropout) C_MEMPTR(input, row, col, slice) = 0;
			}
		}
	}
	//Util::printCube(input, "input:");
}













