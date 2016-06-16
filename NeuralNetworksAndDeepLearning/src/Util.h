/*
 * Util.h
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#ifndef UTIL_H_
#define UTIL_H_

//#include "layer/LayerConfig.h"
#include <string>
#include <armadillo>
#include <iostream>

using namespace std;
using namespace arma;


#define	LOG_DEBUG	0
#define	LOG_INFO	1
#define	LOG_WARN	2
#define	LOG_ERROR	3

#define LOG(fp, log_level, ...) log_print(fp, log_level, __FILE__, __LINE__, __func__, __VA_ARGS__ )
void log_print(FILE *fp, int log_level, const char* filename, const int line, const char *func, char *fmt, ...);


#define C_MEM(cb, r, c, s) cb.mem[r + c*cb.n_rows + s*cb.n_elem_slice]
#define C_MEMPTR(cb, r, c, s) cb.memptr()[r + c*cb.n_rows + s*cb.n_elem_slice]
#define M_MEM(m, r, c) m.mem[r + c*m.n_rows]
#define M_MEMPTR(m, r, c) m.memptr()[r + c*m.n_rows]



#define CPU_MODE	0




typedef fvec rvec;
typedef fmat rmat;
typedef fcube rcube;

typedef unsigned int UINT;


typedef float DATATYPE;


//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)




/*
enum class ActivationType {
	Sigmoid, Softmax, ReLU
};

enum class CostType {
	CrossEntropy, LogLikelihood, Quadratic
};
*/







class Util {
public:
	Util() {}
	virtual ~Util() {}

	static int random(int min, int max);
	static int pack4BytesToInt(unsigned char *buffer);
	static void printVec(const rvec &vector, string name);
	static void printMat(const rmat &matrix, string name);
	static void printCube(const rcube &c, string name);
	static void printUCube(const ucube &c, string name);

	static int getPrint() { return Util::print; }
	static void setPrint(bool print) { Util::print = print; };

	//static void convertCubeToVec(const io_dim &cube_dim, const cube &c, vec &v);

	/**
	 * input cube를 output_dim을 따르는 cube output으로 변환
	 * @param output_dim
	 * @param input
	 * @param output
	 */
	static void convertCube(const rcube &input, rcube &output);

	static void dropoutLayer(rcube &input, double p_dropout);




private:
	static bool print;

};

#endif /* UTIL_H_ */
