/*
 * Util.h
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#ifndef UTIL_H_
#define UTIL_H_

#include "layer/LayerConfig.h"
#include <string>
#include <armadillo>

using namespace std;
using namespace arma;


class Util {
public:
	Util();
	virtual ~Util();

	static int random(int min, int max);
	static int pack4BytesToInt(unsigned char *buffer);
	static void printVec(const vec &vector, string name);
	static void printMat(const mat &matrix, string name);
	static void printCube(const cube &c, string name);

	static int getPrint() { return Util::print; }
	static void setPrint(bool print) { Util::print = print; };

	//static void convertCubeToVec(const io_dim &cube_dim, const cube &c, vec &v);

	/**
	 * input cube를 output_dim을 따르는 cube output으로 변환
	 * @param output_dim
	 * @param input
	 * @param output
	 */
	static void convertCube(const cube &input, cube &output);

	static void dropoutLayer(cube &input, double p_dropout);

private:
	static bool print;

};

#endif /* UTIL_H_ */
