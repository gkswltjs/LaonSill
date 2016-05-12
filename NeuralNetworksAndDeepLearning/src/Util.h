/*
 * Util.h
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#ifndef UTIL_H_
#define UTIL_H_

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

	static int getPrint() { return Util::print; }
	static void setPrint(bool print) { Util::print = print; };

private:
	static bool print;

};

#endif /* UTIL_H_ */
