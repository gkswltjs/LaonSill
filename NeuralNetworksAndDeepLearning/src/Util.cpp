/*
 * Util.cpp
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#include "Util.h"

#include <cstdlib>


bool Util::print = true;



Util::Util() {
	// TODO Auto-generated constructor stub

}

Util::~Util() {
	// TODO Auto-generated destructor stub
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

void Util::printVec(const vec &vector, string name) {
	if(Util::print) {
		cout << "-------------------------------------" << endl;
		cout << "name: " << name << endl;
		cout << "address: " << &vector << endl;
		cout << "rows x cols: " << vector.n_rows << " x " << vector.n_cols << endl;
		vector.print("vec values: ");
		cout << endl;
		cout << "-------------------------------------" << endl;
	}
}

void Util::printMat(const mat &matrix, string name) {
	if(Util::print) {
		cout << "-------------------------------------" << endl;
		cout << "name: " << name << endl;
		cout << "address: " << &matrix << endl;
		cout << "rows x cols: " << matrix.n_rows << " x " << matrix.n_cols << endl;
		matrix.print("mat values: ");
		cout << endl;
		cout << "-------------------------------------" << endl;
	}

}























