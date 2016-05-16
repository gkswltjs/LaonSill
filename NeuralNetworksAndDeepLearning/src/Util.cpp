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






















