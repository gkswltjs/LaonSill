/*
 * Util.cpp
 *
 *  Created on: 2016. 4. 20.
 *      Author: jhkim
 */

#include "Util.h"
#include "exception/Exception.h"

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

void Util::printCube(const cube &c, string name) {
	if(Util::print) {
		cout << "-------------------------------------" << endl;
		cout << "name: " << name << endl;
		cout << "address: " << &c << endl;
		cout << "rows x cols x slices: " << c.n_rows << " x " << c.n_cols << " x " << c.n_slices << endl;
		c.print("cube values: ");
		cout << endl;
		cout << "-------------------------------------" << endl;
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


void Util::convertCube(const cube &input, cube &output) {
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
		cube temp = reshape(input, output.n_cols, output.n_rows, output.n_slices);
		for(unsigned int i = 0; i < output.n_slices; i++) {
			output.slice(i) = temp.slice(i).t();
		}
		return;
	}
}




















