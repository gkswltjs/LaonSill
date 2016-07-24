#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;




int main_test(void) {
	//ImageCropper imageCropper("/home/jhkim/image/ILSVRC2012", 224, 1);
	//imageCropper.crop();

	const int rows = 4;
	const int cols = 5;
	const int M = rows*cols;
	const int channels = 3;


	cube a = randu<cube>(rows, cols, channels);

	a.print("a:");

	float temp[M*channels];
	for(int channel = 0; channel < channels; channel++) {
		for(int row = 0; row < rows; row++) {
			for(int col = 0; col < cols; col++) {
				temp[col+row*cols+channel*M] = a.mem[row+col*rows+channel*M];
			}
		}
		cout << endl;
		//for(int i = 0; i < M; i++) {
		//	cout << a.mem[i+channel*M] << ", ";
		//	temp[i+channel*M] = a.mem[i+channel*M];
		//}
		//cout << endl;
	}

	Cube<float> b(temp, 1, M, channels);
	b.print("b:");






	//a.reshape(1, M, channels);
	//a.print("a.reshape:");
	//a.resize(1, M, channels);
	//a.print("a.resize");
	//a.reshape(M, 1, channels);
	//a.print("a.reshape:");








	/*
	const int cols = 7;
	const int rows = 7;
	const int M = cols*rows;
	const int N = 3;

	for(int i = 0; i < M; i++) {
		for(int j = 0; j < N; j++) {
			int delta_index = i*N+j;
			//style_delta[delta_index] = 0.0;
			//style_delta_temp[delta_index] = 0.0;

			cout << "style_delta for " << delta_index << " is 0 ... " << endl;

			int row = (int)(i/rows);
			int col = (int)(i%rows);
			cout << "if > 0, style_delta for row " << row << ", col " << col << " ie, " << row+col*cols+j*M << " is has value ... " << endl;
		}
	}
	*/
	return 0;
}
