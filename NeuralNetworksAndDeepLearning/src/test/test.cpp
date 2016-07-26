#include <iostream>

#include "../util/ImagePacker.h"

using namespace std;



int main_test(void) {
	ImagePacker imagePacker("/home/jhkim/image/ILSVRC2012/crop_sample", 1000, 0, 0, 0);
	imagePacker.load();
	imagePacker.show();
	return 0;
}
