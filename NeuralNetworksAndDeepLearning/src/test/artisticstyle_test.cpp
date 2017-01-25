#define ARTISTICSTYLE_TEST 0
#if ARTISTICSTYLE_TEST

#include "ArtisticStyle.h"



using namespace std;



void setup();
void cleanup();




int main() {

	ArtisticStyle<float>* artisticStyle = new ArtisticStyle<float>(
			0,
			//"/data/backup/artistic/tubingen_320.jpg",
			//"/data/backup/artistic/starry_night_320.jpg",
			"/home/jkim/Downloads/sampleR32G64B128.png",
			"/home/jkim/Downloads/sampleR32G64B128.png",
			{"convLayer1"},
			{"fc1"},
			0.001f,					// contentReconstructionFactor
			1.0f,					// styleReconstructionFactor
			0.1f,					// learningRate
			"fc1",					// end
			true,					// plotContentCost
			true					// plotStyleCost
	);

	if (!false)
		artisticStyle->style();

	delete artisticStyle;
}



void setup() {

}

void cleanup() {

}

#endif
