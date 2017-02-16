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

	Data<float>* data = new Data<float>("data");
	SyncMem<float>* result = new SyncMem<float>();
	data->reshape({1, 3, 4, 5});

	float buf[3*4*5] = {
			0.93301918,  0.85572464,  0.53074557,  0.55668099,
			 0.43905592,  0.38570177,  0.11129506,  0.6400981 ,
			 0.75371358,  0.73441122,  0.40722504,  0.45642823,
			 0.44346879,  0.45388681,  0.65185627,  0.69507116,
			 0.29838659,  0.32703678,  0.34482918,  0.19069375,

			0.16940107,  0.51810323,  0.86771267,  0.05425753,
			 0.07832988,  0.24952664,  0.25265331,  0.16491378,
			 0.96545931,  0.94063219,  0.34883184,  0.01240625,
			 0.90434571,  0.27640548,  0.88474852,  0.84949007,
			 0.69732465,  0.518344  ,  0.19846743,  0.57705033,

			0.70128302,  0.52730274,  0.44268469,  0.37660983,
			 0.41909051,  0.65054396,  0.41915488,  0.71233381,
			 0.06136083,  0.17017925,  0.71443403,  0.77168627,
			 0.0886824 ,  0.50155922,  0.51394144,  0.59506308,
			 0.12115273,  0.45954319,  0.91505774,  0.21416636
	};

	data->set_host_data(buf);

	artisticStyle->on();
	data->print_data({}, true);
	artisticStyle->off();


	artisticStyle->createGramMatrixFromData(data, result);

	if (false)
		artisticStyle->style();

	delete artisticStyle;
}



void setup() {

}

void cleanup() {

}

#endif
