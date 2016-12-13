/**
 * @file Atari.cpp
 * @date 2016-12-08
 * @author moonhoen lee
 * @brief 
 * @details
 */
#include <SDL.h>
#include <CImg.h>

#include "ale_interface.hpp"
#include "common/ColourPalette.hpp"

#include "common.h"
#include "Atari.h"
#include "SysLog.h"

using namespace std;
using namespace cimg_library;

#define USE_CIMG_DISPLAY    1

void Atari::run(char* romFilePath) {
    // (5-C-1)
    ALEInterface ale;

    // Get & Set the desired settings
    ale.setInt("random_seed", 123);
    //The default is already 0.25, this is just an example
    ale.setFloat("repeat_action_probability", 0.25);

    ale.setBool("display_screen", true);
    ale.setBool("sound", true);

    // Load the ROM file. (Also resets the system for new settings to
    // take effect.)
    ale.loadROM(romFilePath);

    ALEScreen screen = ale.getScreen();
    int screenWidth = screen.width();
    int screenHeight = screen.height();

    CImg<pixel_t> imgData(screenWidth, screenHeight, 1, 3);
    CImg<pixel_t> imgData2(80, 80, 1, 3);
#ifdef USE_CIMG_DISPLAY
    CImgDisplay imgDisplay(imgData, "Original Image");
    CImgDisplay imgDisplay2(imgData2, "Scaled Image");
#endif
    pixel_t *imgPixels = (pixel_t*)malloc(sizeof(pixel_t) * screenWidth * screenHeight * 3); 
    SASSERT0(imgPixels != NULL);
    ColourPalette colourPalette;
    colourPalette.setPalette("standard", "");

    int allocSize = sizeof(float) * screenWidth * screenHeight * 4;
    float *img = (float*)malloc(allocSize);
    SASSERT0(img != NULL);

    // Get the vector of legal actions
    ActionVect legal_actions = ale.getLegalActionSet();

    // Play 10 episodes
    for (int episode = 0; episode < 10; episode++) {
        float totalReward = 0;
        while (!ale.game_over()) {
            screen = ale.getScreen();

            for (int i = 0; i < screenWidth; i++) {
                for (int j = 0; j < screenHeight; j++) {
                    int r, g, b;
                    colourPalette.getRGB(screen.get(j, i), r, g, b);
                    imgData(i, j, 0, 0) = r;
                    imgData(i, j, 0, 1) = g;
                    imgData(i, j, 0, 2) = b;
                }
            }
       
            imgData2 = imgData;
            imgData2.resize(80, 80, 1, 3, 3);

            // fill img
            //  이 방법이 최선일까?
            // (1) assign image[i] = image[i+1]  for i = 0..2
            //    (I do not use memmove(). It is very dangerous function as far as i know)
            for (int i = 0; i < 3; i++) {
                int srcIndex = (i + 1) * screenWidth * screenHeight;
                int dstIndex = i * screenWidth * screenHeight;
                int copySize = screenWidth * screenHeight * sizeof(float);
                memcpy((void*)&img[dstIndex], (void*)&img[srcIndex], copySize);
            }

            // (2) fill image[3]
            //     by column major order
            int imgOffset = 3 * screenWidth * screenHeight;
            for (int i = 0; i < screenHeight; i++) {
                for (int j = 0; j < screenWidth; j++) {
                    // [ITU BT.709] RGB to Luminance
                    img[imgOffset] = ((float)imgData2(j, i, 0, 0) * 0.2126 +  
                                      (float)imgData2(j, i, 0, 1) * 0.7152 + 
                                      (float)imgData2(j, i, 0, 2) * 0.0722) / 255.0;
                    imgOffset++;
                }
            }

#ifdef USE_CIMG_DISPLAY
            imgDisplay = imgData;
            imgDisplay2 = imgData2;
#endif
            
            Action a = legal_actions[rand() % legal_actions.size()];
            // Apply the action and get the resulting reward
            float reward = ale.act(a);
            totalReward += reward;
        }
        cout << "Episode " << episode << " ended with score: " << totalReward << endl;
        ale.reset_game();
    }
}
