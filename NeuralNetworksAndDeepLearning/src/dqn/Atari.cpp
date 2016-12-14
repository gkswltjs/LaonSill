/**
 * @file Atari.cpp
 * @date 2016-12-08
 * @author moonhoen lee
 * @brief 
 * @details
 */

#define USE_OPENCV              1
//#define SHOW_OPENCV_WINDOW      1
#define USE_RESTRICT_ACTION     1

#include <SDL.h>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#else
#include <CImg.h>
#endif

#include "ale_interface.hpp"
#include "common/ColourPalette.hpp"

#include "common.h"
#include "Atari.h"
#include "SysLog.h"
#include "AtariNN.h"

using namespace std;
//using namespace cimg_library;
//using namespace cv;

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

#ifdef USE_OPENCV
    cv::Mat imgData(screenHeight, screenWidth, CV_8UC3);
    cv::Mat imgData2;
    cv::Mat imgData3;
#ifdef SHOW_OPENCV_WINDOW
    cv::namedWindow("test image");
    cv::namedWindow("test image2");
#endif  /* ifdef SHOW_OPENCV_WINDOWS */
#else
    cimg_library::CImg<pixel_t> imgData(screenWidth, screenHeight, 1, 3);
    cimg_library::CImg<pixel_t> imgData2(84, 84, 1, 3);

    cimg_library::CImgDisplay imgDisplay(imgData, "Original Image");
    cimg_library::CImgDisplay imgDisplay2(imgData2, "Scaled Image");
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

    // create network
    AtariNN *nn = new AtariNN();

    nn->createNetwork();
    nn->buildDQNLayer();

    // Play 10 episodes
    for (int episode = 0; episode < 10; episode++) {
        float totalReward = 0;
        while (!ale.game_over()) {
            screen = ale.getScreen();

            for (int i = 0; i < screenWidth; i++) {
                for (int j = 0; j < screenHeight; j++) {
                    int r, g, b;
                    colourPalette.getRGB(screen.get(j, i), r, g, b);
#ifdef USE_OPENCV
                    // BGR
                    imgData.at<cv::Vec3b>(j, i)[0] = b;
                    imgData.at<cv::Vec3b>(j, i)[1] = g;
                    imgData.at<cv::Vec3b>(j, i)[2] = r;
#else
                    imgData(i, j, 0, 0) = r;
                    imgData(i, j, 0, 1) = g;
                    imgData(i, j, 0, 2) = b;
#endif
                }
            }
       
#ifdef USE_OPENCV
            cv::resize(imgData, imgData2, cv::Size(84, 84));
            cv::cvtColor(imgData2, imgData3, CV_BGR2GRAY);
#ifdef SHOW_OPENCV_WINDOW
            cv::imshow("test image", imgData2);
            cv::imshow("test image2", imgData3);
            cv::waitKey(1);
#endif
#else
            imgData2 = imgData;
            // resize to 84 * 84 * 1(Z axis) * 3(RGB) with linear interploation type(3)
            imgData2.resize(84, 84, 1, 3, 3);
#endif

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
#ifdef USE_OPENCV
                    img[imgOffset] = (float)(imgData3.at<unsigned char>(i, j)) / 255.0;
#else
                    img[imgOffset] = ((float)imgData2(j, i, 0, 0) * 0.2126 +  
                                      (float)imgData2(j, i, 0, 1) * 0.7152 + 
                                      (float)imgData2(j, i, 0, 2) * 0.0722) / 255.0;
#endif
                    imgOffset++;
                }
            }

#ifdef USE_OPENCV

#else
            imgDisplay = imgData;
            imgDisplay2 = imgData2;
#endif
           
#ifndef USE_RESTRICT_ACTION
            Action a = legal_actions[rand() % legal_actions.size()];
            // Apply the action and get the resulting reward
#else
            // 0 for noop, 1 for fire, 3 for right, 4 for left, action size = 4
            int actionIndex = rand() % 4;
            Action a;

            switch(actionIndex) {
            case 0:
                a = legal_actions[0]; 
                break;
            case 1:
                a = legal_actions[1];
                break;
            case 2:
                a = legal_actions[3];
                break;
            case 3:
            default:
                a = legal_actions[4];
                break;
            }

#endif
            float reward = ale.act(a);
            totalReward += reward;
        }
        cout << "Episode " << episode << " ended with score: " << totalReward << endl;
        ale.reset_game();
    }

#ifdef USE_OPENCV
#ifdef SHOW_OPENCV_WINDOW
    cv::destroyWindow("test image");
#endif
#endif
}
