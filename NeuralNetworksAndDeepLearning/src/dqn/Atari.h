/**
 * @file Atari.h
 * @date 2016-12-08
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef ATARI_H
#define ATARI_H 
class Atari {
public: 
    Atari() {}
    virtual ~Atari() {}
    static void run(char* romFilePath);
};
#endif /* ATARI_H */
