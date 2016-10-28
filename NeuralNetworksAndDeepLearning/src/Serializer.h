/**
 * @file Serializer.h
 * @date 2016-10-25
 * @author mhlee
 * @brief 
 * @details
 */

#ifndef SERIALIZER_H
#define SERIALIZER_H 

#include "MessageHeader.h"

class Serializer {
public:
                Serializer() {}
    virtual    ~Serializer() {}
   
    static int  serializeInt(int data, int offset, char* msg);
    static int  deserializeInt(int& data, int offset, char* msg);

    static int  serializeMsgHdr(MessageHeader msgHdr, char* msg);
    static int  deserializeMsgHdr(MessageHeader& msgHdr, char* msg);
};

#endif /* SERIALIZER_H */
