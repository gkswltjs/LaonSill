/**
 * @file Serializer.cpp
 * @date 2016-10-25
 * @author mhlee
 * @brief 
 * @details
 */

#include <string.h>
#include <arpa/inet.h>

#include "Serializer.h"

using namespace std;

int Serializer::serializeInt(int data, int offset, char* msg) {
    int temp;
    temp = htonl(data);
    memcpy((void*)(msg + offset), (void*)&temp, sizeof(int));
   
    return offset + sizeof(int);
}

int Serializer::deserializeInt(int& data, int offset, char* msg) {
    int temp;
    memcpy((void*)&temp, (void*)(msg + offset), sizeof(int));
    data = ntohl(temp);
   
    return offset + sizeof(int);
}

int Serializer::serializeMsgHdr(MessageHeader msgHdr, char* msg) {
    // @See: MessageHeader.h
    int offset = 0;
    offset = Serializer::serializeInt(msgHdr.getMsgType(), offset, msg);
    offset = Serializer::serializeInt(msgHdr.getMsgLen(), offset, msg);
    return offset;
}

int Serializer::deserializeMsgHdr(MessageHeader& msgHdr, char* msg) {
    // @See: MessageHeader.h
    int msgType;
    int msgLen;
    int offset = 0;

    offset = Serializer::deserializeInt(msgType, offset, msg); 
    offset = Serializer::deserializeInt(msgLen, offset, msg); 

    msgHdr.setMsgType((MessageHeader::MsgType)msgType);
    msgHdr.setMsgLen(msgLen);

    return offset;
}
