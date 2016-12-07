/**
 * @file MsgSerializer.cpp
 * @date 2016-10-25
 * @author moonhoen lee
 * @brief 
 * @details
 */

#include <string.h>
#include <arpa/inet.h>

#include "MsgSerializer.h"

using namespace std;

int MsgSerializer::serializeInt(int data, int offset, char* msg) {
    int temp;
    temp = htonl(data);
    memcpy((void*)(msg + offset), (void*)&temp, sizeof(int));
   
    return offset + sizeof(int);
}

int MsgSerializer::deserializeInt(int& data, int offset, char* msg) {
    int temp;
    memcpy((void*)&temp, (void*)(msg + offset), sizeof(int));
    data = ntohl(temp);
   
    return offset + sizeof(int);
}

int MsgSerializer::serializeMsgHdr(MessageHeader msgHdr, char* msg) {
    // @See: MessageHeader.h
    int offset = 0;
    offset = MsgSerializer::serializeInt(msgHdr.getMsgType(), offset, msg);
    offset = MsgSerializer::serializeInt(msgHdr.getMsgLen(), offset, msg);
    return offset;
}

int MsgSerializer::deserializeMsgHdr(MessageHeader& msgHdr, char* msg) {
    // @See: MessageHeader.h
    int msgType;
    int msgLen;
    int offset = 0;

    offset = MsgSerializer::deserializeInt(msgType, offset, msg); 
    offset = MsgSerializer::deserializeInt(msgLen, offset, msg); 

    msgHdr.setMsgType((MessageHeader::MsgType)msgType);
    msgHdr.setMsgLen(msgLen);

    return offset;
}
