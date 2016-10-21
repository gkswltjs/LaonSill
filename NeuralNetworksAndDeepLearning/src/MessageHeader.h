/**
 * @file MessageHeader.h
 * @date 2016-10-20
 * @author mhlee
 * @brief 
 * @details
 */

#ifndef MESSAGEHEADER_H
#define MESSAGEHEADER_H 

// Serialize시 packet 구성
// |---------+--------+------------------------------------------|
// | MsgType | MsgLen | MsgBody                                  |
// | int(4)  | int(4) | variable size => MsgLen - 8 byte         |
// |---------+--------+------------------------------------------|

class MessageHeader {
public:
    enum MsgType : int {
        Welcome = 0,
        WelcomReply,

        HaltMachine,
        GoodBye,
    };

    MessageHeader()                     {}
    virtual ~MessageHeader()            {}

    int getMsgLen()                     { return this->msgLen; }
    MsgType getMsgType()                { return this->msgType; }

    void setMsgType(MsgType msgType)    { this->msgType = msgType; }
    void setMsgLen(int msgLen)          { this->msgLen = msgLen; }

private:
    MsgType                             msgType;
    int                                 msgLen;
};

#endif /* MESSAGEHEADER_H */
