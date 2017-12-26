/**
 * @file YOLO.h
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef YOLO_H
#define YOLO_H 

template<typename Dtype>
class YOLO {
public: 
    YOLO() {}
    virtual ~YOLO() {}
    static void runPretrain();
    static void run();
private:
    static LayersConfig<Dtype>* createYoloPreLayersConfig();
    static LayersConfig<Dtype>* createYoloLayersConfig();
};

#endif /* YOLO_H */
