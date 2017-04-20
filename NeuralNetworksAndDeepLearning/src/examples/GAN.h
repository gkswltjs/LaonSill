/**
 * @file GAN.h
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef GAN_H
#define GAN_H 

template<typename Dtype>
class GAN {
public: 
    GAN() {}
    virtual ~GAN() {}

    static void run();
private:
    static void setLayerTrain(LayersConfig<Dtype>* lc, bool train);
    static LayersConfig<Dtype>* createDOfGANLayersConfig();
    static LayersConfig<Dtype>* createGD0OfGANLayersConfig();
};
#endif /* GAN_H */
