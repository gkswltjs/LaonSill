/**
 * @file KistiGan.h
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef KISTIGAN_H
#define KISTIGAN_H 

template<typename Dtype>
class KistiGan {
public: 
    KistiGan() {}
    virtual ~KistiGan() {}

    static void run();
private:
    static void setLayerTrain(LayersConfig<Dtype>* lc, bool train);
    static LayersConfig<Dtype>* createDOfGANLayersConfig();
    static LayersConfig<Dtype>* createGD0OfGANLayersConfig();
};
#endif /* KISTIGAN_H */
