/**
 * @file KISTIKeyword.h
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef KISTIKEYWORD_H
#define KISTIKEYWORD_H 

#include <vector>

typedef struct top10Sort_s {
    float value;
    int index;

    bool operator < (const struct top10Sort_s &x) const {
        return value < x.value;
    }
} top10Sort;

template<typename Dtype>
class KISTIKeyword {
public: 
    KISTIKeyword() {}
    virtual ~KISTIKeyword() {}

    static void run();
private:
#if 0
    static LayersConfig<Dtype>* createKistiVGG19NetLayersConfig();
    static int getTop10GuessSuccessCount(const float* data, const float* label, int batchCount,
        int depth, bool train, int epoch, const float* image, int imageBaseIndex,
        std::vector<KistiData> etriData);
#endif
};

#endif /* KISTIKEYWORD_H */
