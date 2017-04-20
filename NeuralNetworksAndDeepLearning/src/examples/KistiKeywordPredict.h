/**
 * @file KistiKeywordPredict.h
 * @date 2017-04-20
 * @author moonhoen lee
 * @brief 
 * @details
 */

#ifndef KISTIKEYWORDPREDICT_H
#define KISTIKEYWORDPREDICT_H 

#include <vector>

typedef struct top10Sort_s {
    float value;
    int index;

    bool operator < (const struct top10Sort_s &x) const {
        return value < x.value;
    }
} top10Sort;

template<typename Dtype>
class KistiKeywordPredict {
public: 
    KistiKeywordPredict() {}
    virtual ~KistiKeywordPredict() {}

    static void run();
private:
    static LayersConfig<Dtype>* createKistiVGG19NetLayersConfig();
    static int getTop10GuessSuccessCount(const float* data, const float* label, int batchCount,
        int depth, bool train, int epoch, const float* image, int imageBaseIndex,
        std::vector<KistiData> etriData);
};

#endif /* KISTIKEYWORDPREDICT_H */
