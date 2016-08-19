/*
 * Debug.h
 *
 *  Created on: 2016. 8. 17.
 *      Author: jhkim
 */

#ifndef DEBUG_H_
#define DEBUG_H_

class DataSet;
class LayersConfig;






DataSet* createMnistDataSet();
DataSet* createImageNet10CatDataSet();
DataSet* createImageNet100CatDataSet();


LayersConfig* createCNNDoubleLayersConfig();
LayersConfig* createGoogLeNetLayersConfig();
LayersConfig* createInceptionLayersConfig();


#endif /* DEBUG_H_ */
