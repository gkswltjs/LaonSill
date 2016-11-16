/*
 * RoIInputLayer.cpp
 *
 *  Created on: Nov 11, 2016
 *      Author: jkim
 */

#include "RoIInputLayer.h"
#include "ImagePackDataSet.h"
#include "PascalVOC.h"
#include "RoIDBUtil.h"

using namespace std;

template <typename Dtype>
RoIInputLayer<Dtype>::RoIInputLayer() {
	// TODO Auto-generated constructor stub
}

template <typename Dtype>
RoIInputLayer<Dtype>::RoIInputLayer(Builder* builder) {

	initialize();
}

template <typename Dtype>
RoIInputLayer<Dtype>::~RoIInputLayer() {
	// TODO Auto-generated destructor stub
	delete imdb;
}

template <typename Dtype>
void RoIInputLayer<Dtype>::initialize() {
	imdb = combinedRoidb("voc_2007_trainval");
	cout << imdb->roidb.size() << " roidb entries ... " << endl;

	// Train a Fast R-CNN network.
	filterRoidb(imdb->roidb);

	cout << "Computing bounding-box regression targets ... " << endl;

	RoIDBUtil::addBboxRegressionTargets(imdb->roidb, bboxMeans, bboxStds);
	cout << "done" << endl;
}


template <typename Dtype>
void RoIInputLayer<Dtype>::feedforward() {

}

template <typename Dtype>
void RoIInputLayer<Dtype>::shape() {

}





template <typename Dtype>
IMDB* RoIInputLayer<Dtype>::getImdb(const string& imdb_name) {
	IMDB* imdb = new PascalVOC("trainval_sample", "2007",
			"/home/jkim/Dev/git/py-faster-rcnn/data/VOCdevkit2007");
	imdb->loadGtRoidb();

	return imdb;
}

template <typename Dtype>
void RoIInputLayer<Dtype>::getTrainingRoidb(IMDB* imdb) {
	cout << "Appending horizontally-flipped training examples ... " << endl;
	imdb->appendFlippedImages();
	cout << "done" << endl;

	cout << "Preparing training data ... " << endl;
	//rdl_roidb.prepare_roidb(imdb)
	cout << "done" << endl;
}

template <typename Dtype>
IMDB* RoIInputLayer<Dtype>::getRoidb(const string& imdb_name) {
	IMDB* imdb = getImdb(imdb_name);
	cout << "Loaded dataset " << imdb->name << " for training ... " << endl;
	getTrainingRoidb(imdb);

	return imdb;
}

template <typename Dtype>
IMDB* RoIInputLayer<Dtype>::combinedRoidb(const string& imdb_name) {
	IMDB* imdb = getRoidb(imdb_name);
	return imdb;
}


template <typename Dtype>
bool RoIInputLayer<Dtype>::isValidRoidb(RoIDB& roidb) {
	// Valid images have
	// 	(1) At least one foreground RoI OR
	// 	(2) At least one background RoI

	roidb.max_overlaps;
	vector<uint32_t> fg_inds, bg_inds;
	// find boxes with sufficient overlap
	roidb.print();
	np_where_s(roidb.max_overlaps, GE, TRAIN_FG_THRESH, fg_inds);
	// select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
	np_where(roidb.max_overlaps, {LT, LE}, {TRAIN_BG_THRESH_HI, TRAIN_BG_THRESH_LO}, bg_inds);

	// image is only valid if such boxes exist
	return (fg_inds.size() > 0 || bg_inds.size() > 0);
}

template <typename Dtype>
void RoIInputLayer<Dtype>::filterRoidb(vector<RoIDB>& roidb) {
	// Remove roidb entries that have no usable RoIs.

	const uint32_t numRoidb = roidb.size();
	for (int i = numRoidb-1; i >= 0; i--) {
		if (!isValidRoidb(roidb[i])) {
			roidb.erase(roidb.begin()+i);
		}
	}

	const uint32_t numAfter = roidb.size();
	cout << "Filtered " << numRoidb - numAfter << " roidb entries: " <<
			numRoidb << " -> " << numAfter << endl;
}







template class RoIInputLayer<float>;
