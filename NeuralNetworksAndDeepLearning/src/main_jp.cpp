

#include <stddef.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <map>
#include <cfloat>
#include <cassert>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "3rd_party/tinyxml2/tinyxml2.h"


using namespace std;
using namespace tinyxml2;
using namespace cv;


#if 0
const uint32_t GT = 0;
const uint32_t GE = 1;
const uint32_t EQ = 2;
const uint32_t LE = 3;
const uint32_t LT = 4;

const float TRAIN_FG_THRESH = 0.5f;
const float TRAIN_BG_THRESH_HI = 0.5f;
const float TRAIN_BG_THRESH_LO = 0.0f;
const float TRAIN_BBOX_THRESH = 0.5f;

template <typename Dtype>
static void printArray(const string& name, const vector<Dtype>& array,
		const bool printName=true) {
	if (printName) {
		cout << name << ": " << endl;
	}

	const uint32_t arraySize = array.size();
	cout << "[ ";
	for (uint32_t i = 0; i < arraySize; i++) {
		if (i < arraySize-1) {
			cout << array[i] << ", ";
		} else {
			cout << array[i];
		}
	}
	cout << "]" << endl;
}

template <typename Dtype>
static void print2dArray(const string& name, const vector<vector<Dtype>>& array,
		const bool printName=true) {
	if (printName) {
		cout << name << ": " << endl;
	}

	const uint32_t arraySize = array.size();
	cout << "[ " << endl;
	for (uint32_t i = 0; i < arraySize; i++) {
		printArray(name, array[i], false);
	}
	cout << "]" << endl;
}

template <typename Dtype>
static void printPrimitive(const string& name, const Dtype data,
		const bool printName=true) {
	cout << name << ": " << data << endl;
}


template <typename Dtype>
static void np_maxByAxis(vector<vector<Dtype>>& array, vector<Dtype>& result) {
	const uint32_t numArrayElem = array.size();
	assert(numArrayElem >= 1);
	const uint32_t numAxisElem = array[0].size();
	assert(numAxisElem >= 1);

	result.clear();
	result.resize(numArrayElem);
	Dtype max;
	for (uint32_t i = 0; i < numArrayElem; i++) {
		for (uint32_t j = 0; j < numAxisElem; j++) {
			if (j == 0) max = array[i][0];
			else if (array[i][j] > max) {
				max = array[i][j];
			}
		}
		result[i] = max;
	}
}

template <typename Dtype>
static void np_argmax(const vector<vector<Dtype>>& array, vector<uint32_t>& result) {
	const uint32_t numArrayElem = array.size();
	assert(numArrayElem >= 1);
	const uint32_t numAxisElem = array[0].size();
	assert(numAxisElem >= 1);

	result.clear();
	result.resize(numArrayElem);
	Dtype max;
	uint32_t maxIndex;
	for (uint32_t i = 0; i < numArrayElem; i++) {
		for (uint32_t j = 0; j < numAxisElem; j++) {
			if (j == 0) {
				max = array[i][0];
				maxIndex = 0;
			}
			else if (array[i][j] > max) {
				max = array[i][j];
				maxIndex = j;
			}
		}
		result[i] = maxIndex;
	}
}


static void np_where_s(vector<float>& array, uint32_t comp, float criteria,
		vector<uint32_t>& result) {
	const uint32_t numArrayElem = array.size();
	assert(numArrayElem > 0);

	switch (comp) {
	case GT:
	for (uint32_t i = 0; i < numArrayElem; i++) if (array[i] > criteria) result.push_back(i);
	break;
	case GE:
	for (uint32_t i = 0; i < numArrayElem; i++) if (array[i] >= criteria) result.push_back(i);
	break;
	case EQ:
	for (uint32_t i = 0; i < numArrayElem; i++) if (array[i] == criteria) result.push_back(i);
	break;
	case LE:
	for (uint32_t i = 0; i < numArrayElem; i++) if (array[i] <= criteria) result.push_back(i);
	break;
	case LT:
	for (uint32_t i = 0; i < numArrayElem; i++) if (array[i] < criteria) result.push_back(i);
	break;
	default:
		cout << "invalid comp: " << comp << endl;
		exit(1);
	}
}

template <typename Dtype>
static void np_where_s(const vector<vector<Dtype>>& array, const Dtype criteria,
		const uint32_t loc, vector<uint32_t>& result) {
	const uint32_t numArrayElem = array.size();
	assert(numArrayElem > 0);
	assert(loc < array[0].size());

	result.clear();
	for (uint32_t i = 0; i < numArrayElem; i++) {
		if (array[i][loc] == criteria) {
			result.push_back(i);
		}
	}
}

static void np_where(vector<float>& array, const vector<uint32_t>& comp,
		const vector<float>& criteria, vector<uint32_t>& result) {

	if (comp.size() < 1 ||
			criteria.size() < 1 ||
			comp.size() != criteria.size() ||
			array.size() < 1) {

		cout << "invalid array dimension ... " << endl;
		exit(1);
	}

	result.clear();

	const uint32_t numComps = comp.size();
	const uint32_t numArrayElem = array.size();
	bool cond;
	for (uint32_t i = 0; i < numArrayElem; i++) {
		cond = true;

		for (uint32_t j = 0; j < numComps; j++) {
			switch (comp[j]) {
			case GT: if (array[i] <= criteria[j])	cond = false; break;
			case GE: if (array[i] <	criteria[j])	cond = false; break;
			case EQ: if (array[i] != criteria[j])	cond = false; break;
			case LE: if (array[i] > criteria[j])	cond = false; break;
			case LT: if (array[i] >= criteria[j])	cond = false; break;
			default:
				cout << "invalid comp: " << comp[j] << endl;
				exit(1);
			}
			if (!cond) break;
		}
		if (cond) result.push_back(i);
	}
}




template <typename Dtype>
static void np_tile(const vector<Dtype>& array, const uint32_t repeat,
		vector<vector<Dtype>>& result) {

	result.clear();
	for (uint32_t i = 0; i < repeat; i++) {
		result.push_back(array);
	}
}


template <typename Dtype>
static void py_arrayElemsWithArrayInds(const vector<Dtype>& array,
		const vector<uint32_t>& inds, vector<Dtype>& result) {

	const uint32_t arraySize = array.size();
	const uint32_t indsSize = inds.size();

	//result.clear();
	result.resize(indsSize);
	for (uint32_t i = 0; i < indsSize; i++) {
		assert(inds[i] < arraySize);
		result[i] = array[inds[i]];
	}
}






struct Size {
	uint32_t width;
	uint32_t height;
	uint32_t depth;

	void print() {
		cout << "Size:" << endl <<
				"\twidth: " << width << endl <<
				"\theight: " << height << endl <<
				"\tdepth: " << depth << endl;
	}
};

struct Object {
	string name;
	uint32_t label;
	uint32_t difficult;
	uint32_t xmin;
	uint32_t ymin;
	uint32_t xmax;
	uint32_t ymax;

	void print() {
		cout << "Object: " << endl <<
				"\tname: " << name << endl <<
				"\tlabel: " << label << endl <<
				"\tdifficult: " << difficult << endl <<
				"\txmin: " << xmin << endl <<
				"\tymin: " << ymin << endl <<
				"\txmax: " << xmax << endl <<
				"\tymax: " << ymax << endl;
	}
};

struct Annotation {
	string filename;
	Size size;
	vector<Object> objects;

	void print() {
		cout << "Annotation:" << endl <<
				"\tfilename: " << filename << endl;
		size.print();
		for (uint32_t i = 0; i < objects.size(); i++) {
			objects[i].print();
		}
	}
};


struct RoIDB {
	RoIDB() {};
	/*
	RoIDB(const RoIDB& roidb) {
		this->boxes = roidb.boxes;
		this->gt_classes = roidb.gt_classes;
		this->gt_overlaps = roidb.gt_overlaps;
		this->flipped = roidb.flipped;
	}
	*/

	vector<vector<uint32_t>> boxes;			// Annotation의 원본 bounding box (x1, y1, x2, y2)
	vector<uint32_t> gt_classes;			// 각 bounding box의 class, [9, 9, 9]의 형식
	vector<vector<float>> gt_overlaps;		// 각 bounding box의 gt_box와의 IoU Value
	vector<uint32_t> max_classes;
	vector<float> max_overlaps;
	vector<vector<float>> bbox_targets;
	bool flipped;							// Original, Flipped Image 여부
	string image;
	uint32_t width;
	uint32_t height;

	void print() {
		cout << ":::RoIDB:::" << endl;
		print2dArray("boxes", boxes);
		printArray("gt_classes", gt_classes);
		print2dArray("gt_overlaps", gt_overlaps);
		printArray("max_classes", max_classes);
		printArray("max_overlaps", max_overlaps);
		print2dArray("bbox_targets", bbox_targets);
		printPrimitive("flipped", flipped);
		printPrimitive("image", image);
		printPrimitive("width", width);
		printPrimitive("height", height);
	}
};


struct IMDB {
	IMDB(const string& name) {
		this->name = name;
		this->numClasses = 0;
	}
	virtual ~IMDB() {}

	void appendFlippedImages() {
		const uint32_t numImages = this->imageIndex.size();

		uint32_t oldx1, oldx2;
		for (uint32_t i = 0; i < numImages; i++) {
			RoIDB roidb(this->roidb[i]);
			const uint32_t numObjects = this->roidb[i].boxes.size();
			for (uint32_t j = 0; j < numObjects; j++) {
				oldx1 = roidb.boxes[j][0];
				oldx2 = roidb.boxes[j][2];
				roidb.boxes[j][0] = roidb.width - oldx2 - 1;
				roidb.boxes[j][2] = roidb.width - oldx1 - 1;
			}
			roidb.flipped = true;
			this->roidb.push_back(roidb);
			this->imageIndex.push_back(this->imageIndex[i]);
		}
	}

	virtual void getWidths(vector<uint32_t>& widths) {
		cout << "IMDB::getWidths() is not supported ... " << endl;
		exit(1);
	}

	virtual void loadGtRoidb() {
		cout << "IMDB::loadGtRoidb() is not supported ... " << endl;
		exit(1);
	}


	string name;
	uint32_t numClasses;
	vector<uint32_t> clasess;
	vector<string> imageIndex;
	vector<RoIDB> roidb;
};


struct PascalVOC : public IMDB {

	PascalVOC(const string& imageSet, const string& year,
			const string& devkitPath) : IMDB("voc_" + year + "_" + imageSet) {

		this->year = year;
		this->imageSet = imageSet;
		this->devkitPath = devkitPath;
		this->dataPath = devkitPath + "/VOC" + year;
		this->classes = {
				"__background__",
				"aeroplane", "bicycle", "bird", "boat",
				"bottle", "bus", "car", "cat", "chair",
				"cow", "diningtable", "dog", "horse",
				"motorbike", "person", "pottedplant",
				"sheep", "sofa", "train", "tvmonitor"
		};
		buildClassToInd();
		this->imageExt = ".jpg";
		loadImageSetIndex();
	}

	void buildClassToInd() {
		for (uint32_t i = 0; i < numClasses; i++) {
			printf("Label [%02d]: %s\n", i, classes[i].c_str());
			classToInd[classes[i]] = i;
		}
	}

	void loadImageSetIndex() {
		string imageSetFile = this->dataPath + "/ImageSets/Main/" + this->imageSet + ".txt";

		ifstream ifs(imageSetFile.c_str(), ios::in);
		if (!ifs.is_open()) {
			cout << "no such file: " << imageSetFile << endl;
			exit(1);
		}

		stringstream strStream;
		strStream << ifs.rdbuf();

		ifs.close();

		char line[256];
		uint32_t count = 0;
		while (!strStream.eof()) {
			strStream.getline(line, 256);
			if (strlen(line) < 1) {
				continue;
			}
			imageIndex.push_back(string(line));
		}
		const uint32_t numTrainval = imageIndex.size();
		cout << "numTrainval: " << numTrainval << endl;
		for (uint32_t i = 0; i < numTrainval; i++) {
			cout << imageIndex[i] << endl;
		}
	}

	void loadPascalAnnotation(const string& index, RoIDB& roidb) {
		const string filename = this->dataPath + "/Annotations/" + index + ".xml";
		Annotation annotation;
		readAnnotation(filename, annotation);

		roidb.image = this->dataPath + "/JPEGImages/" + index + this->imageExt;
		roidb.width = annotation.size.width;
		roidb.height = annotation.size.height;

		const uint32_t numObjs = annotation.objects.size();

		roidb.boxes.resize(numObjs);
		roidb.gt_classes.resize(numObjs);
		roidb.gt_overlaps.resize(numObjs);

		for (uint32_t i = 0; i < numObjs; i++) {
			// boxes
			roidb.boxes[i].resize(4);
			roidb.boxes[i][0] = annotation.objects[i].xmin-1;	// xmin
			roidb.boxes[i][1] = annotation.objects[i].ymin-1;	// ymin
			roidb.boxes[i][2] = annotation.objects[i].xmax-1;	// xmax
			roidb.boxes[i][3] = annotation.objects[i].ymax-1;	// ymax

			// gt_classes
			roidb.gt_classes[i] = annotation.objects[i].label;

			// overlaps
			roidb.gt_overlaps[i].resize(this->numClasses);
			roidb.gt_overlaps[i][roidb.gt_classes[i]] = 1.0;
		}
		roidb.flipped = false;

		// max_classes
		roidb.max_classes = roidb.gt_classes;

		// max_overlaps
		roidb.print();
		np_maxByAxis(roidb.gt_overlaps, roidb.max_overlaps);
		roidb.print();

		// XXX: gt_overlaps의 경우 sparse matrix로 변환될 필요가 있음.
	}


	void readAnnotation(const string& filename, Annotation& annotation) {
		XMLDocument annotationDocument;
		XMLNode* annotationNode;

		annotationDocument.LoadFile(filename.c_str());
		annotationNode = annotationDocument.FirstChild();

		// filename
		XMLElement* filenameElement = annotationNode->FirstChildElement("filename");
		annotation.filename = filenameElement->GetText();

		// size
		XMLElement* sizeElement = annotationNode->FirstChildElement("size");
		sizeElement->FirstChildElement("width")->QueryIntText((int*)&annotation.size.width);
		sizeElement->FirstChildElement("height")->QueryIntText((int*)&annotation.size.height);
		sizeElement->FirstChildElement("depth")->QueryIntText((int*)&annotation.size.depth);

		// object
		for (XMLElement* objectElement = annotationNode->FirstChildElement("object");
				objectElement != 0;
				objectElement = objectElement->NextSiblingElement("object")) {
			Object object;
			object.name = objectElement->FirstChildElement("name")->GetText();
			object.label = convertClassToInd(object.name);
			objectElement->FirstChildElement("difficult")->QueryIntText((int*)&object.difficult);

			XMLElement* bndboxElement = objectElement->FirstChildElement("bndbox");
			bndboxElement->FirstChildElement("xmin")->QueryIntText((int*)&object.xmin);
			bndboxElement->FirstChildElement("ymin")->QueryIntText((int*)&object.ymin);
			bndboxElement->FirstChildElement("xmax")->QueryIntText((int*)&object.xmax);
			bndboxElement->FirstChildElement("ymax")->QueryIntText((int*)&object.ymax);

			if (!object.difficult) {
				annotation.objects.push_back(object);
			}
		}
		annotation.print();
	}

	uint32_t convertClassToInd(const string& cls) {
		map<string, uint32_t>::iterator itr = classToInd.find(cls);
		if(itr == classToInd.end()) {
			cout << "invalid class: " << cls;
			exit(1);
		}
		return itr->second;
	}

	void getWidths(vector<uint32_t>& widths) {
		widths = this->widths;
	}

	void loadGtRoidb() {
		const uint32_t numImageIndex = this->imageIndex.size();
		for (uint32_t i = 0; i < numImageIndex; i++) {
			RoIDB roidb;
			loadPascalAnnotation(imageIndex[i], roidb);
			this->roidb.push_back(roidb);
		}
		// XXX: gtRoidb를 dump to file
	}

	//IMDB imdb;
	string year;
	string imageSet;
	string devkitPath;
	string dataPath;
	map<string, uint32_t> classToInd;
	string imageExt;
	//vector<string> imageIndex;
	vector<uint32_t> widths;

	const uint32_t numClasses = 21;
	vector<string> classes;
};


struct BboxTransformUtil {

public:
	static void bboxTransform(const vector<vector<uint32_t>>& ex_rois,
			const vector<vector<uint32_t>>& gt_rois, vector<vector<float>>& result) {
		assert(ex_rois.size() == gt_rois.size());

		float ex_width, ex_height, ex_ctr_x, ex_ctr_y;
		float gt_width, gt_height, gt_ctr_x, gt_ctr_y;

		const uint32_t numRois = ex_rois.size();
		for (uint32_t i = 0; i < numRois; i++) {
			 ex_width = ex_rois[i][2] - ex_rois[i][0] + 1.0f;
			 ex_height = ex_rois[i][3] - ex_rois[i][1] + 1.0f;
			 ex_ctr_x = ex_rois[i][0] + 0.5f * ex_width;
			 ex_ctr_y = ex_rois[i][1] + 0.5f * ex_height;

			 gt_width = gt_rois[i][2] - gt_rois[i][0] + 1.0f;
			 gt_height = gt_rois[i][3] - gt_rois[i][1] + 1.0f;
			 gt_ctr_x = gt_rois[i][0] + 0.5f * gt_width;
			 gt_ctr_y = gt_rois[i][1] + 0.5f * gt_height;

			 // result[i][0] for label
			 result[i][1] = (gt_ctr_x - ex_ctr_x) / ex_width;
			 result[i][2] = (gt_ctr_y - ex_ctr_y) / ex_height;
			 result[i][3] = std::log(gt_width / ex_width);
			 result[i][4] = std::log(gt_height / ex_height);
		}
	}
};


struct RoIDBUtil {
public:
	static void addBboxRegressionTargets(vector<RoIDB>& roidb) {
		// Add information needed to train bounding-box regressors.
		assert(roidb.size() > 0);

		const uint32_t numImages = roidb.size();
		// Infer numfer of classes from the number of columns in gt_overlaps
		const uint32_t numClasses = roidb[0].gt_overlaps[0].size();

		for (uint32_t i = 0; i < numImages; i++) {
			computeTargets(roidb[i]);
		}

		vector<vector<float>> means;
		np_tile({0.0f, 0.0f, 0.0f, 0.0f}, numClasses, means);
		print2dArray("bbox target means", means);

		vector<vector<float>> stds;
		np_tile({0.1f, 0.1f, 0.2f, 0.2f}, numClasses, stds);
		print2dArray("bbox target stdeves", stds);

		// Normalize targets
		cout << "Normalizing targets" << endl;
		for (uint32_t i = 0; i < numImages; i++) {
			vector<vector<float>>& targets = roidb[i].bbox_targets;
			vector<uint32_t> clsInds;
			for (uint32_t j = 1; j < numClasses; j++) {
				np_where_s(targets, static_cast<float>(j), 0, clsInds);

				for (uint32_t k = 0; k < clsInds.size(); k++) {
					targets[k][1] = (targets[k][1] - means[j][0]) / stds[j][0];
					targets[k][2] = (targets[k][2] - means[j][1]) / stds[j][1];
					targets[k][3] = (targets[k][3] - means[j][2]) / stds[j][2];
					targets[k][4] = (targets[k][4] - means[j][3]) / stds[j][3];
				}
			}
			print2dArray("bbox_targets", targets);
		}
	}

	static void computeTargets(RoIDB& roidb) {
		roidb.print();

		// Compute bounding-box regression targets for an image.
		// Indices of ground-truth ROIs
		vector<uint32_t> gt_inds;
		// XXX: 1.0f float compare check
		np_where_s(roidb.max_overlaps, EQ, 1.0f, gt_inds);
		if (gt_inds.size() < 1) {
			// Bail if the image has no ground-truth ROIs
		}
		// Indices of examples for which we try to make predictions
		vector<uint32_t> ex_inds;
		np_where_s(roidb.max_overlaps, GE, TRAIN_BBOX_THRESH, ex_inds);

		// Get IoU overlap between each ex ROI and gt ROI
		vector<vector<float>> ex_gt_overlaps;
		bboxOverlaps(roidb.boxes, gt_inds, ex_inds, ex_gt_overlaps);
		print2dArray("ex_gt_overlaps", ex_gt_overlaps);

		// Find which gt ROI each ex ROI has max overlap with:
		// this will be the ex ROI's gt target
		vector<uint32_t> gt_assignment;
		np_argmax(ex_gt_overlaps, gt_assignment);
		vector<uint32_t> gt_rois_inds;
		vector<vector<uint32_t>> gt_rois;
		py_arrayElemsWithArrayInds(gt_inds, gt_assignment, gt_rois_inds);
		py_arrayElemsWithArrayInds(roidb.boxes, gt_rois_inds, gt_rois);
		print2dArray("gt_rois", gt_rois);
		vector<vector<uint32_t>> ex_rois;
		py_arrayElemsWithArrayInds(roidb.boxes, ex_inds, ex_rois);
		print2dArray("ex_rois", ex_rois);

		const uint32_t numRois = roidb.boxes.size();
		const uint32_t numEx = ex_inds.size();
		vector<vector<float>>& targets = roidb.bbox_targets;
		targets.resize(numRois);
		for (uint32_t i = 0; i < numRois; i++) {
			targets[i].resize(5);
			// XXX: init to zero ... ?
		}
		print2dArray("targets", targets);

		for (uint32_t i = 0; i < numEx; i++) {
			targets[i][0] = roidb.max_classes[i];
		}
		print2dArray("targets", targets);
		BboxTransformUtil::bboxTransform(ex_rois, gt_rois, targets);
		print2dArray("targets", targets);

		roidb.print();
	}

	static void bboxOverlaps(const vector<vector<uint32_t>>& rois,
			const vector<uint32_t>& gt_inds, const vector<uint32_t>& ex_inds,
			vector<vector<float>>& result) {

		const uint32_t numEx = ex_inds.size();
		const uint32_t numGt = gt_inds.size();

		result.resize(numEx);
		for (uint32_t i = 0; i < numEx; i++) {
			result[i].resize(numGt);
			for (uint32_t j = 0; j < numGt; j++) {
				result[i][j] = iou(rois[ex_inds[i]], rois[gt_inds[j]]);
			}
		}
	}

	static float iou(const vector<uint32_t>& box1, const vector<uint32_t>& box2) {
		float iou = 0.0f;
		uint32_t left, right, top, bottom;
		left = std::max(box1[0], box2[0]);
		right = std::min(box1[2], box2[2]);
		top = std::max(box1[1], box2[1]);
		bottom = std::min(box1[3], box2[3]);

		if(left < right &&
				top < bottom) {
			float i = (right-left)*(bottom-top);
			float u = (box1[2]-box1[0])*(box1[3]-box1[1]) +
					(box2[2]-box2[0])*(box2[3]-box2[1]) - i;
			iou = i/u;
		}
		return iou;
	}
};







IMDB* get_imdb(const string& imdb_name) {
	IMDB* imdb = new PascalVOC("trainval_sample", "2007",
			"/home/jkim/Dev/git/py-faster-rcnn/data/VOCdevkit2007");
	imdb->loadGtRoidb();

	return imdb;
}

void get_training_roidb(IMDB* imdb) {
	cout << "Appending horizontally-flipped training examples ... " << endl;
	imdb->appendFlippedImages();
	cout << "done" << endl;

	cout << "Preparing training data ... " << endl;
	//rdl_roidb.prepare_roidb(imdb)
	cout << "done" << endl;
}

IMDB* get_roidb(const string& imdb_name) {
	IMDB* imdb = get_imdb(imdb_name);
	cout << "Loaded dataset " << imdb->name << " for training ... " << endl;
	get_training_roidb(imdb);

	return imdb;
}

IMDB* combined_roidb(const string& imdb_name) {
	IMDB* imdb = get_roidb(imdb_name);
	return imdb;
}



bool is_valid_roidb(RoIDB& roidb) {
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

void filter_roidb(vector<RoIDB>& roidb) {
	// Remove roidb entries that have no usable RoIs.

	const uint32_t numRoidb = roidb.size();
	for (int i = numRoidb-1; i >= 0; i--) {
		if (!is_valid_roidb(roidb[i])) {
			roidb.erase(roidb.begin()+i);
		}
	}

	const uint32_t numAfter = roidb.size();
	cout << "Filtered " << numRoidb - numAfter << " roidb entries: " <<
			numRoidb << " -> " << numAfter << endl;
}





void train_net(vector<RoIDB>& roidb) {
	// Train a Fast R-CNN network.
	filter_roidb(roidb);

	cout << "Computing bounding-box regression targets ... " << endl;
	RoIDBUtil::addBboxRegressionTargets(roidb);
	cout << "done" << endl;

	// create network, load saved params and run ...

	// set roidb to roi_data_layer ...


}


int main_(void) {

	IMDB* imdb = combined_roidb("voc_2007_trainval");
	cout << imdb->roidb.size() << " roidb entries ... " << endl;

	train_net(imdb->roidb);


	for (uint32_t i = 0; i < imdb->roidb.size(); i++) {
		imdb->roidb[i].print();
	}


	return 0;
}

#endif



static uint32_t np_round_(float a) {
	uint32_t lower = static_cast<uint32_t>(floorf(a) + 0.1f);
	if (lower % 2 == 0) return lower;

	uint32_t upper = static_cast<uint32_t>(ceilf(a) + 0.1f);
	return upper;
}


int main_(void) {
	cv::Mat image = cv::imread("/home/jkim/Downloads/sampleR32G64B128.png");

	namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);
	imshow("MyWindow", image);

	cv::resize(image, image, cv::Size(), 1.5, 1.5, CV_INTER_LINEAR);

	namedWindow("resize", CV_WINDOW_AUTOSIZE);
	imshow("resize", image);

	waitKey(0);
	destroyAllWindows();

	return 0;
}










