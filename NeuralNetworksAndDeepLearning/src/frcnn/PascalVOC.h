/*
 * PascalVOC.h
 *
 *  Created on: Nov 14, 2016
 *      Author: jkim
 */

#ifndef PASCALVOC_H_
#define PASCALVOC_H_


#include <map>
#include <vector>
#include <string>
#include <fstream>

#include "frcnn_common.h"
#include "tinyxml2/tinyxml2.h"
#include "IMDB.h"


class PascalVOC : public IMDB {
public:
	PascalVOC(const std::string& imageSet, const std::string& year,
			const std::string& devkitPath) : IMDB("voc_" + year + "_" + imageSet) {

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
		std::string imageSetFile = this->dataPath + "/ImageSets/Main/" + this->imageSet + ".txt";

		std::ifstream ifs(imageSetFile.c_str(), std::ios::in);
		if (!ifs.is_open()) {
			std::cout << "no such file: " << imageSetFile << std::endl;
			exit(1);
		}

		std::stringstream strStream;
		strStream << ifs.rdbuf();

		ifs.close();

		char line[256];
		uint32_t count = 0;
		while (!strStream.eof()) {
			strStream.getline(line, 256);
			if (strlen(line) < 1) {
				continue;
			}
			imageIndex.push_back(std::string(line));
		}
		const uint32_t numTrainval = imageIndex.size();
		std::cout << "numTrainval: " << numTrainval << std::endl;
		for (uint32_t i = 0; i < numTrainval; i++) {
			std::cout << imageIndex[i] << std::endl;
		}
	}

	void loadPascalAnnotation(const std::string& index, RoIDB& roidb) {
		const std::string filename = this->dataPath + "/Annotations/" + index + ".xml";
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


	void readAnnotation(const std::string& filename, Annotation& annotation) {
		tinyxml2::XMLDocument annotationDocument;
		tinyxml2::XMLNode* annotationNode;

		annotationDocument.LoadFile(filename.c_str());
		annotationNode = annotationDocument.FirstChild();

		// filename
		tinyxml2::XMLElement* filenameElement = annotationNode->FirstChildElement("filename");
		annotation.filename = filenameElement->GetText();

		// size
		tinyxml2::XMLElement* sizeElement = annotationNode->FirstChildElement("size");
		sizeElement->FirstChildElement("width")->QueryIntText((int*)&annotation.size.width);
		sizeElement->FirstChildElement("height")->QueryIntText((int*)&annotation.size.height);
		sizeElement->FirstChildElement("depth")->QueryIntText((int*)&annotation.size.depth);

		// object
		for (tinyxml2::XMLElement* objectElement = annotationNode->FirstChildElement("object");
				objectElement != 0;
				objectElement = objectElement->NextSiblingElement("object")) {
			Object object;
			object.name = objectElement->FirstChildElement("name")->GetText();
			object.label = convertClassToInd(object.name);
			objectElement->FirstChildElement("difficult")->QueryIntText((int*)&object.difficult);

			tinyxml2::XMLElement* bndboxElement = objectElement->FirstChildElement("bndbox");
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

	uint32_t convertClassToInd(const std::string& cls) {
		std::map<std::string, uint32_t>::iterator itr = classToInd.find(cls);
		if(itr == classToInd.end()) {
			std::cout << "invalid class: " << cls << std::endl;
			exit(1);
		}
		return itr->second;
	}

	void getWidths(std::vector<uint32_t>& widths) {
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
	std::string year;
	std::string imageSet;
	std::string devkitPath;
	std::string dataPath;
	std::map<std::string, uint32_t> classToInd;
	std::string imageExt;
	//std::vector<std::string> imageIndex;
	std::vector<uint32_t> widths;

	const uint32_t numClasses = 21;
	std::vector<std::string> classes;
};



#endif /* PASCALVOC_H_ */
