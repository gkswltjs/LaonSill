/*
 * Datum.h
 *
 *  Created on: Jun 29, 2017
 *      Author: jkim
 */

#ifndef DATUM_H_
#define DATUM_H_

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>











class DatumHeader {
	int numData;
	std::vector<std::string> classes;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & numData;
		ar & classes;
	}
};


class Datum {
public:
	int channels;
	int height;
	int width;
	int label;
	bool encoded;
	std::vector<float> float_data;
	std::string data;


	Datum()
	: channels(0), height(0), width(0), label(0), encoded(false), data("") {}


	void print() {
		std::cout << "channels: " << this->channels << std::endl;
		std::cout << "height: " << this->height << std::endl;
		std::cout << "width: " << this->width << std::endl;
		std::cout << "label: " << this->label << std::endl;
		std::cout << "encoded: " << this->encoded << std::endl;
		std::cout << "float_data: ";
		for (int i = 0; i < this->float_data.size(); i++) {
			std::cout << this->float_data[i] << ",";
		}
		std::cout << std::endl;
	}



protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & channels;
		ar & height;
		ar & width;
		ar & label;
		ar & encoded;
		ar & float_data;
		ar & data;
	}
};



// The normalized bounding box [0, 1] w.r.t. the input image size
class NormalizedBBox {
public:
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	int label;
	bool difficult;
	float score;
	float size;

	NormalizedBBox() {
		this->xmin = 0.f;
		this->ymin = 0.f;
		this->xmax = 0.f;
		this->ymax = 0.f;
		this->label = 0;
		this->difficult = false;
		this->score = 0.f;
		this->size = 0.f;
	}

	void print() {
		std::cout << "\txmin: " 		<< this->xmin		<< std::endl;
		std::cout << "\tymin: " 		<< this->ymin		<< std::endl;
		std::cout << "\txmax: " 		<< this->xmax		<< std::endl;
		std::cout << "\tymax: " 		<< this->ymax		<< std::endl;
		std::cout << "\tlabel: " 		<< this->label		<< std::endl;
		std::cout << "\tdifficult: "	<< this->difficult	<< std::endl;
		std::cout << "\tscore: " 		<< this->score		<< std::endl;
		std::cout << "\tsize: "			<< this->size		<< std::endl;
	}

protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & xmin;
		ar & ymin;
		ar & xmax;
		ar & ymax;
		ar & label;
		ar & difficult;
		ar & score;
		ar & size;
	}
};


class Annotation_s {
public:
	int instance_id;
	NormalizedBBox bbox;

	void print() {
		std::cout << "instance_id: " << this->instance_id << std::endl;
		std::cout << "bbox: " << std::endl;
		this->bbox.print();
	}

protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & instance_id;
		ar & bbox;
	}
};

class AnnotationGroup {
public:
	int group_label;
	std::vector<Annotation_s> annotations;

	Annotation_s* add_annotation() {
		Annotation_s annotation;
		this->annotations.push_back(annotation);
		return &this->annotations.back();
	}

	void print() {
		std::cout << "group_label: " << this->group_label << std::endl;
		for (int i = 0; i < this->annotations.size(); i++) {
			this->annotations[i].print();
		}
	}

protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & group_label;
		ar & annotations;
	}
};



class AnnotatedDatum : public Datum {
public:
	std::vector<AnnotationGroup> annotation_groups;

	AnnotationGroup* add_annotation_group() {
		AnnotationGroup annotation_group;
		this->annotation_groups.push_back(annotation_group);
		return &this->annotation_groups.back();
	}

	void print() {
		Datum::print();
		for (int i = 0; i < annotation_groups.size(); i++) {
			annotation_groups[i].print();
		}
	}

protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		Datum::serialize(ar, version);
		ar & annotation_groups;
	}
};
























template <typename T>
const std::string serializeToString(T* datum) {
	std::ostringstream ofs;
	boost::archive::text_oarchive oa(ofs);
	oa << (*datum);
	return ofs.str();
}

template const std::string serializeToString<Datum>(Datum* datum);
//template const std::string serializeToString<AnnotatedDatum>(AnnotatedDatum* datum);


template <typename T>
void deserializeFromString(const std::string& data, T* datum) {
	std::istringstream ifs(data);
	boost::archive::text_iarchive ia(ifs);
	ia >> (*datum);
}

template void deserializeFromString<Datum>(const std::string& data, Datum* datum);
//template void deserializeFromString<AnnotatedDatum>(const std::string& data, AnnotatedDatum* datum);








#endif /* DATUM_H_ */
