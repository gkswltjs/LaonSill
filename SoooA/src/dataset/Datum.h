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


class Datum {
public:
	int channels;
	int height;
	int width;

	std::string data;
	int label;

	std::vector<float> float_data;
	bool encoded;

	void print() {
		std::cout << "channels: " << this->channels << std::endl;
		std::cout << "height: " << this->height << std::endl;
		std::cout << "width: " << this->width << std::endl;
		std::cout << "label: " << this->label << std::endl;
		std::cout << "encoded: " << this->encoded << std::endl;
	}

	static const std::string serializeToString(Datum* datum) {
		std::ostringstream ofs;
		boost::archive::text_oarchive oa(ofs);
		oa << (*datum);
		return ofs.str();
	}

	static void deserializeFromString(const std::string& data, Datum* datum) {
		std::istringstream ifs(data);
		boost::archive::text_iarchive ia(ifs);
		ia >> (*datum);
	}

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & channels;
		ar & height;
		ar & width;

		ar & data;
		ar & label;

		ar & float_data;
		ar & encoded;
	}
};



#endif /* DATUM_H_ */
