/*
 * SDF.h
 *
 *  Created on: Jun 28, 2017
 *      Author: jkim
 */

#ifndef SDF_H_
#define SDF_H_

#include <fstream>
#include <string>
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "Datum.h"

enum Mode { READ, NEW };



inline std::string format_int(int n, long numberOfLeadingZeros = 0) {
	std::ostringstream s;
	s << std::setw(numberOfLeadingZeros) << std::setfill('0') << n;
	return s.str();
}

inline long format_str_to_int(const std::string& format_str) {
	const char* ptr = format_str.c_str();
	long result = 0;
	for (int i = 0; i < format_str.length(); i++) {
		result += (ptr[format_str.length() - i - 1] - '0') * std::pow(10, i);
	}
	return result;
}







template <typename T>
void printArray(std::vector<T>& array) {
	for (int i = 0; i < array.size(); i++) {
		std::cout << array[i] << ",";
	}
	std::cout << std::endl;
}


class SDFHeader {
public:
	int numSets;
	std::vector<int> setSizes;
	std::vector<long> setStartPos;


protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {

		std::cout << "serialize()" << std::endl;

		if (dynamic_cast<boost::archive::text_iarchive*>(&ar)) {
			std::string numSets;
			std::vector<std::string> setSizes;
			std::vector<std::string> setStartPos;
			ar & numSets;
			ar & setSizes;
			ar & setStartPos;

			this->numSets = (int)format_str_to_int(numSets);
			for (int i = 0; i < setSizes.size(); i++) {
				this->setSizes.push_back((int)format_str_to_int(setSizes[i]));
			}
			for (int i = 0; i < setStartPos.size(); i++) {
				this->setStartPos.push_back(format_str_to_int(setStartPos[i]));
			}
		} else if (dynamic_cast<boost::archive::text_oarchive*>(&ar)) {
			std::string numSets = format_int(this->numSets, 10);
			std::vector<std::string> setSizes;
			for (int i = 0; i < this->setSizes.size(); i++) {
				setSizes.push_back(format_int(this->setSizes[i], 10));
			}
			std::vector<std::string> setStartPos;
			for (int i = 0; i < this->setStartPos.size(); i++) {
				setStartPos.push_back(format_int(this->setStartPos[i], 10));
			}

			std::cout << "numSets: " << numSets << std::endl;
			std::cout << "setSizes: ";
			printArray(setSizes);
			std::cout << "setStartPos: ";
			printArray(setStartPos);

			ar & numSets;
			ar & setSizes;
			ar & setStartPos;
		} else {

		}
	}
};


class SDF {
public:
	SDF(const std::string& source, const Mode mode);
	SDF(const SDF& sdf);
	virtual ~SDF();

	void open();
	void close();

	void put(const std::string& key, const std::string& value);
	void commit();

	const std::string getNextValue();


private:
	void sdf_open();
	void sdf_close();

public:
	std::string source;
	Mode mode;

private:
	std::vector<std::string> keys;
	std::vector<std::string> values;

	size_t dbSize;

	std::ifstream ifs;
	std::ofstream ofs;
	boost::archive::text_iarchive* ia;
	boost::archive::text_oarchive* oa;


	const std::string dataName = "data.sdf";
	const std::string lockName = "lock.sdf";

};

#endif /* SDF_H_ */
