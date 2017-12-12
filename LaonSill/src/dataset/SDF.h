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
#include "SysLog.h"

enum Mode { READ, NEW };



inline std::string format_str(const std::string& str, long length = 0) {
	const int strLength = str.length();
	if (strLength >= length) {
		return str;
	}

	std::string leading(length - strLength, '.');
	return leading + str;
}

inline std::string unformat_str(const std::string& str) {
	const int strLength = str.length();
	int startPos = 0;

	const char* ptr = str.c_str();
	for (int i = 0; i < strLength; i++) {
		if (ptr[i] != '.') {
			break;
		}
		startPos++;
	}

	// '=' for zero length string (empty string)
	SASSERT0(startPos <= strLength);
	return str.substr(startPos, strLength);
}




inline std::string format_int(int n, long numberOfLeadingZeros = 0) {
	std::ostringstream s;
	s << std::setw(numberOfLeadingZeros) << std::setfill('0') << n;
	return s.str();
}

inline long unformat_int(const std::string& format_str) {
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














class LabelItem {
public:
	LabelItem() : label(0) {}

	void print() {
		std::cout << "LabelItem: " 		<< this->name 			<< std::endl;
		std::cout << "\tlabel: " 		<< this->label 			<< std::endl;
		std::cout << "\tdisplay_name: " << this->displayName	<< std::endl;
		if (this->color.size() > 0) {
			std::cout << "\tcolor: [" << this->color[0] << ", " << this->color[1] << ", " <<
					this->color[2] << "]" << std::endl;
		}
	}

	bool operator==(const LabelItem& other) {
		return (this->name == other.name &&
				this->label == other.label &&
				this->displayName == other.displayName &&
				this->color == other.color);
	}

public:
	std::string name;
	int label;
	std::string displayName;
	std::vector<int> color;


protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {

		// read (input)
		if (dynamic_cast<boost::archive::text_iarchive*>(&ar)) {
			std::string name;
			std::string label;
			std::string displayName;
			std::vector<std::string> color;

			ar & name;
			ar & label;
			ar & displayName;
			ar & color;

			this->name = unformat_str(name);
			this->label = unformat_int(label);
			this->displayName = unformat_str(displayName);
			this->color.clear();
			for (int i = 0; i < color.size(); i++) {
				this->color.push_back(unformat_int(color[i]));
			}
		}
		// write (output)
		else if (dynamic_cast<boost::archive::text_oarchive*>(&ar)) {
			std::string name = format_str(this->name, 32);
			std::string label = format_int(this->label, 10);
			std::string displayName = format_str(this->displayName, 32);
			std::vector<std::string> color;
			for (int i = 0; i < this->color.size(); i++) {
				color.push_back(format_int(this->color[i], 3));
			}

			ar & name;
			ar & label;
			ar & displayName;
			ar & color;
		} else {

		}
	}
};




class SDFHeader {
public:
	SDFHeader() : numSets(0) {}

	int numSets;
	std::vector<std::string> names;
	std::vector<int> setSizes;
	std::vector<long> setStartPos;
	std::vector<LabelItem> labelItemList;

	void init(int numSets) {
		this->numSets = numSets;
		this->names.resize(numSets, "");
		this->setSizes.resize(numSets, 0);
		this->setStartPos.resize(numSets, 0);
	}

	void print() {
		std::cout << "numSets: " << this->numSets << std::endl;
		std::cout << "names: " << std::endl;
		for (int i = 0; i < this->names.size(); i++) {
			std::cout << "\t" << this->names[i] << std::endl;
		}
		std::cout << "setSizes: " << std::endl;
		for (int i = 0; i < this->setSizes.size(); i++) {
			std::cout << "\t" << this->setSizes[i] << std::endl;
		}
		std::cout << "setStartPos: " << std::endl;
		for (int i = 0; i < this->setStartPos.size(); i++) {
			std::cout << "\t" << this->setStartPos[i] << std::endl;
		}
		std::cout << "labelItemList: " << std::endl;
		for (int i = 0; i < std::min(10, this->labelItemList.size()); i++) {
			this->labelItemList[i].print();
		}
		if (this->labelItemList.size() > 10) {
			std::cout << "printed only first 10 label items ... " << std::endl
		}
	}


protected:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {

		// read (input)
		if (dynamic_cast<boost::archive::text_iarchive*>(&ar)) {
			std::string numSets;
			std::vector<std::string> names;
			std::vector<std::string> setSizes;
			std::vector<std::string> setStartPos;
			ar & numSets;
			ar & names;
			ar & setSizes;
			ar & setStartPos;
			ar & this->labelItemList;

			this->numSets = (int)unformat_int(numSets);
			this->names.clear();
			for (int i = 0; i < names.size(); i++) {
				this->names.push_back(unformat_str(names[i]));
			}
			this->setSizes.clear();
			for (int i = 0; i < setSizes.size(); i++) {
				this->setSizes.push_back((int)unformat_int(setSizes[i]));
			}
			this->setStartPos.clear();
			for (int i = 0; i < setStartPos.size(); i++) {
				this->setStartPos.push_back(unformat_int(setStartPos[i]));
			}
		}
		// write (output)
		else if (dynamic_cast<boost::archive::text_oarchive*>(&ar)) {
			std::string numSets = format_int(this->numSets, 10);
			std::vector<std::string> names;
			for (int i = 0; i < this->names.size(); i++) {
				SASSERT0(this->names[i].length() <= 32);
				names.push_back(format_str(this->names[i], 32));
			}
			std::vector<std::string> setSizes;
			for (int i = 0; i < this->setSizes.size(); i++) {
				setSizes.push_back(format_int(this->setSizes[i], 10));
			}
			std::vector<std::string> setStartPos;
			for (int i = 0; i < this->setStartPos.size(); i++) {
				setStartPos.push_back(format_int(this->setStartPos[i], 20));
			}

			ar & numSets;
			ar & names;
			ar & setSizes;
			ar & setStartPos;
			ar & this->labelItemList;
		} else {

		}
	}

};

//BOOST_CLASS_VERSION(SDFHeader, 0);


/***************
 * SDF에 boost를 통해 serialize되는 신규 class의 객체가 있는 경우,
 * header 이전에 반드시 dummy로 한 번 serialize해주어야 한다.
 * -> SDFHeader, LabelItem이 적용되어 있는 상태
 */
class SDF {
public:
	SDF(const std::string& source, const Mode mode);
	SDF(const SDF& sdf);
	virtual ~SDF();

	void open();
	void close();

	void initHeader(SDFHeader& header);
	void updateHeader(SDFHeader& header);
	SDFHeader getHeader();
	long getCurrentPos();
	void setCurrentPos(long currentPos);
	void selectDataSet(const std::string& dataSet);
	void selectDataSet(const int dataSetIdx);
	const std::string& curDataSet();

	// SDF가 특정 이름의 dataSet을 갖고 있는지 확인
	int findDataSet(const std::string& dataSet);

	void put(const std::string& value);
	void commit();

	const std::string getNextValue();


private:
	void update_dataset_idx_map();
	void sdf_open();
	void sdf_close();

public:
	std::string source;
	Mode mode;

private:
	SDFHeader header;
	int headerStartPos;
	int bodyStartPos;

	//std::vector<std::string> keys;
	std::vector<std::string> values;

	size_t dbSize;

	std::ifstream ifs;
	std::ofstream ofs;
	boost::archive::text_iarchive* ia;
	boost::archive::text_oarchive* oa;

	std::vector<long> currentPos;
	std::map<std::string, int> dataSetIdxMap;
	int curDataSetIdx;


	const std::string DATA_NAME = "data.sdf";
	const std::string LOCK_NAME = "lock.sdf";
	const std::string SDF_STRING = "SOOOA_DATA_FORMAT";

};

#endif /* SDF_H_ */
