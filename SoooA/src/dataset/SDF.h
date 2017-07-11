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

class SDF {
public:
	SDF(const std::string& source, const Mode mode);
	virtual ~SDF();

	void open();
	void close();

	void put(const std::string& key, const std::string& value);
	void commit();

	const std::string getNextValue();


private:
	void sdf_open();
	void sdf_close();

private:
	std::string source;
	Mode mode;


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
