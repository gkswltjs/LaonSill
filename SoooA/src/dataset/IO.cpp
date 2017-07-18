
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <iostream>


#include "IO.h"
#include "SysLog.h"

using namespace std;
using namespace boost::property_tree;


static bool matchExt(const std::string& fn, std::string en) {
	size_t p = fn.rfind('.') + 1;
	std::string ext = p != fn.npos ? fn.substr(p) : fn;
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
	std::transform(en.begin(), en.end(), en.begin(), ::tolower);

	if (ext == en) {
		return true;
	}
	if (en == "jpg" && ext == "jpeg") {
		return true;
	}
	return false;
}


bool ReadImageToDatum(const string& filename, const vector<int>& label, const int height,
		const int width, const int min_dim, const int max_dim, const bool channel_separated,
		const bool is_color, const string& encoding, Datum* datum) {
	SASSERT0(label.size() > 0);
	cv::Mat cv_img = ReadImageToCVMat(filename, height, width, min_dim, max_dim, is_color);

	if (cv_img.data) {
		if (encoding.size()) {
			if ((cv_img.channels() == 3) == is_color && !height && !width &&
					!min_dim && !max_dim && matchExt(filename, encoding)) {
				datum->channels = cv_img.channels();
				datum->height = cv_img.rows;
				datum->width = cv_img.cols;
				bool result = ReadFileToDatum(filename, label[0], datum);
				if (label.size() > 1) {
					SASSERT0(datum->float_data.size() == 0);
					for (int i = 1; i < label.size(); i++) {
						datum->float_data.push_back((float)label[i]);
					}
				}
				return result;
			}
			EncodeCVMatToDatum(cv_img, encoding, datum);
			datum->label = label[0];
			// multiple labels case
			if (label.size() > 1) {
				SASSERT0(datum->float_data.size() == 0);
				for (int i = 1; i < label.size(); i++) {
					datum->float_data.push_back((float)label[i]);
				}
			}
			return true;
		}

		CVMatToDatum(cv_img, channel_separated, datum);
		datum->label = label[0];

		// multiple labels case
		if (label.size() > 1) {
			SASSERT0(datum->float_data.size() == 0);
			for (int i = 1; i < label.size(); i++) {
				datum->float_data.push_back((float)label[i]);
			}
		}
		return true;
	} else {
		return false;
	}
}


bool ReadRichImageToAnnotatedDatum(const string& filename, const string& labelname,
		const int height, const int width, const int min_dim, const int max_dim,
		const bool is_color, const string& encoding, const string& labeltype,
		const map<string, int>& name_to_label, AnnotatedDatum* anno_datum) {
	// Read image to datum.
	bool status = ReadImageToDatum(filename, {-1}, height, width, min_dim, max_dim, true,
			is_color, encoding, anno_datum);

	if (status == false) {
		return status;
	}
	anno_datum->annotation_groups.clear();
	if (!boost::filesystem::exists(labelname)) {
		return true;
	}

	int ori_height;
	int ori_width;
	GetImageSize(filename, &ori_height, &ori_width);
	SASSERT(labeltype == "xml", "only label_type 'xml' is supported.");

	return ReadXMLToAnnotatedDatum(labelname, ori_height, ori_width, name_to_label,
			anno_datum);

}





cv::Mat ReadImageToCVMat(const string& filename, const int height, const int width,
		const int min_dim, const int max_dim, const bool is_color) {
	cv::Mat cv_img;
	int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
	if (!cv_img_origin.data) {
		cout << "Could not open or find file " << filename << endl;
		return cv_img_origin;
	}
	SASSERT0(min_dim == 0 && max_dim == 0);
	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	} else {
		cv_img = cv_img_origin;
	}

	//for (int i = 0; i < 30; i++) {
	//	printf("%d,", cv_img.data[i]);
	//}
	//cout << endl;

	return cv_img;
}

void CVMatToDatum(const cv::Mat& cv_img, const bool channel_separated, Datum* datum) {
	SASSERT0(cv_img.depth() == CV_8U);
	datum->channels = cv_img.channels();
	datum->height = cv_img.rows;
	datum->width = cv_img.cols;
	//datum->clear_data ...
	datum->float_data.clear();
	datum->encoded = false;

	int datum_channels = datum->channels;
	int datum_height = datum->height;
	int datum_width = datum->width;
	int datum_size = datum_channels * datum_height * datum_width;
	string buffer(datum_size, ' ');

	// all B / all G / all R 구조로 channel을 나누어서 저장
	if (channel_separated) {
		for (int h = 0; h < datum_height; h++) {
			const uchar* ptr = cv_img.ptr<uchar>(h);
			int img_index = 0;
			for (int w = 0; w < datum_width; w++) {
				for (int c = 0; c < datum_channels; c++) {
					int datum_index = (c * datum_height + h) * datum_width + w;
					buffer[datum_index] = static_cast<char>(ptr[img_index++]);
				}
			}
		}
	}
	// test로 channel분리하지 않고 opencv가 제공하는 포맷 그대로 저장
	else {
		for (int h = 0; h < datum_height; h++) {
			const uchar* ptr = cv_img.ptr<uchar>(h);
			int img_index = 0;
			for (int w = 0; w < datum_width; w++) {
				for (int c = 0; c < datum_channels; c++) {
					int datum_index = h * datum_width * datum_channels + img_index;
					buffer[datum_index] = static_cast<char>(ptr[img_index++]);
					//if (h == 0 && w < 10) {
						//printf("%d,", (uchar)buffer[datum_index]);
					//}
				}
			}
		}
	}
	datum->data = buffer;
}

cv::Mat DecodeDatumToCVMat(const Datum* datum, bool is_color) {
	SASSERT0(datum->channels == 1 || datum->channels == 3);

	int cv_type;
	if (datum->channels == 3) {
		cv_type = CV_8UC3;
	} else if (datum->channels == 1) {
		cv_type = CV_8U;
	}

	cv::Mat cv_img(datum->height, datum->width, cv_type, (uchar*)datum->data.c_str());

	if (!cv_img.data) {
		cout << "Could not decode datum." << endl;
	}
	return cv_img;
}




bool ReadXMLToAnnotatedDatum(const string& labelname, const int img_height,
		const int img_width, const map<string, int>& name_to_label,
		AnnotatedDatum* anno_datum) {
	ptree pt;
	read_xml(labelname, pt);

	// Parse annotation.
	int width = 0;
	int height = 0;
	try {
		height = pt.get<int>("annotation.size.height");
		width = pt.get<int>("annotation.size.width");
	} catch (const ptree_error &e) {
		cout << "When parsing " << labelname << ": " << e.what() << endl;
		height = img_height;
		width = img_width;
	}
	if (height != img_height) {
		cout << labelname << " inconsistent image height." << endl;
	}
	if (width != img_width) {
		cout << labelname << " inconsistent image width." << endl;
	}
	SASSERT(width != 0 && height != 0, "%s no valid image width/height.", labelname.c_str());

	int instance_id = 0;
	BOOST_FOREACH(ptree::value_type& v1, pt.get_child("annotation")) {
		ptree pt1 = v1.second;
		if (v1.first == "object") {
			Annotation_s* anno = NULL;
			bool difficult = false;
			ptree object = v1.second;
			BOOST_FOREACH(ptree::value_type& v2, object.get_child("")) {
				ptree pt2 = v2.second;
				if (v2.first == "name") {
					string name = pt2.data();
					if (name_to_label.find(name) == name_to_label.end()) {
						cout << "Unknown name: " << name << endl;
					}
					int label = name_to_label.find(name)->second;
					bool found_group = false;
					for (int g = 0; g < anno_datum->annotation_groups.size(); g++) {
						AnnotationGroup* anno_group = &anno_datum->annotation_groups[g];
						if (label == anno_group->group_label) {
							if (anno_group->annotations.size() == 0) {
								instance_id = 0;
							} else {
								instance_id = anno_group->annotations[
								    anno_group->annotations.size() - 1].instance_id + 1;
							}
							anno = anno_group->add_annotation();
							found_group = true;
						}
					}
					if (!found_group) {
						// If there is no such annotation_group, create a new one.
						AnnotationGroup* anno_group = anno_datum->add_annotation_group();
						anno_group->group_label = label;
						anno = anno_group->add_annotation();
						instance_id = 0;
					}
					anno->instance_id = instance_id++;
				} else if (v2.first == "difficult") {
					difficult = pt2.data() == "1";
				} else if (v2.first == "bndbox") {
					int xmin = pt2.get("xmin", 0);
					int ymin = pt2.get("ymin", 0);
					int xmax = pt2.get("xmax", 0);
					int ymax = pt2.get("ymax", 0);
					SASSERT0(anno != NULL);
					if (xmin > width || xmin < 0 || xmax > width || xmax < 0)
						cout << labelname << " bounding box exceeds image boundary." << endl;
					if (ymin > height || ymin < 0 || ymax > height || ymax < 0)
						cout << labelname << " bounding box exceeds image boundary." << endl;
					if (xmin > xmax || ymin > ymax)
						cout << labelname << " bounding box irregular." << endl;
					// Store the normalized bounding box.
					NormalizedBBox* bbox = &anno->bbox;
					bbox->xmin = static_cast<float>(xmin) / width;
					bbox->ymin = static_cast<float>(ymin) / height;
					bbox->xmax = static_cast<float>(xmax) / width;
					bbox->ymax = static_cast<float>(ymax) / height;
					bbox->difficult = difficult;
				}
			}
		}
	}
	return true;
}




void GetImageSize(const string& filename, int* height, int* width) {
	cv::Mat cv_img = cv::imread(filename);
	if (!cv_img.data) {
		cout << "Could not open or find file " << filename << endl;
		return;
	}
	*height = cv_img.rows;
	*width = cv_img.cols;
}


bool ReadFileToDatum(const std::string& filename, const int label, Datum* datum) {
	std::streampos size;

	fstream file(filename.c_str(), ios::in | ios::binary | ios::ate);
	if (file.is_open()) {
		size = file.tellg();
		std::string buffer(size, ' ');
		file.seekg(0, ios::beg);
		file.read(&buffer[0], size);
		file.close();
		datum->data = buffer;
		datum->label = label;
		datum->encoded = true;
		return true;
	} else {
		return false;
	}
}

void EncodeCVMatToDatum(const cv::Mat& cv_img, const std::string& encoding, Datum* datum) {
	std::vector<uchar> buf;
	cv::imencode("." + encoding, cv_img, buf);
	datum->data = string(reinterpret_cast<char*>(&buf[0]), buf.size());
	datum->channels = cv_img.channels();
	datum->height = cv_img.rows;
	datum->width = cv_img.cols;
	datum->encoded = true;
}













