
#include <stdlib.h>
#include <stdio.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/return_value_policy.hpp>

#include "Tools.h"
#include "DataReader.h"


using namespace std;


BOOST_PYTHON_MODULE(libSoooAClient) {
	using namespace boost::python;

	// SDF Validation


	// SDF Summary
	typedef vector<int> VectorInt;
	class_<VectorInt>("VectorInt")
		.def(vector_indexing_suite<VectorInt>());

	typedef vector<float> VectorFloat;
	class_<VectorFloat>("VectorFloat")
		.def(vector_indexing_suite<VectorFloat>());

	class_<Datum>("Datum")
		.def("getImgSize",			&Datum::getImgSize)
		.def("hasLabel",			&Datum::hasLabel)
		.def("info",				&Datum::print)
		.def_readonly("channels",	&Datum::channels)
		.def_readonly("height",		&Datum::height)
		.def_readonly("width",		&Datum::width)
		.def_readonly("label",		&Datum::label)
		.def_readonly("encoded",	&Datum::encoded)
		.def_readonly("float_data",	&Datum::float_data)
		.def_readonly("data",		&Datum::data)
	;

	enum_<AnnotationType>("AnnotationType")
		.value("ANNO_NONE", ANNO_NONE)
		.value("BBOX", BBOX)
	;

	class_<NormalizedBBox>("NormalizedBBox")
		.def("info", &NormalizedBBox::print)
		.def_readonly("xmin",		&NormalizedBBox::xmin)
		.def_readonly("ymin",		&NormalizedBBox::ymin)
		.def_readonly("xmax",		&NormalizedBBox::xmax)
		.def_readonly("ymax",		&NormalizedBBox::ymax)
		.def_readonly("label",		&NormalizedBBox::label)
		.def_readonly("difficult",	&NormalizedBBox::difficult)
		.def_readonly("score",		&NormalizedBBox::score)
		.def_readonly("size",		&NormalizedBBox::size)
	;

	class_<Annotation_s>("Annotation")
		.def("info", 					&Annotation_s::print)
		.def_readonly("instance_id",	&Annotation_s::instance_id)
		.def_readonly("bbox", 			&Annotation_s::bbox)
	;

	class_<vector<Annotation_s>>("vector_annotation")
			.def(vector_indexing_suite<vector<Annotation_s>>());

	class_<AnnotationGroup>("AnnotationGroup")
		.def("info", 					&AnnotationGroup::print)
		.def_readonly("group_label",	&AnnotationGroup::group_label)
		.def_readonly("annotations",	&AnnotationGroup::annotations)
	;
	class_<vector<AnnotationGroup>>("vector_annotationgroup")
		.def(vector_indexing_suite<vector<AnnotationGroup>>());

	class_<AnnotatedDatum, bases<Datum>>("AnnotatedDatum")
		.def("info", 						&AnnotatedDatum::print)
		.def_readonly("type", 				&AnnotatedDatum::type)
		.def_readonly("annotation_groups",	&AnnotatedDatum::annotation_groups)
	;

	typedef DataReader<class Datum> DataReaderDatum;
	class_<DataReaderDatum>("DataReaderDatum", init<const string&>())
		.def("getNumData", &DataReaderDatum::getNumData)
		.def("getNextData", &DataReaderDatum::getNextData,
				return_value_policy<manage_new_object>())
		.def("peekNextData", &DataReaderDatum::peekNextData,
				return_value_policy<manage_new_object>())
	;

	typedef DataReader<class AnnotatedDatum> DataReaderAnnoDatum;
	class_<DataReaderAnnoDatum>("DataReaderAnnoDatum", init<const string&>())
		.def("getNumData", &DataReaderAnnoDatum::getNumData)
		.def("getNextData", &DataReaderAnnoDatum::getNextData,
				return_value_policy<manage_new_object>())
		.def("peekNextData", &DataReaderAnnoDatum::peekNextData,
				return_value_policy<manage_new_object>())
	;

	// SDF Building
	class_<ConvertMnistDataParam>("ConvertMnistDataParam")
		.def_readwrite("imageFilePath",	&ConvertMnistDataParam::imageFilePath)
		.def_readwrite("labelFilePath",	&ConvertMnistDataParam::labelFilePath)
		.def_readwrite("outFilePath",	&ConvertMnistDataParam::outFilePath)
	;
	class_<ConvertImageSetParam>("ConvertImageSetParam")
		.def_readwrite("gray",				&ConvertImageSetParam::gray)
		.def_readwrite("shuffle",			&ConvertImageSetParam::shuffle)
		.def_readwrite("multiLabel",		&ConvertImageSetParam::multiLabel)
		.def_readwrite("channelSeparated",	&ConvertImageSetParam::channelSeparated)
		.def_readwrite("resizeWidth",		&ConvertImageSetParam::resizeWidth)
		.def_readwrite("resizeHeight",		&ConvertImageSetParam::resizeHeight)
		.def_readwrite("checkSize",			&ConvertImageSetParam::checkSize)
		.def_readwrite("encoded",			&ConvertImageSetParam::encoded)
		.def_readwrite("encodeType",		&ConvertImageSetParam::encodeType)
		.def_readwrite("basePath",			&ConvertImageSetParam::basePath)
		.def_readwrite("datasetPath",		&ConvertImageSetParam::datasetPath)
		.def_readwrite("outPath",			&ConvertImageSetParam::outPath)
	;
	class_<ConvertAnnoSetParam, bases<ConvertImageSetParam>>("ConvertAnnoSetParam")
		.def_readwrite("annoType",			&ConvertAnnoSetParam::annoType)
		.def_readwrite("labelType",			&ConvertAnnoSetParam::labelType)
		.def_readwrite("labelMapFile",		&ConvertAnnoSetParam::labelMapFile)
		.def_readwrite("checkLabel",		&ConvertAnnoSetParam::checkLabel)
		.def_readwrite("minDim",			&ConvertAnnoSetParam::minDim)
		.def_readwrite("maxDim",			&ConvertAnnoSetParam::maxDim)
	;
	def("ConvertMnistData", convertMnistData);
	def("ConvertImageSet", convertImageSet);
	def("ConvertAnnoSet", convertAnnoSet);

	// ETC Tools
	def("Denormalize", denormalize);
	def("ComputeImageMean", computeImageMean);

	enum_<Mode>("Mode")
		.value("READ", READ)
		.value("NEW", NEW)
	;
	class_<SDF>("SDF", init<const string&, const Mode>())
		.def("open",			&SDF::open)
		.def("close",			&SDF::close)
		.def("put",				&SDF::put)
		.def("commit",			&SDF::commit)
		//.def("getNextValue",	&SDF::getNextValue, return_value_policy<manage_new_object>())
	;


}



















