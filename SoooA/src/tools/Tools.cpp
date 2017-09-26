#include <string>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/variant.hpp>

#include "Tools.h"
#include "SysLog.h"
#include "Datum.h"
#include "IO.h"
#include "DataReader.h"
#include "ssd_common.h"
#include "ParamManipulator.h"

using namespace std;
namespace fs = ::boost::filesystem;


void printDenormalizeParamUsage(char* prog) {
    fprintf(stderr, "Usage: %s -i old_param_path -o new_param_path\n", prog);
    fprintf(stderr, "\t-i: old param file path\n");
	fprintf(stderr, "\t-o: new param file path\n");
    exit(EXIT_FAILURE);
}

void denormalizeTest(int argc, char** argv) {
	string old_param_path;
	string new_param_path;

	int opt;
	while ((opt = getopt(argc, argv, "i:o:")) != -1) {
		if (!optarg) {
			printDenormalizeParamUsage(argv[0]);
		}

		switch (opt) {
		case 'i':
			old_param_path = string(optarg);
			break;
		case 'o':
			new_param_path = string(optarg);
			break;
		default:
			printDenormalizeParamUsage(argv[0]);
			break;
		}
	}

	if (!old_param_path.length() || !new_param_path.length()) {
		printDenormalizeParamUsage(argv[0]);
	}

	cout << endl;
	cout << "::: Denormalize Param Configurations :::" << endl;
	cout << "old_param_path: " << old_param_path << endl;
	cout << "new_param_path: " << new_param_path << endl;
	cout << ":::::::::::::::::::::::::::::::::::::::::" << endl;
	cout << endl;

	/*
	ParamManipulator<float> pm(
			"/home/jkim/Dev/SOOOA_HOME/network/frcnn_630000.param",
			"/home/jkim/Dev/SOOOA_HOME/network/frcnn_630000_dn.param");
			*/
	ParamManipulator<float> pm(old_param_path, new_param_path);
	//pm.printParamList();

	pm.denormalizeParams({"bbox_pred_weight", "bbox_pred_bias"},
			{0.f, 0.f, 0.f, 0.f},
			{0.1f, 0.1f, 0.2f, 0.2f});

	pm.save();
}



inline std::string format_int(int n, int numberOfLeadingZeros = 0) {
	std::ostringstream s;
	s << std::setw(numberOfLeadingZeros) << std::setfill('0') << n;
	return s.str();
}


uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}


void printConvertMnistDataUsage(char* prog) {
    fprintf(stderr, "Usage: %s -i image_file -l label_file -o db_path\n", prog);
    fprintf(stderr, "\t-i: image file path\n");
	fprintf(stderr, "\t-l: label file path\n");
	fprintf(stderr, "\t-o: output db file path\n");
    exit(EXIT_FAILURE);
}


void convertMnistDataTest(int argc, char** argv) {
	string image_filename;
	string label_filename;
	string db_path;

	int opt;
	while ((opt = getopt(argc, argv, "i:l:o:")) != -1) {
		if (!optarg) {
			printConvertMnistDataUsage(argv[0]);
		}

		switch (opt) {
		case 'i':
			image_filename = string(optarg);
			break;
		case 'l':
			label_filename = string(optarg);
			break;
		case 'o':
			db_path = string(optarg);
			break;
		default:
			printConvertMnistDataUsage(argv[0]);
			break;
		}
	}

	if (!image_filename.length() ||
			!label_filename.length() ||
			!db_path.length()) {
		printConvertMnistDataUsage(argv[0]);
	}

	cout << endl;
	cout << "::: Convert Mnist Data Configurations :::" << endl;
	cout << "image_filename: " << image_filename << endl;
	cout << "label_filename: " << label_filename << endl;
	cout << "db_path: " << db_path << endl;
	cout << ":::::::::::::::::::::::::::::::::::::::::" << endl;
	cout << endl;

    //const string image_filename = "/home/jkim/Dev/git/caffe/data/mnist/train-images-idx3-ubyte";
    //const string label_filename = "/home/jkim/Dev/git/caffe/data/mnist/train-labels-idx1-ubyte";
    //const string db_path = "/home/jkim/imageset/lmdb/mnist_train_lmdb/";

    // Open files
    ifstream image_file(image_filename, ios::in | ios::binary);
    ifstream label_file(label_filename, ios::in | ios::binary);
    if (!image_file) {
        cout << "Unable to open file " << image_filename << endl;
        SASSERT0(false);
    }
    if (!label_file) {
        cout << "Unable to open file " << label_filename << endl;
        SASSERT0(false);
    }
    // Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2051) {
        cout << "Incorrect image file magic." << endl;
        SASSERT0(false);
    }
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2049) {
        cout << "Incorrect label file magic." << endl;
        SASSERT0(false);
    }
    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    SASSERT0(num_items == num_labels);
    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    SDF sdf(db_path, Mode::NEW);
    sdf.open();

    // Storing to db
    char label;
    char* pixels = new char[rows * cols];
    int count = 0;
    string value;

    Datum datum;
    datum.channels = 1;
    datum.height = rows;
    datum.width = cols;
    cout << "A total of " << num_items << " items." << endl;
    cout << "Rows: " << rows << " Cols: " << cols << endl;

    sdf.put("num_data", std::to_string(num_items));
    sdf.commit();

    //string buffer(rows * cols, ' ');
    for (int item_id = 0; item_id < num_items; ++item_id) {
        image_file.read(pixels, rows * cols);
        label_file.read(&label, 1);

        //for (int i = 0; i < rows*cols; i++) {
        //   buffer[i] = pixels[i];
        //}
        //datum.data = buffer;
        datum.data.assign(reinterpret_cast<const char*>(pixels), rows * cols);
        datum.label = label;

        string key_str = format_int(item_id, 8);
        //value = Datum::serializeToString(&datum);
        value = serializeToString(&datum);

        sdf.put(key_str, value);

        if (++count % 1000 == 0) {
            sdf.commit();
            cout << "Processed " << count << " files." << endl;
        }
    }
    // write the last batch

    if (count % 1000 != 0) {
        sdf.commit();
        cout << "Processed " << count << " files." << endl;
    }
    delete[] pixels;
    sdf.close();

}


void printConvertImageSetUsage(char* prog) {
    fprintf(stderr, "Usage: %s [-g | -s | -m | -c | -w resize_width | -h resize_height | -l dataset_path] -i image_path -o output_path\n", prog);
    fprintf(stderr, "\t-g: gray image input\n");
    fprintf(stderr, "\t-s: shuffle the image set\n");
    fprintf(stderr, "\t-m: multiple labels\n");
    fprintf(stderr, "\t-c: channel not separated\n");
    fprintf(stderr, "\t-w: resize image with specified width\n");
    fprintf(stderr, "\t-h: resize image with specified height\n");
    fprintf(stderr, "\t-i: image path\n");
    fprintf(stderr, "\t-d: dataset path\n");
    fprintf(stderr, "\t-o: output path\n");

    exit(EXIT_FAILURE);
}


void convertImageSetTest(int argc, char** argv) {
	//const string argv1 = "/home/jkim/imageset/jpegs/";
	//const string argv2 = "/home/jkim/imageset/labels/train.txt";
	//const string argv3 = "/home/jkim/imageset/lmdb/train_lmdb/";
	//const string argv1 = "/home/jkim/Backups/ilsvrc12_train/";
	//const string argv2 = "/home/jkim/Dev/git/caffe/data/ilsvrc12/train.txt";
	//const string argv3 = "/home/jkim/imageset/lmdb/ilsvrc_lmdb/";

	string argv1;
	string argv2;
	string argv3;

	bool 	FLAGS_gray = false;
	bool 	FLAGS_shuffle = false;		// default false
	bool	FLAGS_multi_label = false;
	bool	FLAGS_channel_separated = true;
	int 	FLAGS_resize_width = 0;		// default 0
	int		FLAGS_resize_height = 0;	// default 0
	bool	FLAGS_check_size = false;
	bool	FLAGS_encoded = false;
	string	FLAGS_encode_type = "";

	//g: gray
	//s: shuffle
	//w: resize_width
	//h: resize_height
	//i: image_path
	//d: dataset
	//o: output
	int opt;
	while ((opt = getopt(argc, argv, "gsmcw:h:i:d:o:")) != -1) {
		switch (opt) {
		case 'g':
			FLAGS_gray = true;
			break;
		case 's':
			FLAGS_shuffle = true;
			break;
		case 'm':
			FLAGS_multi_label = true;
			break;
		case 'c':
			FLAGS_channel_separated = false;
			break;
		case 'w':
			if (optarg) FLAGS_resize_width = atoi(optarg);
			else printConvertImageSetUsage(argv[0]);
			break;
		case 'h':
			if (optarg) FLAGS_resize_height = atoi(optarg);
			else printConvertImageSetUsage(argv[0]);
			break;
		case 'i':
			if (optarg) argv1 = string(optarg);
			else printConvertImageSetUsage(argv[0]);
			break;
		case 'd':
			if (optarg) argv2 = string(optarg);
			else printConvertImageSetUsage(argv[0]);
			break;
		case 'o':
			if (optarg) argv3 = string(optarg);
			else printConvertImageSetUsage(argv[0]);
			break;
		default:
			printConvertImageSetUsage(argv[0]);
			break;
		}
	}

	if (!argv1.length() ||
			// assume image only mode, if dataset is not provided
			// !argv2.length() ||
			!argv3.length()) {
		printConvertImageSetUsage(argv[0]);
	}

	cout << endl;
	cout << "::: Convert Image Set Configurations :::" << endl;
	cout << "gray: " << FLAGS_gray << endl;
	cout << "shuffle: " << FLAGS_shuffle << endl;
	cout << "multi_pabel: " << FLAGS_multi_label << endl;
	cout << "channel_separated: " << FLAGS_channel_separated << endl;
	cout << "resize_width: " << FLAGS_resize_width << endl;
	cout << "resize_height: " << FLAGS_resize_height << endl;
	cout << "image_path: " << argv1 << endl;
	cout << "dataset_path: " << argv2 << endl;
	cout << "output_path: " << argv3 << endl;
	cout << "::::::::::::::::::::::::::::::::::::::::" << endl;
	cout << endl;


	const bool is_color = !FLAGS_gray;
	const bool multi_label = FLAGS_multi_label;
	const bool channel_separated = FLAGS_channel_separated;
	const bool check_size = FLAGS_check_size;
	const bool encoded = FLAGS_encoded;
	const string encode_type = FLAGS_encode_type;
	const bool image_only_mode = (argv2.length() == 0) ? true : false;

	vector<pair<string, vector<int>>> lines;
	string line;
	size_t pos;
	int label;
	vector<int> labelList;


	// image and label pairs are provided
	if (!image_only_mode) {
		ifstream infile(argv2);
		while (std::getline(infile, line)) {
			labelList.clear();
			pos = line.find_last_of(' ');

			// sinlge label is provided
			if (!multi_label) {
				labelList.push_back(atoi(line.substr(pos + 1).c_str()));
				lines.push_back(std::make_pair(line.substr(0, pos), labelList));
			}
			// multiple labels are provided
			else {
				string first = line.substr(0, pos);
				string labels = line.substr(pos + 1);
				pos = labels.find_first_of(',');
				while (pos != string::npos) {
					labelList.push_back(atoi(labels.substr(0, pos).c_str()));
					labels = labels.substr(pos + 1);
					pos = labels.find_first_of(',');
				}
				labelList.push_back(atoi(labels.substr(0, pos).c_str()));
				lines.push_back(std::make_pair(first.substr(0, first.length()), labelList));
				/*
				cout << "first: " << lines.back().first << endl;
				cout << "second: ";
				for (int i = 0; i < lines.back().second.size(); i++) {
					cout << lines.back().second[i] << ", ";
				}
				cout << endl;
				*/
			}
		}
	}
	// only images provided
	else {
		SASSERT(fs::exists(argv1), "image path %s not exists ... ", argv1.c_str());
		SASSERT(fs::is_directory(argv1), "image path %s is not directory ... ", argv1.c_str());

		const string ext = ".jpg";
		fs::path image_path(argv1);
		fs::recursive_directory_iterator it(image_path);
		fs::recursive_directory_iterator endit;

		int count = 0;
		while (it != endit) {
			if (fs::is_regular_file(*it) && it->path().extension() == ext) {
				string path = it->path().filename().string();
				// path를 그대로 전달할 경우 error ...
				// substr() 호출한 결과를 전달할 경우 문제 x
				labelList.push_back(0);
				lines.push_back(std::make_pair(path.substr(0, path.length()), labelList));
				//lines.push_back(std::make_pair<string, vector<int>>(path, {0}));
			}
			it++;
		}
	}

	if (FLAGS_shuffle) {
		// randomly shuffle data
		std::random_shuffle(lines.begin(), lines.end());
	} else {

	}

	cout << "A total of " << lines.size() << " images." << endl;
	/*
	for (int i = 0; i < std::min<int>(lines.size(), 100); i++) {
		cout << "fn: " << lines[i].first << ", label: " << lines[i].second << endl;
	}
	*/


	if (encode_type.size() && !encoded) {
		cout << "encode_type specified, assuming encoded=true.";
	}

	int resize_height = std::max<int>(0, FLAGS_resize_height);
	int resize_width = std::max<int>(0, FLAGS_resize_width);

	// Create new DB
	SDF* sdf = new SDF(argv3, Mode::NEW);
	sdf->open();

	// Storing to db
	string root_folder(argv1);
	Datum datum;
	int count = 0;
	int data_size = 0;
	bool data_size_initialized = false;


	// 이 시점에서 data 수를 저장할 경우
	// 아래 status가 false인 경우 등의 상황에서 수가 정확하지 않을 가능성이 있음.
	sdf->put("num_data", std::to_string(lines.size()));
	sdf->commit();


	for (int line_id = 0; line_id < lines.size(); line_id++) {
		bool status;
		string enc = encode_type;
		if (encoded && !enc.size()) {
			// Guess the encoding type from the file name
			string fn = lines[line_id].first;
			size_t p = fn.rfind('.');
			if (p == fn.npos) {
				cout << "Failed to guess the encoding of '" << fn << "'";
			}
			enc = fn.substr(p);
			std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
		}
		status = ReadImageToDatum(root_folder + lines[line_id].first, lines[line_id].second,
				resize_height, resize_width, 0, 0, channel_separated, is_color, enc, &datum);

		if (status == false) {
			continue;
		}
		SASSERT0(!check_size);

		// sequencial
		string key_str = format_int(line_id, 8) + "_" + lines[line_id].first;

		// Put in db
		//string out = Datum::serializeToString(&datum);
		string out = serializeToString(&datum);
		sdf->put(key_str, out);

		if (++count % 1000 == 0) {
			// Commit db
			sdf->commit();
			//
			cout << "Processed " << count << " files." << endl;
		}
	}

	// write the last batch
	if (count % 1000 != 0) {
		sdf->commit();
		cout << "Processed " << count << " files." << endl;
	}

	sdf->close();
}
















void convertAnnoSetTest(int argc, char** argv) {
	bool 	FLAGS_gray = false;
	bool 	FLAGS_shuffle = false;		// default false
	bool	FLAGS_multi_label = false;
	bool	FLAGS_channel_separated = true;
	int 	FLAGS_resize_width = 0;		// default 0
	int		FLAGS_resize_height = 0;	// default 0
	bool	FLAGS_check_size = false;	// check that all the datum have the same size
	bool	FLAGS_encoded = true;		// default true, the encoded image will be save in datum
	string	FLAGS_encode_type = "jpg";	// default "", what type should we encode the image as ('png', 'jpg', ... )
	string	FLAGS_anno_type = "detection";		// default "classification"
	string	FLAGS_label_type = "xml";
	string	FLAGS_label_map_file = "/home/jkim/Dev/git/caffe_ssd/data/VOC0712/labelmap_voc.prototxt";	// default ""
	bool	FLAGS_check_label = true;			// default false, check that there is no duplicated name/label
	int		FLAGS_min_dim = 0;
	int		FLAGS_max_dim = 0;

	const string argv1 = "/home/jkim/Dev/git/caffe_ssd/data/VOCdevkit/"; // base path
	const string argv2 = "/home/jkim/Dev/git/caffe_ssd/data/VOC0712/trainval.txt"; // dataset ... (trainval.txt, ...)
	const string argv3 = "/home/jkim/Dev/SOOOA_HOME/data/sdf/voc2007_train_sdf/";		// sdf path
	//const string argv2 = "/home/jkim/Dev/git/caffe_ssd/data/VOC0712/test.txt"; // dataset ... (trainval.txt, ...)
	//const string argv3 = "/home/jkim/Dev/SOOOA_HOME/data/sdf/voc2007_test_sdf/";		// sdf path

	const bool is_color = !FLAGS_gray;
	const bool check_size = FLAGS_check_size;
	const bool encoded = FLAGS_encoded;
	const string encode_type = FLAGS_encode_type;
	const string anno_type = FLAGS_anno_type;
	AnnotationType type;
	const string label_type = FLAGS_label_type;
	const string label_map_file = FLAGS_label_map_file;
	const bool check_label = FLAGS_check_label;
	//map<string, int> name_to_label;



	ifstream infile(argv2);
	vector<pair<string, boost::variant<int, string>>> lines;
	string filename;
	int label;
	string labelname;
	SASSERT(anno_type == "detection", "only anno_type 'detection' is supported.");
	type = AnnotationType::BBOX;
	LabelMap<float> label_map(label_map_file);
	label_map.build();

	while (infile >> filename >> labelname) {
		lines.push_back(make_pair(filename, labelname));
	}

	if (FLAGS_shuffle) {
		// randomly shuffle data
		cout << "Shuffling data" << endl;
		//shuffle(lines.begin(), lines.end());
		std::random_shuffle(lines.begin(), lines.end());
	}
	cout << "A total of " << lines.size() << " images." << endl;

	if (encode_type.size() && !encoded) {
		cout << "encode_type specified, assuming encoded=true." << endl;
	}

	int min_dim = std::max<int>(0, FLAGS_min_dim);
	int max_dim = std::max<int>(0, FLAGS_max_dim);
	int resize_height = std::max<int>(0, FLAGS_resize_height);
	int resize_width = std::max<int>(0, FLAGS_resize_width);

	// Create new DB
	SDF* sdf = new SDF(argv3, Mode::NEW);
	sdf->open();

	// Storing to db
	string root_folder(argv1);
	AnnotatedDatum anno_datum;
	int count = 0;
	int data_size = 0;
	bool data_size_initialized = false;

	// 이 시점에서 data 수를 저장할 경우
	// 아래 status가 false인 경우 등의 상황에서 수가 정확하지 않을 가능성이 있음.
	sdf->put("num_data", std::to_string(lines.size()));
	sdf->commit();

	for (int line_id = 0; line_id < lines.size(); line_id++) {
		bool status = true;
		string enc = encode_type;
		if (encoded && !enc.size()) {
			// Guess the encoding type from the file name
			string fn = lines[line_id].first;
			size_t p = fn.rfind('.');
			if (p == fn.npos) {
				cout << "Failed to guess the encoding of '" << fn << "'";
			}
			enc = fn.substr(p);
			std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
		}
		filename = root_folder + lines[line_id].first;
		labelname = root_folder + boost::get<string>(lines[line_id].second);
		status = ReadRichImageToAnnotatedDatum(filename, labelname, resize_height,
				resize_width, min_dim, max_dim, is_color, enc, type, label_type,
				label_map.labelToIndMap, &anno_datum);
		anno_datum.type = AnnotationType::BBOX;
		//anno_datum.print();

		if (status == false) {
			cout << "Failed to read " << lines[line_id].first << endl;
			continue;
		}
		SASSERT0(!check_size);

		// sequencial
		string key_str = format_int(line_id, 8) + "_" + lines[line_id].first;

		// Put in db
		//string out = Datum::serializeToString(&datum);
		string out = serializeToString(&anno_datum);
		sdf->put(key_str, out);

		if (++count % 1000 == 0) {
			// Commit db
			sdf->commit();
			//
			cout << "Processed " << count << " files." << endl;
		}
	}

	// write the last batch
	if (count % 1000 != 0) {
		sdf->commit();
		cout << "Processed " << count << " files." << endl;
	}

	sdf->close();

}





void computeImageMean(int argc, char** argv) {
	DataReader<Datum> dr("/home/jkim/Dev/SOOOA_HOME/data/sdf/plantynet_train_0.25/");
	int numData = dr.getNumData();
	cout << "numData: " << numData << endl;

	size_t mean[3] = {0, 0, 0};
	size_t elemCnt[3] = {0, 0, 0};

	int i = 0;
	while (i < numData) {
		Datum* datum = dr.getNextData();

		const int channels = datum->channels;
		const int height = datum->height;
		const int width = datum->width;
		const uchar* dataPtr = (uchar*)datum->data.c_str();

		for (int c = 0; c < channels; c++) {
			int imgArea = height * width;
			for (int idx = 0; idx < imgArea; idx++) {
				mean[c] += dataPtr[c * imgArea + idx];
			}
			elemCnt[c] += imgArea;
		}

		if (++i % 1000 == 0) {
			cout << "Processed " << i << " images." << endl;
		}
	}

	if (i % 1000 != 0) {
		cout << "Processed " << i << " images." << endl;
	}


	for (i = 0; i < 3; i++) {
		double m = 0.0;
		if (elemCnt[i] > 0) {
			m = mean[i] / (double)elemCnt[i];
		}
		cout << "sum: " << mean[i];
		cout << "\tcount: " << elemCnt[i];
		cout << "\t" << i << "th mean: " << m << endl;
	}
}

