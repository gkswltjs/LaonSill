#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>




#include "LayerTestInterface.h"
#include "LayerTest.h"
#include "LearnableLayerTest.h"

#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "ReluLayer.h"
#include "PoolingLayer.h"
#include "SoftmaxWithLossLayer.h"

#include "NetworkTestInterface.h"
#include "NetworkTest.h"

#include "TestUtil.h"
#include "Debug.h"

#include "PlanParser.h"
#include "WorkContext.h"
#include "PhysicalPlan.h"
#include "LearnableLayer.h"
#include "PropMgmt.h"
#include "InitParam.h"
#include "Perf.h"
#include "ColdLog.h"
#include "Broker.h"
#include "ResourceManager.h"
#include "PlanOptimizer.h"
#include "LayerFunc.h"
#include "gpu_nms.hpp"
#include "cnpy.h"
#include "ParamManipulator.h"
#include "Datum.h"
#include "SDF.h"
#include "IO.h"
#include "DataReader.h"



using namespace std;
namespace fs = ::boost::filesystem;






void plainTest(int argc, char** argv);
void denormalizeTest(int argc, char** argv);
void convertMnistDataTest(int argc, char** argv);
void convertImageSetTest(int argc, char** argv);
void dataReaderTest(int argc, char** argv);

void layerTest(int argc, char** argv);
void networkTest(int argc, char** argv);


#if 0
int main(int argc, char** argv) {
	cout << "begin test ... " << endl;
	cout.precision(10);
	cout.setf(ios::fixed);

	plainTest(argc, argv);
	//layerTest(argc, argv);
	//networkTest(argc, argv);

	cout << "end test ... " << endl;
	return 0;
}
#endif

void plainTest(int argc, char** argv) {
	//denormalizeTest(argc, argv);
	//convertMnistDataTest(argc, argv);
	convertImageSetTest(argc, argv);
	//dataReaderTest(argc, argv);
}



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
        assert(false);
    }
    if (!label_file) {
        cout << "Unable to open file " << label_filename << endl;
        assert(false);
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
        assert(false);
    }
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2049) {
        cout << "Incorrect label file magic." << endl;
        assert(false);
    }
    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    assert(num_items == num_labels);
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
        value = Datum::serializeToString(&datum);

        sdf.put(key_str, value);

        if (++count % 1000 == 0) {
            sdf.commit();
        }
    }
    // write the last batch

    if (count % 1000 != 0) {
        sdf.commit();
    }
    cout << "Processed " << count << " files." << endl;
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
		assert(!check_size);

		// sequencial
		string key_str = format_int(line_id, 8) + "_" + lines[line_id].first;

		// Put in db
		string out = Datum::serializeToString(&datum);
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









































void dataReaderTest(int argc, char** argv) {
	DataReader<Datum> dr("/home/jkim/imageset/lmdb/flatten_sdf/");
	int numData = dr.getNumData();
	cout << "numData: " << numData << endl;

	for (int i = 0; i < std::min(numData, 100); i++) {
		Datum* datum = dr.getNextData();
		cout << i << " label: " << datum->label;
		if (datum->float_data.size() > 0) {
			for (int j = 0; j < datum->float_data.size(); j++) {
				cout << "," << (int)datum->float_data[j];
			}
		}
		cout << endl;
		//cv::Mat cv_img = DecodeDatumToCVMat(datum, true);
		//cv::imshow("result", cv_img);
		//cv::waitKey(0);
	}
}

void layerTest() {
	const int gpuid = 0;
	vector<LayerTestInterface<float>*> layerTestList;

#if 0
	ConvLayer<float>::Builder* convBuilder = new typename ConvLayer<float>::Builder();
	convBuilder->id(10)
		->name("fc6")
		->filterDim(3, 3, 512, 1024, 6, 1)
		->dilation(6)
		->inputs({"pool5"})
		->outputs({"fc6"});
	layerTestList.push_back(new LearnableLayerTest<float>(convBuilder));
#endif

#if 0
	FullyConnectedLayer<float>::Builder* fcBuilder =
			new typename FullyConnectedLayer<float>::Builder();
	fcBuilder->id(4)
			->name("ip1")
			->nOut(500)
			->inputs({"pool2"})
			->outputs({"ip1"});
	layerTestList.push_back(new LearnableLayerTest<float>(fcBuilder));
#endif

#if 0
	Layer<float>::Builder* reluBuilder =
			new typename ReluLayer<float>::Builder();
	reluBuilder->id(42)
			->name("relu1")
			->inputs({"ip1"})
			->outputs({"relu1"});
	layerTestList.push_back(new LayerTest<float>(reluBuilder));
#endif

#if 0
	PoolingLayer<float>::Builder* poolBuilder =
			new typename PoolingLayer<float>::Builder();
	poolBuilder->id(3)
			->name("pool1/3x3_s2")
			->poolDim(3, 3, 0, 2)
			->poolingType(Pooling<float>::Max)
			->inputs({"conv1/7x7_s2"})
			->outputs({"pool1/3x3_s2"});
	layerTestList.push_back(new LayerTest<float>(poolBuilder));
#endif

#if 0
	SoftmaxWithLossLayer<float>::Builder* softmaxWithLossBuilder =
			new typename SoftmaxWithLossLayer<float>::Builder();
	softmaxWithLossBuilder->id(42)
			->name("loss")
			->inputs({"ip2", "label"})
			->outputs({"loss"});
	layerTestList.push_back(new LayerTest<float>(softmaxWithLossBuilder));
#endif

#if 0
	LRNLayer<float>::Builder* lrnBuilder =
			new typename LRNLayer<float>::Builder();
	lrnBuilder->id(42)
			->name("pool1/norm1")
			->lrnDim(5, 0.0001, 0.75, 1.0)
			->inputs({"pool1/3x3_s2"})
			->outputs({"pool1/norm1"});
	layerTestList.push_back(new LayerTest<float>(lrnBuilder));
#endif

#if 0
	DepthConcatLayer<float>::Builder* depthConcatBuilder =
			new typename DepthConcatLayer<float>::Builder();
	depthConcatBuilder->id(24)
			->name("inception_3a/output")
			->propDown({true, true, true, true})
			->inputs({
				"inception_3a/1x1",
				"inception_3a/3x3",
				"inception_3a/5x5",
				"inception_3a/pool_proj"})
			->outputs({"inception_3a/output"});

	layerTestList.push_back(new LayerTest<float>(depthConcatBuilder));
#endif

#if 0
	SplitLayer<float>::Builder* splitBuilder =
			new typename SplitLayer<float>::Builder();
	splitBuilder->id(24)
			->name("pool2/3x3_s2_pool2/3x3_s2_0_split")
			->inputs({"pool2/3x3_s2"})
			->outputs({"pool2/3x3_s2_pool2/3x3_s2_0_split_0",
				"pool2/3x3_s2_pool2/3x3_s2_0_split_1",
				"pool2/3x3_s2_pool2/3x3_s2_0_split_2",
				"pool2/3x3_s2_pool2/3x3_s2_0_split_3"});

	layerTestList.push_back(new LayerTest<float>(splitBuilder));
#endif

#if 0
	RoIInputLayer<float>::Builder* roiInputBuilder =
			new typename RoIInputLayer<float>::Builder();
	roiInputBuilder->id(0)
			->name("input-data")
			->numClasses(21)
			->pixelMeans({102.9801f, 115.9465f, 122.7717f})	// BGR
			->outputs({"data", "im_info", "gt_boxes"});

	layerTestList.push_back(new LayerInputTest<float>(roiInputBuilder));
#endif

#if 0
	RoITestInputLayer<float>::Builder* roiTestInputBuilder =
			new typename RoITestInputLayer<float>::Builder();
	roiTestInputBuilder->id(0)
			->name("input-data")
			->numClasses(21)
			->pixelMeans({102.9801f, 115.9465f, 122.7717f})	// BGR
			->outputs({"data", "im_info"});

	layerTestList.push_back(new LayerInputTest<float>(roiTestInputBuilder));
#endif

#if 0
	AnchorTargetLayer<float>::Builder* anchorTargetBuilder =
			new typename AnchorTargetLayer<float>::Builder();
	anchorTargetBuilder->id(14)
			->name("rpn-data")
			->featStride(16)
			->inputs({
				"rpn_cls_score_rpn_cls_score_0_split_1",
				"gt_boxes_input-data_2_split_0",
				"im_info_input-data_1_split_0",
				"data_input-data_0_split_1"})
			->propDown({false, false, false, false})
			->outputs({
				"rpn_labels",
				"rpn_bbox_targets",
				"rpn_bbox_inside_weights",
				"rpn_bbox_outside_weights"});

	layerTestList.push_back(new LayerTest<float>(anchorTargetBuilder));
#endif

#if 0
	ProposalLayer<float>::Builder* proposalBuilder =
			new typename ProposalLayer<float>::Builder();
	proposalBuilder->id(19)
			->name("proposal")
			->featStride(16)
			->inputs({
				"rpn_cls_prob_reshape",
				"rpn_bbox_pred_rpn_bbox_pred_0_split_1",
				"im_info_input-data_1_split_1"})
			->propDown({false, false, false})
			->outputs({"rpn_rois"});

	NetworkConfig<float>* networkConfig = (new typename NetworkConfig<float>::Builder())
			->networkPhase(NetworkPhase::TrainPhase)
			->build();

	layerTestList.push_back(new LayerTest<float>(proposalBuilder, networkConfig));
#endif

#if 0
	ProposalTargetLayer<float>::Builder* proposalTargetBuilder =
			new typename ProposalTargetLayer<float>::Builder();
	proposalTargetBuilder->id(20)
			->name("roi-data")
			->numClasses(21)
			->inputs({
				"rpn_rois",
				"gt_boxes_input-data_2_split_1"})
			->propDown({false, false})
			->outputs({
				"rois",
				"labels",
				"bbox_targets",
				"bbox_inside_weights",
				"bbox_outside_weights"});

	layerTestList.push_back(new LayerTest<float>(proposalTargetBuilder));
#endif

#if 0
	ReshapeLayer<float>::Builder* reshapeBuilder =
			new typename ReshapeLayer<float>::Builder();
	reshapeBuilder->id(13)
			->name("rpn_cls_score_reshape")
			->shape({0, 2, -1, 0})
			->inputs({"rpn_cls_score_rpn_cls_score_0_split_0"})
			->propDown({false})
			->outputs({"rpn_cls_score_reshape"});

	layerTestList.push_back(new LayerTest<float>(reshapeBuilder));
#endif

#if 0
	SmoothL1LossLayer<float>::Builder* smoothL1LossBuilder =
			new typename SmoothL1LossLayer<float>::Builder();
	smoothL1LossBuilder->id(16)
			->name("rpn_loss_bbox")
			->lossWeight(1.0f)
			->sigma(3.0f)
			->inputs({
				"rpn_bbox_pred_rpn_bbox_pred_0_split_0",
				"rpn_bbox_targets",
				"rpn_bbox_inside_weights",
				"rpn_bbox_outside_weights"})
			->propDown({false, false, false, false})
			->outputs({"rpn_loss_bbox"});

	layerTestList.push_back(new LayerTest<float>(smoothL1LossBuilder));
#endif

#if 0
	RoIPoolingLayer<float>::Builder* roiPoolingBuilder =
			new typename RoIPoolingLayer<float>::Builder();
	roiPoolingBuilder->id(31)
			->name("roi_pool5")
			->pooledW(6)
			->pooledH(6)
			->spatialScale(0.0625f)
			->inputs({
				"conv5_relu5_0_split_1",
				"rois"})
			->outputs({"pool5"});

	layerTestList.push_back(new LayerTest<float>(roiPoolingBuilder));
#endif

#if 0
	FrcnnTestOutputLayer<float>::Builder* frcnnTestOutputBuilder =
			new typename FrcnnTestOutputLayer<float>::Builder();
	frcnnTestOutputBuilder->id(350)
			->name("test_output")
			//->maxPerImage(5)
			->thresh(0.5)
			->inputs({"rois", "im_info", "cls_prob", "bbox_pred"});

	layerTestList.push_back(new LayerTest<float>(frcnnTestOutputBuilder));
#endif

#if 0
	AnnotationDataLayer<float>::Builder* annotationDataBuilder =
			new typename AnnotationDataLayer<float>::Builder();
	annotationDataBuilder->id(0)
			->name("data")
			->flip(true)
			->imageHeight(300)
			->imageWidth(300)
			->imageSetPath("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/trainval.txt")
			->baseDataPath("/home/jkim/Dev/git/caffe_ssd/data/VOCdevkit/")
			->labelMapPath("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/labelmap_voc.prototxt")
			//->pixelMeans({102.9801f, 115.9465f, 122.7717f})	// BGR
			->pixelMeans({104.f, 117.f, 123.f})	// BGR
			->outputs({"data", "label"});
	layerTestList.push_back(new LayerTest<float>(annotationDataBuilder));
#endif

#if 0
	AccuracyLayer<float>::Builder* accuracyBuilder =
			new typename AccuracyLayer<float>::Builder();
	accuracyBuilder->id(390)
			->name("accuracy")
			->topK(5)
			->axis(2)
			->inputs({"fc8_fc8_0_split_1", "label_data_1_split_1"})
			->outputs({"accuracy"});
	layerTestList.push_back(new LayerTest<float>(accuracyBuilder));
#endif

#if 0
	NormalizeLayer<float>::Builder* normalizeBuilder =
			new typename NormalizeLayer<float>::Builder();
	normalizeBuilder->id(0)
			->name("conv4_3_norm")
			->acrossSpatial(false)
			->scaleFiller(ParamFillerType::Constant, 20.0f)
			->channelShared(false)
			->inputs({"conv4_3_relu4_3_0_split_1"})
			->outputs({"conv4_3_norm"});

	NetworkConfig<float>* networkConfig = (new typename NetworkConfig<float>::Builder())
			->networkPhase(NetworkPhase::TrainPhase)
			->batchSize(4)
			->build();

	layerTestList.push_back(new LearnableLayerTest<float>(normalizeBuilder, networkConfig));
#endif

#if 0
	PermuteLayer<float>::Builder* builder =
			new typename PermuteLayer<float>::Builder();
	builder->id(0)
			->name("conv4_3_norm_mbox_loc_perm")
			->orders({0, 2, 3, 1})
			->inputs({"conv4_3_norm_mbox_loc"})
			->outputs({"conv4_3_norm_mbox_loc_perm"});

	layerTestList.push_back(new LayerTest<float>(builder));
#endif

#if 0
	FlattenLayer<float>::Builder* builder =
			new typename FlattenLayer<float>::Builder();
	builder->id(0)
			->name("conv4_3_norm_mbox_loc_flat")
			->axis(1)
			->endAxis(3)
			->inputs({"conv4_3_norm_mbox_loc_perm"})
			->outputs({"conv4_3_norm_mbox_loc_flat"});

	layerTestList.push_back(new LayerTest<float>(builder));
#endif

#if 0
	PriorBoxLayer<float>::Builder* builder =
			new typename PriorBoxLayer<float>::Builder();
	builder->id(0)
			->name("conv4_3_norm_mbox_priorbox")
			->minSizes({30.0})
			->maxSizes({60.0})
			->aspectRatios({2.0})
			->flip(true)
			->clip(false)
			->variances({0.1, 0.1, 0.2, 0.2})
			->step(8.0)
			->offset(0.5)
			->inputs({"conv4_3_norm_conv4_3_norm_0_split_2", "data_data_0_split_1"})
			->outputs({"conv4_3_norm_mbox_priorbox"});

	layerTestList.push_back(new LayerTest<float>(builder));
#endif

#if 0
	ConcatLayer<float>::Builder* builder =
			new typename ConcatLayer<float>::Builder();
	builder->id(0)
			->name("mbox_loc")
			->axis(1)
			->inputs({
				"conv4_3_norm_mbox_loc_flat",
				"fc7_mbox_loc_flat",
				"conv6_2_mbox_loc_flat",
				"conv7_2_mbox_loc_flat",
				"conv8_2_mbox_loc_flat",
				"conv9_2_mbox_loc_flat"})
			->outputs({"mbox_loc"});

	layerTestList.push_back(new LayerTest<float>(builder));
#endif

#if 0
	MultiBoxLossLayer<float>::Builder* builder =
			new typename MultiBoxLossLayer<float>::Builder();
	builder->id(0)
			->name("mbox_loss")
			->locLossType("SMOOTH_L1")
			->confLossType("SOFTMAX")
			->locWeight(1.0)
			->numClasses(21)
			->shareLocation(true)
			->matchType("PER_PREDICTION")
			->overlapThreshold(0.5)
			->usePriorForMatching(true)
			->backgroundLabelId(0)
			->useDifficultGt(true)
			->negPosRatio(3.0)
			->negOverlap(0.5)
			->codeType("CENTER_SIZE")
			->ignoreCrossBoundaryBbox(false)
			->miningType("MAX_NEGATIVE")
			->propDown({true, true, false, false})
			->inputs({"mbox_loc", "mbox_conf", "mbox_priorbox", "label"})
			->outputs({"mbox_loss"});

	layerTestList.push_back(new LayerTest<float>(builder));
#endif

#if 0
	DetectionOutputLayer<float>::Builder* builder =
			new typename DetectionOutputLayer<float>::Builder();
	builder->id(0)
			->name("detection_out")
			->numClasses(21)
			->shareLocation(true)
			->backgroundLabelId(0)
			->nmsThreshold(0.449999988079)
			->topK(400)
			->outputDirectory("/home/jkim/Dev/data/ssd/data/VOCdevkit/results/VOC2007/SSD_300x300/Main")
			->outputNamePrefix("comp4_det_test_")
			->outputFormat("VOC")
			->labelMapFile("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/labelmap_voc.prototxt")
			->nameSizeFile("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/test_name_size.txt")
			->numTestImage(4952)
			->codeType("CENTER_SIZE")
			->keepTopK(200)
			->confidenceThreshold(0.00999999977648)
			->visualize(true)
			->propDown({false, false, false, false})
			->inputs({"mbox_loc", "mbox_conf_flatten", "mbox_priorbox", "data_data_0_split_7"})
			->outputs({"detection_out"});

	layerTestList.push_back(new LayerTest<float>(builder));
#endif

#if 0
	DetectionEvaluateLayer<float>::Builder* builder =
			new typename DetectionEvaluateLayer<float>::Builder();
	builder->id(0)
			->name("detection_eval")
			->numClasses(21)
			->backgroundLabelId(0)
			->overlapThreshold(0.5)
			->evaluateDifficultGt(false)
			->nameSizeFile("/home/jkim/Dev/git/caffe_ssd/data/VOC0712/test_name_size.txt")
			->propDown({false, false})
			->inputs({"detection_out", "label"})
			->outputs({"detection_eval"});

	layerTestList.push_back(new LayerTest<float>(builder));
#endif

	LayerTestInterface<float>::globalSetUp(gpuid);
	for (uint32_t i = 0; i < layerTestList.size(); i++) {
		LayerTestInterface<float>* layerTest = layerTestList[i];
		layerTest->setUp();
		layerTest->forwardTest();
		layerTest->backwardTest();
		layerTest->cleanUp();
	}
	LayerTestInterface<float>::globalCleanUp();
}





#define NETWORK_LENET		0
#define NETWORK_VGG19		1
#define NETWORK_FRCNN		2
#define NETWORK_FRCNN_TEST	3
#define NETWORK_SSD			4
#define NETWORK_SSD_TEST	5
#define NETWORK_VGG16		6
#define NETWORK_DUMMY		7
#define NETWORK				NETWORK_FRCNN

#define EXAMPLE_LENET_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/git/soooa/SoooA/src/examples/LeNet/lenet_train_test.json")
#define EXAMPLE_FRCNN_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/git/soooa/SoooA/src/examples/frcnn/frcnn_train_test.json")
#define EXAMPLE_VGG16_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/git/soooa/SoooA/src/examples/VGG16/vgg16_train_test.json")
#define EXAMPLE_DUMMY_TRAIN_NETWORK_FILEPATH	("/home/jkim/Dev/SOOOA_HOME/.json")


//void saveNetworkParams(LayersConfig<float>* layersConfig);


void networkTest() {
	const int gpuid 				= 0;


	WorkContext::curBootMode = BootMode::DeveloperMode;

	InitParam::init();
	Perf::init();
	SysLog::init();
	ColdLog::init();
	Job::init();
	Task::init();
	Broker::init();
	Network<float>::init();

	ResourceManager::init();
	PlanOptimizer::init();
	LayerFunc::init();
	LayerPropList::init();

	Util::setOutstream(&cout);
	Util::setPrint(false);


#if NETWORK == NETWORK_LENET
	const string networkFilePath = string(EXAMPLE_LENET_TRAIN_NETWORK_FILEPATH);
	const string networkName = "lenet";
	const int numSteps = 3;
#elif NETWORK == NETWORK_VGG19
	// VGG19
	//LayersConfig<float>* layersConfig = createVGG19NetLayersConfig<float>();
	LayersConfig<float>* layersConfig = createInceptionLayersConfig<float>();
	const string networkName		= "inception";
	const int batchSize 			= 8;
	const float baseLearningRate 	= 0.01;
	const float weightDecay 		= 0.0002;
	const float momentum 			= 0.0;
	const LRPolicy lrPolicy 		= LRPolicy::Step;
	const int stepSize 				= 320000;
	const NetworkPhase networkPhase	= NetworkPhase::TrainPhase;
	const float gamma 				= 0.96;
#elif NETWORK == NETWORK_FRCNN
	const string networkFilePath = string(EXAMPLE_FRCNN_TRAIN_NETWORK_FILEPATH);
	const string networkName = "frcnn";
	const int numSteps = 2;

	/*
	// FRCNN
	LayersConfig<float>* layersConfig = createFrcnnTrainOneShotLayersConfig<float>();
	const string networkName		= "frcnn";
	const int batchSize 			= 1;
	const float baseLearningRate 	= 0.01;
	const float weightDecay 		= 0.0005;
	//const float baseLearningRate 	= 1;
	//const float weightDecay 		= 0.000;
	const float momentum 			= 0.0;
	const LRPolicy lrPolicy 		= LRPolicy::Step;
	const int stepSize 				= 50000;
	const NetworkPhase networkPhase	= NetworkPhase::TrainPhase;
	const float gamma 				= 0.1;
	*/
#elif NETWORK == NETWORK_FRCNN_TEST
	// FRCNN_TEST
	LayersConfig<float>* layersConfig = createFrcnnTestOneShotLayersConfig<float>();
	const string networkName		= "frcnn";
	const NetworkPhase networkPhase	= NetworkPhase::TestPhase;
#elif NETWORK == NETWORK_SSD
	const int numSteps = 3;

	LayersConfig<float>* layersConfig = createSSDNetLayersConfig<float>();
	const string networkName		= "ssd";
	const int batchSize 			= 1;
	const float baseLearningRate 	= 0.001;
	const float weightDecay 		= 0.0005;
	//const float baseLearningRate 	= 1;
	//const float weightDecay 		= 0.000;
	//const float momentum 			= 0.0;
	const float momentum 			= 0.0;
	const LRPolicy lrPolicy 		= LRPolicy::Fixed;
	const int stepSize 				= 50000;
	const NetworkPhase networkPhase	= NetworkPhase::TrainPhase;
	const float gamma 				= 0.1;
#elif NETWORK == NETWORK_SSD_TEST
	const int numSteps = 1;

	LayersConfig<float>* layersConfig = createSSDNetTestLayersConfig<float>();
	const string networkName		= "ssd";
	const int batchSize 			= 2;
	const float baseLearningRate 	= 0.01;
	const float weightDecay 		= 0.0005;
	//const float baseLearningRate 	= 1;
	//const float weightDecay 		= 0.000;
	//const float momentum 			= 0.0;
	const float momentum 			= 0.0;
	const LRPolicy lrPolicy 		= LRPolicy::Fixed;
	const int stepSize 				= 50000;
	const NetworkPhase networkPhase	= NetworkPhase::TrainPhase;
	const float gamma 				= 0.1;
#elif NETWORK == NETWORK_VGG16
	const string networkFilePath = string(EXAMPLE_VGG16_TRAIN_NETWORK_FILEPATH);
	const string networkName = "vgg16";
	const int numSteps = 1;
#else
	cout << "invalid network ... " << endl;
	exit(1);
#endif



#if 0
	Network<float>* network = new Network<float>(networkConfig);
	// (2) network config 정보를 layer들에게 전달한다.
	for(uint32_t i = 0; i < layersConfig->_layers.size(); i++) {
		layersConfig->_layers[i]->setNetworkConfig(network->config);
	}
	network->setLayersConfig(layersConfig);
	network->loadPretrainedWeights();
#endif

#if 0
	Data<float>::printConfig = true;
	SyncMem<float>::printConfig = true;
	for (int i = 0; i < layersConfig->_learnableLayers.size(); i++) {
		for (int j = 0; j < layersConfig->_learnableLayers[i]->_params.size(); j++) {
			layersConfig->_learnableLayers[i]->_params[j]->print_data({}, false);
		}
	}
	Data<float>::printConfig = false;
	SyncMem<float>::printConfig = false;
	exit(1);
#endif

	NetworkTestInterface<float>::globalSetUp(gpuid);

	NetworkTest<float>* networkTest =
			new NetworkTest<float>(networkFilePath, networkName, numSteps);

	networkTest->setUp();
	networkTest->updateTest();
	networkTest->cleanUp();

	NetworkTestInterface<float>::globalCleanUp();
}


#if 0
void saveNetworkParams(LayersConfig<float>* layersConfig) {
	const string savePathPrefix = "/home/jkim/Dev/SOOOA_HOME/network";
	ofstream paramOfs(
			(savePathPrefix+"/SSD_CAFFE_TRAINED.param").c_str(),
			ios::out | ios::binary);

	uint32_t numLearnableLayers = layersConfig->_learnableLayers.size();
	uint32_t numParams = 0;
	for (uint32_t i = 0; i < numLearnableLayers; i++) {
		numParams += layersConfig->_learnableLayers[i]->numParams();
	}

	paramOfs.write((char*)&numParams, sizeof(uint32_t));
	for (uint32_t i = 0; i < numLearnableLayers; i++) {
		layersConfig->_learnableLayers[i]->saveParams(paramOfs);
	}
	paramOfs.close();
}
#endif
