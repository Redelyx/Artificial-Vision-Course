/*L03*/

#include "../utils.h"

struct ArgumentList {
    std::string image_name; //!< image file name
};

bool ParseInputs(ArgumentList& args, int argc, char** argv)
{
    int c;

    while ((c = getopt(argc, argv, "hi:")) != -1)
        switch (c) {
        case 'i':
            args.image_name = optarg;
            break;
        case 'h':
        default:
            std::cout << "usage: " << argv[0] << " -i <image_name>" << std::endl;
            std::cout << "exit:  type q" << std::endl
                      << std::endl;
            std::cout << "Allowed options:" << std::endl
                      << "   -h                       produce help message" << std::endl
                      << "   -i arg                   image name. Use %0xd format for multiple images." << std::endl
                      << std::endl;
            return false;
        }
    return true;
}

int main(int argc, char** argv)
{
    char frame_name[256];

    std::cout << "Simple program." << std::endl;

    //////////////////////
    // parse argument list:
    //////////////////////
    ArgumentList args;
    if (!ParseInputs(args, argc, argv)) {
        return 1;
    }

    // generating file name
    sprintf(frame_name, "%s", args.image_name.c_str());

    // opening file
    std::cout << "Opening " << frame_name << std::endl;

    // Per questo esercizione l'immagine e' sicuramente a toni di grigio
    cv::Mat image = cv::imread(frame_name, CV_8UC1);
    cv::Mat out, out1, out2, out3, out4;

    if (image.empty()) {
        std::cout << "Unable to open " << frame_name << std::endl;
        return 1;
    }

    simpleOpen("image", image);

    //////////////////////
    // processing code here

	// ES1 Soglia adattiva
    //

    cv::Mat best_bin(image.rows, image.cols, CV_8UC1);

    minThreshBin(image, 50, out);

    simpleOpen("otsu", out);

    /* double th = cv::threshold(image, out, 0, 255, cv::THRESH_OTSU);
    simpleBin(image, th, out1);
    simpleOpen("otsu2", out1); */

    // ES3 Morfologia Matematica
    //
	// elemento strutturante 3x3
    //
    //    1  1  1
    //    1  1* 1
    //    1  1  1

	cv::Mat se = cv::Mat(3, 3, CV_8UC1, cv::Scalar(1));
	int cx = 1, cy = 1;

	dilation(out, se, cx, cy, out1);
	simpleOpen("dilation", out1);

/* 	cv::dilate(out, out1, se);
	simpleOpen("dilation1", out1); */

	erosion(out, se, cx, cy, out2);
	simpleOpen("erosion", out2);

/* 	cv::erode(out, out2, se);
	simpleOpen("erosion1", out2); */

	/* cv::morphologyEx(out, out1, cv::MORPH_OPEN, se);
	simpleOpen("opening_tru", out1); */

	opening(out, se, cx, cy, out3);
	simpleOpen("opening", out3);

	/* cv::morphologyEx(out, out2, cv::MORPH_CLOSE, se);
	simpleOpen("closing_tru", out2); */

	closing(out, se, cx, cy, out4);
	simpleOpen("closing", out4);


    // wait for key
    cv::waitKey();

    return 0;
}
