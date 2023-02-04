/*L01-BAYER*/
#include "../utils.h"

struct ArgumentList {
    std::string image_name; //!< image file name
    int wait_t; //!< waiting time
};

bool ParseInputs(ArgumentList& args, int argc, char** argv);

int main(int argc, char** argv)
{
    int frame_number = 0;
    char frame_name[256];
    bool exit_loop = false;
    int imreadflags = cv::IMREAD_COLOR;

    std::cout << "Simple program." << std::endl;

    //////////////////////
    // parse argument list:
    //////////////////////
    ArgumentList args;
    if (!ParseInputs(args, argc, argv)) {
        exit(0);
    }

    while (!exit_loop) {
        // generating file name
        //
        // multi frame case
        if (args.image_name.find('%') != std::string::npos)
            sprintf(frame_name, (const char*)(args.image_name.c_str()), frame_number);
        else // single frame case
            sprintf(frame_name, "%s", args.image_name.c_str());

        // opening file
        std::cout << "Opening " << frame_name << std::endl;

        cv::Mat image;
        bool isBayer;

        if (args.image_name.find("RGGB") != std::string::npos || args.image_name.find("GBRG") != std::string::npos || args.image_name.find("BGGR") != std::string::npos) {
            image = cv::imread(frame_name, CV_8UC1);
            isBayer = true;
        } else {
            image = cv::imread(frame_name);
            isBayer = false;
        }
        if (image.empty()) {
            std::cout << "Unable to open " << frame_name << std::endl;
            return 1;
        }

        std::cout << "The image has " << image.elemSize() << " channels, the size is " << image.rows << "x" << image.cols << " pixels "
                  << " the type is " << image.type() << " the pixel size is " << image.elemSize() << " and each channel is " << image.elemSize1() << (image.elemSize1() > 1 ? " bytes" : " byte") << std::endl;

        // display image
        openandwait("original image", image, false);

        cv::Mat out;

        //////////////////////
        // processing code here
        if (isBayer) {
            bayerDownsample(image, out, args.image_name);
            openandwait("bayer downsample", out);

            bayerLuminance(image, out, args.image_name);
            openandwait("bayer luminance", out);

            bayerSimple(image, out, args.image_name);
            openandwait("bayer simple", out);
        } else {

            // ./simple -i images/Lenna.png

            downsampling2x(image, out);
            openandwait("downsampling2x", out);

            downsampling2xCol(image, out);
            openandwait("downsampling col", out);

            downsampling2xRow(image, out);
            openandwait("downsampling row", out);

            flipHoriz(image, out);
            openandwait("flip horiz", out);

            flipVert(image, out);
            openandwait("flip vert", out);

            crop(image, out, 130, 190, 80, 60);
            openandwait("crop", out);

            randomCrop(image, out);
            openandwait("random crop", out);

            addPadding(image, out, 10, 20);
            openandwait("padded", out);

            splitAndShuffle(image, out);
            openandwait("split and shuffle", out);

            colorShuffle(image, out);
            openandwait("color shuffle", out);
        }
        /////////////////////

        // wait for key or timeout
        unsigned char key = cv::waitKey(args.wait_t);
        std::cout << "key " << int(key) << std::endl;

        // here you can implement some looping logic using key value:
        //  - pause
        //  - stop
        //  - step back
        //  - step forward
        //  - loop on the same frame

        switch (key) {
        case 'p':
            std::cout << "Mat = " << std::endl
                      << image << std::endl;
            break;
        case 'q':
            exit_loop = 1;
            break;
        case 'c':
            std::cout << "SET COLOR imread()" << std::endl;
            imreadflags = cv::IMREAD_COLOR;
            break;
        case 'g':
            std::cout << "SET GREY  imread()" << std::endl;
            imreadflags = cv::IMREAD_GRAYSCALE; // Y = 0.299 R + 0.587 G + 0.114 B
            break;
        }

        frame_number++;
    }

    return 0;
}

#if 0
bool ParseInputs(ArgumentList& args, int argc, char **argv) {
  args.wait_t=0;

  cv::CommandLineParser parser(argc, argv,
      "{input   i|in.png|input image, Use %0xd format for multiple images.}"
      "{wait    t|0     |wait before next frame (ms)}"
      "{help    h|<none>|produce help message}"
      );

  if(parser.has("help"))
  {
    parser.printMessage();
    return false;
  }

  args.image_name = parser.get<std::string>("input");
  args.wait_t     = parser.get<int>("wait");

  return true;
}
#else

#include <unistd.h>
bool ParseInputs(ArgumentList& args, int argc, char** argv)
{
    int c;

    while ((c = getopt(argc, argv, "hi:t:")) != -1)
        switch (c) {
        case 't':
            args.wait_t = atoi(optarg);
            break;
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
                      << "   -t arg                   wait before next frame (ms)" << std::endl
                      << std::endl;
            return false;
        }
    return true;
}

#endif
