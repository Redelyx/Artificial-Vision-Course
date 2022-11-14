#include "Functions.h"

struct ArgumentList {
    std::string image_name; //!< image file name
    int m_wait; //!< waiting time
    int m_padding;
    int u_tl;
    int v_tl;
    int width_crop;
    int height_crop;
    unsigned int ex;
    bool random_crop;
};

cv::Mat simpleBin(const cv::Mat image, int threshold)
{
    cv::Mat out = cv::Mat(image.rows, image.cols, image.type());
    for (int r = 0; r < image.rows; r++) {
        for (int c = 0; c < image.cols; c++) {
            if ((int)image.data[(c + r * image.cols) * image.elemSize()] > threshold)
                out.data[(c + r * image.cols) * image.elemSize()] = 255;
            else
                out.data[(c + r * image.cols) * image.elemSize()] = 0;
        }
    }
    return out;
}

void gaussianKrnl(float sigma, int r, cv::Mat& krnl)
{
}

void GaussianBlur(const cv::Mat& src, float sigma, int r, cv::Mat& out, int stride = 1)
{
}

cv::Mat verticalSobel()
{
    cv::Mat krn = cv::Mat(3, 3, CV_32F);
    krn.at<float>(0, 0) = 1;
    krn.at<float>(0, 1) = 0;
    krn.at<float>(0, 2) = -1;
    krn.at<float>(1, 0) = 2;
    krn.at<float>(1, 1) = 0;
    krn.at<float>(1, 2) = -2;
    krn.at<float>(2, 0) = 1;
    krn.at<float>(2, 1) = 0;
    krn.at<float>(2, 2) = -1;
    return krn;
}

cv::Mat horizontalSobel()
{
    cv::Mat krn = cv::Mat(3, 3, CV_32F);
    krn.at<float>(0, 0) = 1;
    krn.at<float>(0, 1) = 2;
    krn.at<float>(0, 2) = 1;
    krn.at<float>(1, 0) = 0;
    krn.at<float>(1, 1) = 0;
    krn.at<float>(1, 2) = 0;
    krn.at<float>(2, 0) = -1;
    krn.at<float>(2, 1) = -2;
    krn.at<float>(2, 2) = -1;
    return krn;
}

void sobel3x3(const cv::Mat& src, cv::Mat& magn, cv::Mat& orient)
{
}

float bilinear(const cv::Mat& src, float r, float c)
{
    return 0;
}

int findPeaks(const cv::Mat& magn, const cv::Mat& src, cv::Mat& out)
{
    return 0;
}

int doubleTh(const cv::Mat& magn, cv::Mat& out, float t1, float t2)
{
    return 0;
}

cv::Mat dilation(const cv::Mat image, const cv::Mat se)
{
}

cv::Mat erosion(const cv::Mat image, const cv::Mat se)
{
}

cv::Mat closing(const cv::Mat image, const cv::Mat se)
{
}

cv::Mat opening(const cv::Mat image, const cv::Mat se)
{
}

double otsuDeviation(const cv::Mat image, int th)
{
}

void lab1(const cv::Mat& image, std::string image_name)
{
    cv::Mat new_m = downsampling2x(image);
    openandwait("downsample2x", new_m, false);
    /*etc.etc...*/
}

void lab1a(const cv::Mat& image, std::string image_name)
{
    sample(image, image_name);
}

bool ParseInputs(ArgumentList& args, int argc, char** argv)
{
    int c;
    args.m_wait = 0;
    while ((c = getopt(argc, argv, "hi:t:p:u:v:W:H:rx:")) != -1)
        switch (c) {
        case 'i':
            args.image_name = optarg;
            break;
        case 't':
            args.m_wait = atoi(optarg);
            break;
        case 'h':
        default:
            std::cout << "usage: " << argv[0] << " -i <image_name>" << std::endl;
            std::cout << "exit:  type q" << std::endl
                      << std::endl;
            std::cout << "Allowed options:" << std::endl
                      << "   -h                       produce help message"
                      << std::endl
                      << "   -i arg                   image name. Use %0xd format "
                         "for multiple images."
                      << std::endl
                      << "   -t arg                   wait before next frame (ms) "
                         "[default = 0]"
                      << std::endl
                      << std::endl
                      << std::endl;
            return false;
        }
    return true;
}

int main(int argc, char** argv)
{
    std::cout << "----- Esercitazione 4 Edges -----" << std::endl;

    int frame_number = 0;
    char frame_name[256];
    bool exit_loop = false;
    unsigned char key;
    ArgumentList args;

    //////////////////////
    // parse argument list:
    //////////////////////

    if (!ParseInputs(args, argc, argv)) {
        return 1;
    }

    //---Lab2---
    int threshold = 15;
    int k = 5;
    float alpha = 0.5;
    std::vector<cv::Mat> frames(k);
    //----------

    while (!exit_loop) {
        // multi frame case
        if (args.image_name.find('%') != std::string::npos)
            sprintf(frame_name, (const char*)(args.image_name.c_str()),
                frame_number);
        else // single frame case
            sprintf(frame_name, "%s", args.image_name.c_str());

        // opening file
        std::cout << "Opening " << frame_name << std::endl;

        cv::Mat image;
        if (args.image_name.find("RGGB") != std::string::npos
            || args.image_name.find("GBRG") != std::string::npos
            || args.image_name.find("BGGR") != std::string::npos
            || args.image_name.find("organs") != std::string::npos)
            image = cv::imread(frame_name, CV_8UC1);
        else
            image = cv::imread(frame_name);

        if (image.empty()) {
            std::cout << "Unable to open " << frame_name << std::endl;
            return 1;
        }

        // display image
        openandwait("Original Image", image, false);

        cv::Mat out;
        /*//---Lab 2---

        if (frame_number == 0) {
                for(int i = 0; i<frames.size(); i++){
                        frames[i] = cv::Mat(image.rows, image.cols, image.type(), cv::Scalar(0));
                }
}

        out = simpleBgSubtraction(image, threshold);

        out = expRunningAverageBgSubtraction(image, threshold, alpha);
        openandwait("out", out, false);
        */
        int th;
        std::cout << "Insert threshold: ";
        std::cin >> th;

        out = simpleBin(image, th);
        openandwait("out", out, false);

        // wait for key or timeout

        unsigned char key = cv::waitKey(args.m_wait);
        std::cout << "key " << int(key) << std::endl;

        // here you can implement some looping logic using key value:
        //  - pause
        //  - stop
        //  - step back
        //  - step forward
        //  - loop on the same frame
        if (key == 'q')
            exit_loop = true;

        frame_number++;
    }
    return 0;
}
