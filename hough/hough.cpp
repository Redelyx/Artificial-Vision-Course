// OpneCV
#include "../utils.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

/*Compute Hough Transform
- src: cv::Mat resulting from canny filter, type is uint8_t
- acc: cv::Mat with accumulation matrix from hough transform */
void houghTransform(const cv::Mat& src, cv::Mat& acc)
{
    int theta = 360;
    float rho = sqrt(src.cols * src.cols + src.rows * src.rows);

    acc = cv::Mat::zeros(theta, rho, CV_32SC1);
    
	for (int r = 0; r < src.rows; r++) {
		for (int c = 0; c < src.cols; c++) {
			if (src.at<uint8_t>(r, c) != 0) {
				//std::cout << "r,c: " << r << " " << c << std::endl;
				for (int a = 0; a < theta; a++) {
                    float rad = a * CV_PI / 180;
                    float dist = r * sin(rad) + c * cos(rad);
                    acc.at<int>(a, (int)dist)++;
                }
            }
        }
    }
}

void houghTransformNMS(const cv::Mat& src, cv::Mat& acc)
{
    acc = src;

    for (int r = 1; r < src.rows - 1; r++) {
        for (int c = 1; c < src.cols - 1; c++) {
            for (int r_i = -1; r_i < 2; r_i++) {
                for (int c_i = -1; c_i < 2; c_i++) {
                    if (acc.at<int>(r, c) < acc.at<int>(r + r_i, c + c_i)) {
                        acc.at<int>(r, c) = 0;
                    }
                }
            }
        }
    }
}

void houghDrawLines(const cv::Mat& src, const cv::Mat& img, cv::Mat& color, int th)
{

    cv::cvtColor(img, color, cv::COLOR_BGR2RGB);

    for (int r = 0; r < src.rows; r++) { // angolo
        for (int c = 0; c < src.cols; c++) { // dist
            if (src.at<int>(r, c) > th) {
				//std::cout << " theta: " << r << " rho: " << c  <<std::endl;
                cv::Point pt1, pt2;
				float rad = r * CV_PI / 180;
				float m = - cos(rad)/sin(rad);

				int q = c/sin(rad); 
				
                pt1.x = -(src.rows+src.cols);
                pt1.y = m*pt1.x+q;
                pt2.x = (src.rows+src.cols);
                pt2.y = m*pt2.x+q;

                cv::line(color, pt1, pt2, cv::Scalar(0, 0, 88), 1, cv::LINE_AA);

            }
        }
    }
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	cv::imshow("image", color);
}
struct ArgumentList {
    std::string image_name; //!< image file name
    int wait_t; //!< waiting time
};

bool ParseInputs(ArgumentList& args, int argc, char** argv);

int main(int argc, char** argv)
{
    int th = 180;

    int imreadflags = cv::IMREAD_GRAYSCALE;

    std::cout << "Simple program." << std::endl;

    //////////////////////
    // parse argument list:
    //////////////////////
    ArgumentList args;
    if (!ParseInputs(args, argc, argv)) {
        exit(0);
    }

    // opening file
    std::cout << "Opening " << args.image_name << std::endl;

    cv::Mat image = cv::imread(args.image_name.c_str(), imreadflags);
    if (image.empty()) {
        std::cout << "Unable to open " << args.image_name << std::endl;
        return 1;
    }
    std::cout << "The image has " << image.elemSize() << " channels, the size is " << image.rows << "x" << image.cols << " pixels "
              << " the type is " << image.type() << " the pixel size is " << image.elemSize() << " and each channel is " << image.elemSize1() << (image.elemSize1() > 1 ? " bytes" : " byte") << std::endl;

    // display original image
    cv::namedWindow("original image", cv::WINDOW_NORMAL);
    cv::imshow("original image", image);

    cv::Mat blurred, edges;
    cv::GaussianBlur(image, blurred, cv::Size(3, 3), 3);
    cv::Canny(blurred, edges, 50, 150);

    // display canny edges
    cv::namedWindow("CANNY", cv::WINDOW_NORMAL);
    cv::imshow("CANNY", edges);

    // YOUR CODE HERE: COMPUTE ACCUMULATOR

    cv::Mat accumulator, accumulator_nms;

    houghTransform(edges, accumulator);

    // display accumulator
    cv::Mat displa, displa1;
    cv::convertScaleAbs(accumulator, displa);
    cv::namedWindow("Hough accumulator", cv::WINDOW_AUTOSIZE);
    cv::imshow("Hough accumulator", displa);

    // YOUR CODE HERE: NON MAXIMA SUPPRESSION FOR ACCUMULATOR (ignore borders maybe)

    houghTransformNMS(accumulator, accumulator_nms);

    // post NMS we convert again to 0-255 and apply a threshold
/*     cv::convertScaleAbs(accumulator_nms, displa1);
    cv::threshold(displa1, displa1, 180, 255, cv::THRESH_BINARY);
    cv::namedWindow("Hough accumulator1", cv::WINDOW_AUTOSIZE);
    cv::imshow("Hough accumulator1", displa1); */

    // image on which we can draw lines
    cv::Mat color;
    // YOUR CODE HERE draw lines

    houghDrawLines(accumulator_nms, image, color, th);

    // display image
    cv::namedWindow("image", cv::WINDOW_NORMAL);
    cv::imshow("image", color);

    // wait for Q key or timeout
    unsigned char c;
    while ((c = cv::waitKey(args.wait_t)) != 'q' && c != 'Q')
        ;

    return 0;
}

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
