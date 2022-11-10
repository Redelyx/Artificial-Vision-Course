#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <ctime>        // std::time

//getopt()
#include <unistd.h>


//---UTILITY---
unsigned char openandwait(const char *windowname, cv::Mat &img, const bool sera);


//---LAB 1---
cv::Mat downsampling2x(const cv::Mat image);
cv::Mat downsampling2xVert(const cv::Mat image);
cv::Mat downsampling2xHoriz(const cv::Mat image);
cv::Mat flipHoriz(const cv::Mat image);
cv::Mat flipVert(const cv::Mat image);
cv::Mat crop(const cv::Mat image, int x, int y, int w, int h);
cv::Mat addPadding(const cv::Mat image, int vPadding, int hPadding);
cv::Mat splitIn4(const cv::Mat image);
cv::Mat colorShuffle(const cv::Mat image);
void sample(const cv::Mat image, std::string image_name);

//---LAB 1a---

cv::Mat myfilter2D(const cv::Mat src, const cv::Mat& krn, cv::Mat& out,  int stride);

//---LAB 3---

//---LAB 4---
