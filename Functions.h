#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

// std:
#include <ctime> // std::time
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// getopt()
#include <unistd.h>

struct ArgumentList {
    std::string image_name; //!< image file name
    int m_wait; //!< waiting time
};

//---Esercizi Completi---

int lab1a(ArgumentList args);
int lab1(ArgumentList args); 
int lab2(ArgumentList args);
int lab3 (ArgumentList args);

//---UTILITY---
cv::Mat transpose(const cv::Mat image);
unsigned char openandwait(const char* windowname, cv::Mat& img, const bool sera);
std::string type2str(int type);
void matTypeCV(const cv::Mat M);

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
cv::Mat conv(const cv::Mat &src, const cv::Mat &krn, int stride);

//---LAB 2---

cv::Mat simpleBgSubtraction(const cv::Mat image, int threshold);
cv::Mat runningAverageBgSubtraction(const cv::Mat image, std::vector<cv::Mat>& v, int threshold, int n);
cv::Mat expRunningAverageBgSubtraction(const cv::Mat image, int threshold, float alpha);

//---LAB 3---

cv::Mat simpleBin(const cv::Mat &image, int threshold);

//---LAB 4---
