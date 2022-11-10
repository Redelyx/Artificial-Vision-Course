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

//esercizi delle scorse esercitazioni

unsigned char openandwait(const char *windowname, cv::Mat &img, const bool sera=true);

void downsampling2x(cv::Mat &img);