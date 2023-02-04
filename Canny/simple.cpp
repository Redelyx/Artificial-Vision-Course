//OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

//getopt()
#include <unistd.h>
#include <queue>
struct ArgumentList
{
  std::string image_name; //!< image file name
  int wait_t;             //!< waiting time
};

bool ParseInputs(ArgumentList &args, int argc, char **argv)
{
  int c;
  args.wait_t = 0;

  while ((c = getopt(argc, argv, "hi:t:")) != -1)
    switch (c)
    {
    case 'i':
      args.image_name = optarg;
      break;
    case 't':
      args.wait_t = atoi(optarg);
      break;
    case 'h':
    default:
      std::cout << "usage: " << argv[0] << " -i <image_name>" << std::endl;
      std::cout << "exit:  type q" << std::endl
                << std::endl;
      std::cout << "Allowed options:" << std::endl
                << "   -h                       produce help message" << std::endl
                << "   -i arg                   image name. Use %0xd format for multiple images." << std::endl
                << "   -t arg                   wait before next frame (ms) [default = 0]" << std::endl
                << std::endl
                << std::endl;
      return false;
    }
  return true;
}
void addPadding(const cv::Mat &src, cv::Mat &out, int paddingR, int paddingC)
{
  out = cv::Mat(src.rows + paddingR * 2, src.cols + paddingC * 2, src.type());
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      for (int c = 0; c < src.elemSize() / src.elemSize1(); c++)
      {
        out.data[(j + paddingC + (i + paddingR) * out.cols) * out.elemSize()] = src.data[(j + i * src.cols) * src.elemSize()];
      }
    }
  }
}
void conv(const cv::Mat &src, const cv::Mat &krn, cv::Mat &out, int stride = 1)
{
  out = cv::Mat(src.rows, src.cols, CV_32SC1);
  cv::Mat image;
  int paddingR = krn.rows / 2;
  int paddingC = krn.cols / 2;
  addPadding(src, image, paddingR, paddingC);
  for (int i = paddingR; i < image.rows - paddingR; i += stride)
  {
    for (int j = paddingC; j < image.cols - paddingC; j += stride)
    {
      float sumKernel = 0;
      for (int yK = -paddingR; yK <= paddingR; yK++)
      {
        for (int xK = -paddingC; xK <= paddingC; xK++)
        {
          sumKernel += krn.at<float>(yK + paddingR, xK + paddingC) * image.at<unsigned char>(i + yK, j + xK);
        }
      }
      out.at<int>(i - paddingR, j - paddingC) = sumKernel;
    }
  }
}
void gaussianKrnl(float sigma, int r, cv::Mat &krnl)
{
  krnl = cv::Mat(2 * r + 1, 1, CV_32F);
  float sumOfGaussian = 0;
  for (int i = -r; i <= r; i++)
  {
    float value = exp(-(i * i) / (2 * sigma * sigma));
    sumOfGaussian += value;
    krnl.at<float>(i + r, 0) = value;
  }

  krnl /= sumOfGaussian;
  for (int i = -r; i <= r; i++)
  {
    std::cout << krnl.at<float>(i + r, 0) << std::endl;
  }
}
void GaussianBlur(const cv::Mat &src, float sigma, int r, cv::Mat &out, int stride = 1)
{
  cv::Mat krn;
  gaussianKrnl(sigma, r, krn);
  cv::Mat tmp;
  conv(src, krn, tmp, stride);

  cv::Mat krn2(1, 2 * r + 1, CV_32F);
  for (int i = -r; i <= r; i++)
  {
    krn2.at<float>(0, i + r) = krn.at<float>(i + r, 0);
  }
  tmp.convertTo(tmp, CV_8UC1);

  conv(tmp, krn2, out, stride);
}
void dxSobel(cv::Mat &krn)
{
  krn = cv::Mat(3, 3, CV_32F);
  krn.at<float>(0, 0) = 1;
  krn.at<float>(0, 1) = 0;
  krn.at<float>(0, 2) = -1;
  krn.at<float>(1, 0) = 2;
  krn.at<float>(1, 1) = 0;
  krn.at<float>(1, 2) = -2;
  krn.at<float>(2, 0) = 1;
  krn.at<float>(2, 1) = 0;
  krn.at<float>(2, 2) = -1;
}
void dySobel(cv::Mat &krn)
{
  krn = cv::Mat(3, 3, CV_32F);
  krn.at<float>(0, 0) = 1;
  krn.at<float>(0, 1) = 2;
  krn.at<float>(0, 2) = 1;
  krn.at<float>(1, 0) = 0;
  krn.at<float>(1, 1) = 0;
  krn.at<float>(1, 2) = 0;
  krn.at<float>(2, 0) = -1;
  krn.at<float>(2, 1) = -2;
  krn.at<float>(2, 2) = -1;
}
void sobel3x3(const cv::Mat &src, cv::Mat &magn, cv::Mat &orient)
{
  magn = cv::Mat(src.rows, src.cols, CV_32F);
  orient = cv::Mat(src.rows, src.cols, CV_32F);

  cv::Mat ix;
  cv::Mat iy;
  cv::Mat krn;

  dxSobel(krn);
  conv(src, krn, ix, 1);
  dySobel(krn);
  conv(src, krn, iy, 1);

  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      orient.at<float>(i, j) = atan2(iy.at<int>(i, j), ix.at<int>(i, j));
      float magnitude = sqrt(ix.at<int>(i, j) * ix.at<int>(i, j) + iy.at<int>(i, j) * iy.at<int>(i, j));
      magn.at<float>(i, j) = magnitude;
    }
  }
}
template <typename T>
float bilinear(const cv::Mat &src, float r, float c)
{
  if (r < 0)
  {
    r = 0;
  }
  if (c < 0)
  {
    c = 0;
  }
  if (r > src.rows - 1)
  {
    r = src.rows - 1;
  }
  if (c > src.cols - 1)
  {
    c = src.cols - 1;
  }

  float yDist = r - (int)r;
  float xDist = c - (int)c;

  int value = src.at<T>(r, c) * (1 - yDist) * (1 - xDist);
  if (r < src.cols - 1)
    value += src.at<T>(r + 1, c) * (yDist) * (1 - xDist);
  if (c < src.cols - 1)
    value += src.at<T>(r, c + 1) * (1 - yDist) * (xDist);
  if (r < src.rows - 1 && c < src.cols - 1)
    value += src.at<T>(r + 1, c + 1) * yDist * xDist;

  return value;
}
void createBilateralKernel(const cv::Mat src, cv::Mat &krn, int d, float sigmaR, float sigmaD, int i, int j)
{
  krn = cv::Mat(d, d, CV_32F);
  int paddingR = d / 2;
  int paddingC = d / 2;
  float sum = 0;
  for (int yK = -paddingR; yK <= paddingR; yK++)
  {
    for (int xK = -paddingC; xK <= paddingC; xK++)
    {
      float range = exp(-(src.at<unsigned char>(i, j) - src.at<unsigned char>(i + yK, j + xK)) / (2 * sigmaR * sigmaR));
      float domain = exp(-(yK * yK + xK + xK) / (2 * sigmaD * sigmaD));
      krn.at<float>(paddingR + yK, paddingC + xK) = range * domain;
      sum += range * domain;
    }
  }
  krn /= sum;
}
void bilateralFilter(const cv::Mat src, cv::Mat &out, int d, float sigmaR, float sigmaD)
{
  out = cv::Mat(src.rows, src.cols, CV_32SC1);
  cv::Mat image;
  int paddingR = d / 2;
  int paddingC = d / 2;
  cv::Mat krn;
  addPadding(src, image, paddingR, paddingC);
  for (int i = paddingR; i < image.rows - paddingR; i++)
  {
    for (int j = paddingC; j < image.cols - paddingC; j++)
    {
      createBilateralKernel(src, krn, d, sigmaR, sigmaD, i, j);
      int sumKernel = 0;
      for (int yK = -paddingR; yK <= paddingR; yK++)
      {
        for (int xK = -paddingC; xK <= paddingC; xK++)
        {
          sumKernel += krn.at<float>(yK + paddingR, xK + paddingC) * image.at<unsigned char>(i + yK, j + xK);
        }
      }
      out.at<int>(i - paddingR, j - paddingC) = sumKernel;
    }
  }
}
int findPeaks(const cv::Mat &magn, const cv::Mat &orient, cv::Mat &out)
{
  out = cv::Mat::zeros(magn.rows, magn.cols, CV_32F);
  for (int i = 0; i < magn.rows; i++)
  {
    for (int j = 0; j < magn.cols; j++)
    {
      float value = magn.at<float>(i, j);
      float theta = orient.at<float>(i, j);
      float e1x = j + cos(theta);
      float e1y = i + sin(theta);
      float e2x = j - cos(theta);
      float e2y = i - sin(theta);
      float v1 = bilinear<float>(magn, e1y, e1x);
      float v2 = bilinear<float>(magn, e2y, e2x);
      if (value > v1 && value > v2)
      {
        out.at<float>(i, j) = value;
      }
    }
  }
}
typedef struct point_tag
{
  int i;
  int j;
  point_tag(int x, int y)
  {
    i = x;
    j = y;
  }
} point;

int doubleTh(const cv::Mat &magn, cv::Mat &out, float t1, float t2)
{
  std::queue<point> whitePoints;
  out = cv::Mat::zeros(magn.rows, magn.cols, CV_8UC1);
  for (int i = 0; i < magn.rows; i++)
  {
    for (int j = 0; j < magn.cols; j++)
    {
      float value = magn.at<float>(i, j);
      if (value >= t2)
      {
        out.at<unsigned char>(i, j) = 255;
        whitePoints.push(point(i, j));
      }
      else if (value >= t1)
      {
        out.at<unsigned char>(i, j) = 127;
      }
    }
  }

  while (whitePoints.size() != 0)
  {
    point p = whitePoints.front();
    whitePoints.pop();
    for (int y = -1; y <= 1; y++)
    {
      for (int x = -1; x <= 1; x++)
      {
        if (p.i + y >= 0 && p.i + y < out.rows && p.j + x >= 0 && p.j + x < out.cols)
        {
          if (out.at<unsigned char>(p.i + y, p.j + x) == 127)
          {
            out.at<unsigned char>(p.i + y, p.j + x) = 255;
            whitePoints.push(point(p.i + y, p.j + x));
          }
        }
      }
    }
  }

  for (int i = 0; i < magn.rows; i++)
  {
    for (int j = 0; j < magn.cols; j++)
    {
      if (out.at<unsigned char>(i, j) < 255)
      {
        out.at<unsigned char>(i, j) = 0;
      }
    }
  }
}

int main(int argc, char **argv)
{
  std::cout << "Simple program." << std::endl;

  //////////////////////
  //parse argument list:
  //////////////////////
  ArgumentList args;
  if (!ParseInputs(args, argc, argv))
  {
    return 1;
  }

  //opening file
  cv::Mat image = cv::imread(args.image_name, CV_8UC1);
  if (image.empty())
  {
    std::cout << "Unable to open " << args.image_name << std::endl;
    return 1;
  }

  //////////////////////
  //processing code here
  cv::Mat gaussImage;
  bilateralFilter(image, gaussImage, 3, 3, 3);

  gaussImage.convertTo(gaussImage, CV_8UC1);
  cv::Mat magn;
  cv::Mat orient;
  cv::Mat newMagn;
  sobel3x3(gaussImage, magn, orient);
  findPeaks(magn, orient, newMagn);
  cv::Mat out;
  doubleTh(newMagn, out, 20, 200);
  /////////////////////

  //display image
  cv::namedWindow("image", cv::WINDOW_NORMAL);
  cv::imshow("image", image);
  cv::namedWindow("out", cv::WINDOW_NORMAL);
  cv::imshow("out", out);
  //wait for key or timeout
  unsigned char key = cv::waitKey(args.wait_t);
  std::cout << "key " << int(key) << std::endl;

  return 0;
}
