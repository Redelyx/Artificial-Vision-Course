#include "Functions.h"

struct ArgumentList {
  std::string image_name;		    //!< image file name
  int m_wait;                     //!< waiting time
  int m_padding;
  int u_tl;
  int v_tl;
  int width_crop;
  int height_crop;
  unsigned int ex;
  bool random_crop;
};

void gaussianKrnl(float sigma, int r, cv::Mat& krnl){

}

void GaussianBlur(const cv::Mat& src, float sigma, int r, cv::Mat& out, int
stride=1){

}

void verticalSobel(cv::Mat &krn)
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

void horizontalSobel(cv::Mat &krn)
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

void sobel3x3(const cv::Mat& src, cv::Mat& magn, cv::Mat& src1){

}

float bilinear(const cv::Mat& src, float r, float c){
    return 0;
}

int findPeaks(const cv::Mat& magn, const cv::Mat& src, cv::Mat& out){
    return 0;
}

int doubleTh(const cv::Mat& magn, cv::Mat& out, float t1, float t2){

}

bool ParseInputs(ArgumentList& args, int argc, char **argv) {
  int c;
  args.m_wait=0;
  args.m_padding=1;
  args.u_tl=100;
  args.v_tl=100;
  args.width_crop=100;
  args.height_crop=100;
  args.random_crop=false;
  args.ex=0xFFFFFFFFu;

  while ((c = getopt (argc, argv, "hi:t:p:u:v:W:H:rx:")) != -1)
    switch (c)
    {
      case 'i':
	args.image_name = optarg;
	break;
      case 'p':
	args.m_padding = atoi(optarg);
	break;
      case 'u':
	args.u_tl = atoi(optarg);
	break;
      case 'v':
	args.v_tl = atoi(optarg);
	break;
      case 'H':
	args.height_crop = atoi(optarg);
	break;
      case 'W':
	args.width_crop = atoi(optarg);
	break;
      case 'r':
	args.random_crop = true;
	break;
      case 'x':
	args.ex = (1<<atoi(optarg));
	break;
      case 't':
	args.m_wait = atoi(optarg);
	break;
      case 'h':
      default:
	std::cout<<"usage: " << argv[0] << " -i <image_name>"<<std::endl;
	std::cout<<"exit:  type q"<<std::endl<<std::endl;
	std::cout<<"Allowed options:"<<std::endl<<
	  "   -h                       produce help message"<<std::endl<<
	  "   -i arg                   image name. Use %0xd format for multiple images."<<std::endl<<
	  "   -p arg                   padding size (default 1)"<<std::endl<<
	  "   -u arg                   crop column (default 100)"<<std::endl<<
	  "   -v arg                   crop row (default 100)"<<std::endl<<
	  "   -W arg                   crop width (default 100)"<<std::endl<<
	  "   -H arg                   crop height (default 100)"<<std::endl<<
	  "   -r                       random crop"<<std::endl<<
	  "   -x arg                   exercises (default all)"<<std::endl<<
	  "   -t arg                   wait before next frame (ms) [default = 0]"<<std::endl<<std::endl<<std::endl;
	return false;
    }
  return true;
}

int main(int argc, char **argv){
    std::cout << "----- Esercitazione 4 Edges -----" << std::endl;
    
    int frame_number = 0;
    char frame_name[256];
    bool exit_loop = false;
    int ksize = 3;
    int stride = 1;
    unsigned char key;
    ArgumentList args;
    
    //////////////////////
    //parse argument list:
    //////////////////////


    if(!ParseInputs(args, argc, argv)){
        return 1;
    }

    while(!exit_loop){
        //generating file name
        //
        //multi frame case
        if(args.image_name.find('%') != std::string::npos)
            sprintf(frame_name,(const char*)(args.image_name.c_str()),frame_number);
        else //single frame case
            sprintf(frame_name,"%s",args.image_name.c_str());

        //opening file
        std::cout << "Opening " << frame_name << std::endl;

        cv::Mat image;
        if(args.image_name.find("RGGB")!=std::string::npos || args.image_name.find("GBRG")!=std::string::npos || args.image_name.find("BGGR")!=std::string::npos)
            image = cv::imread(frame_name,CV_8UC1);
        else
            image = cv::imread(frame_name);

        if(image.empty())
        {
            std::cout<<"Unable to open "<<frame_name<<std::endl;
            return 1;
        }

        //display image
        openandwait("Original Image", image, false);

        //cv::Mat new_m = downsampling2x(image);
        //openandwait("not Image", new_m, false);

        sample(image, args.image_name);


    }
        return 0;
}
