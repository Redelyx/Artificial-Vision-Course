#include "Functions.h"

unsigned char openandwait(const char *windowname, cv::Mat &img, const bool sera=true){
    cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowname, img);
    std::cout << "Image size: " << img.rows << "x" << img.cols << std::endl;
    unsigned char key=cv::waitKey();
    if(key=='q')
        exit(EXIT_SUCCESS);
    if(sera)
        cv::destroyWindow(windowname);
    return key; 
}

void downsampling2x(cv::Mat &image){
    cv::Mat out(image.rows/2, image.cols/2, image.type());
    for(int r = 0; r<out.rows; r++){
        for(int c = 0; c<out.cols; c++){
            for(int k = 0; k<out.channels(); k++)
                out.data[(c + r*out.cols)*out.channels() + k] = image.data[(c*2 + r*2*image.cols)*image.channels() + k]; 
        }
    }
      
    openandwait("downsample 2x", out);
}