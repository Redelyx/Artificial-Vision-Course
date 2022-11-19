/*Cipriani Alice mat.340403*/
#include "Functions.h"

cv::Mat gaussianKrnl(float sigma, int r)
{
    cv::Mat krnl = cv::Mat(2 * r + 1, 1, CV_32F);
    float sum = 0;
    for(int i = -r; i < r+1; i++){
        float value = 1/sqrt(2*CV_PI*sigma*sigma)*exp(-pow(i,2)/(2*pow(sigma, 2)));
        sum+=value;
        krnl.at<float>(i+r, 0) = value;
    }
    krnl /= sum;
    for(int i = 0; i<krnl.rows; i++){
        std::cout<< krnl.at<float>(i,0) << " ";
    }
    std::cout << std::endl;
    return krnl;
}

cv::Mat gaussianBlur(const cv::Mat& src, float sigma, int r, int stride = 1)
{
    cv::Mat krnl = gaussianKrnl(sigma, r);
    cv::Mat tmp = conv(src, krnl, stride);
    tmp.convertTo(tmp, CV_8UC1);
    cv::Mat out = conv(tmp, transpose(krnl), stride); 
    out.convertTo(out, CV_8UC1);
    
    return out;
}

void sobel3x3(const cv::Mat& src)
{
    cv::Mat g, gx, gy, agx, agy, ag;
    cv::Sobel(src, gx, CV_32F, 1, 0, 3);       //applica direttamente Sobel
    cv::Sobel(src, gy, CV_32F, 0, 1, 3);

    // compute magnitude
    cv::pow(gx.mul(gx) + gy.mul(gy), 0.5, g);
    // compute orientation
    cv::Mat orient(gx.size(), CV_32FC1); 
    float *dest = (float *)orient.data;
    float *srcx = (float *)gx.data;
    float *srcy = (float *)gy.data;
    float *magn = (float *)g.data;
    for(int i=0; i<gx.rows*gx.cols; ++i)
      dest[i] = atan2f(srcy[i], srcx[i]) + 2*CV_PI;
    // scale on 0-255 range
    cv::convertScaleAbs(gx, agx);
    cv::convertScaleAbs(gy, agy);
    cv::convertScaleAbs(g, ag);
    cv::namedWindow("sobel verticale", cv::WINDOW_NORMAL);
    cv::imshow("sobel verticale", agx);
    cv::namedWindow("sobel orizzontale", cv::WINDOW_NORMAL);
    cv::imshow("sobel orizzontale", agy);
    cv::namedWindow("sobel magnitude", cv::WINDOW_NORMAL);
    cv::imshow("sobel magnitude", ag);    

    // trick to display orientation
    cv::Mat adjMap;
    cv::convertScaleAbs(orient, adjMap, 255 / (2*CV_PI));
    cv::Mat falseColorsMap;
    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_JET);
    cv::namedWindow("sobel orientation", cv::WINDOW_NORMAL);
    cv::imshow("sobel orientation", falseColorsMap);    
}

float bilinear(const cv::Mat& src, float r, float c)
{
    float result = 0;



    return result;
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

int lab4(ArgumentList args){
    std::cout << "----- Esercitazione 4 Edges -----" << std::endl;

    char frame_name[256];
    bool exit_loop = false;
    unsigned char key;
    
    float sigma = 0.8;
    int radius = 2;

    while (!exit_loop) {
        sprintf(frame_name, "%s", args.image_name.c_str());
        
        std::cout << "Opening " << frame_name << std::endl;

        cv::Mat image;

        image = cv::imread(frame_name, CV_8U);

        if (image.empty()) {
            std::cout << "Unable to open " << frame_name << std::endl;
            return 1;
        }

        openandwait("Original Image", image, false);

        //implementazione
        matTypeCV(image);

        cv::Mat out = gaussianBlur(image, sigma, radius);

        openandwait("out", out, false);
        sobel3x3(image);


        //key management
        unsigned char key = cv::waitKey(args.m_wait);
        std::cout << "key " << int(key) << std::endl;

        // here you can implement some looping logic using key value:
        //  - pause
        //  - stop
        //  - step back
        //  - step forward
        //  - loop on the same frame
        if (key == 's')
            sigma += 0.1;

        std::cout<< "sigma = " << sigma <<std::endl;

        if (key == 'r')
            radius += 1;

        std::cout<< "radius = " << radius <<std::endl;

        if (key == 'q')
            exit_loop = true;

    }

    return 0;
}
int main(int argc, char** argv)
{
    
    ArgumentList args;
    if (!ParseInputs(args, argc, argv)) {
        return 1;
    }


/*  
    char frame_name[256];
    bool exit_loop = false;
    unsigned char key;
    
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
            image = cv::imread(frame_name, CV_8U);

        if (image.empty()) {
            std::cout << "Unable to open " << frame_name << std::endl;
            return 1;
        }

        // display image
        openandwait("Original Image", image, false);
        
        //___implementazione

        sample(image, image_name);

        //___fine implementazione
    
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
    } */
    
    
    return lab2(args);
}
