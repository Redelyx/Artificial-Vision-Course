#include "Functions.h"

//---Esercizi Completi---

int lab1a(ArgumentList args, int argc, char** argv)
{
  char frame_name[256];
  bool exit_loop = false;
  int ksize = 3;
  int stride = 1;

  while(!exit_loop)
  {
    sprintf(frame_name,"%s",args.image_name.c_str());

    //opening file
    std::cout<<"Opening "<<frame_name<<std::endl;

    cv::Mat image = cv::imread(frame_name);
    if(image.empty())
    {
      std::cout<<"Unable to open "<<frame_name<<std::endl;
      return 1;
    }

    //display image
    cv::namedWindow("original image", cv::WINDOW_NORMAL);
    cv::imshow("original image", image);


    // PROCESSING

    // convert to grey scale for following processings
    cv::Mat grey;

    cv::cvtColor(image, grey, cv::COLOR_BGR2GRAY);
    cv::namedWindow("grey", cv::WINDOW_NORMAL);
    cv::imshow("grey", grey);

    cv::Mat blurred;

    // BOX FILTERING
    // void cv::boxFilter(InputArray src, OutputArray dst, int ddepth, Size ksize, Point anchor = Point(-1,-1), bool normalize = true, int borderType = BORDER_DEFAULT )
    cv::boxFilter(grey, blurred, CV_8U, cv::Size(ksize, ksize));
    cv::namedWindow("Box filter Smoothing", cv::WINDOW_NORMAL);
    cv::imshow("Box filter Smoothing", blurred);

    cv::Mat custom_kernel(ksize, ksize, CV_32FC1, 1.0/(ksize*ksize));
    // also possible as cv::Mat custom_kernel = 1.0/(ksize*ksize) * cv::Mat::ones(ksize, ksize, CV_32FC1);
    cv::Mat custom_blurred;
    cv::filter2D(grey, custom_blurred, CV_32F, custom_kernel);
    cv::convertScaleAbs(custom_blurred, custom_blurred);
    cv::namedWindow("Box filter Smoothing (custom)", cv::WINDOW_NORMAL);
    cv::imshow("Box filter Smoothing (custom)", custom_blurred);
    blurred.copyTo(grey);

    // SOBEL FILTERING
    // void cv::Sobel(InputArray src, OutputArray dst, int ddepth, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)

    cv::Mat g, gx, gy, agx, agy, ag;
    cv::Sobel(grey, gx, CV_32F, 1, 0, 3);
    cv::Sobel(grey, gy, CV_32F, 0, 1, 3);
    // compute magnitude
    cv::pow(gx.mul(gx) + gy.mul(gy), 0.5, g);
    // compute orientation
    cv::Mat orientation(gx.size(), CV_32FC1); 
    float *dest = (float *)orientation.data;
    float *srcx = (float *)gx.data;
    float *srcy = (float *)gy.data;
    float *magn = (float *)g.data;
    for(int i=0; i<gx.rows*gx.cols; ++i)
      dest[i] = magn[i]>50 ? atan2f(srcy[i], srcx[i]) + 2*CV_PI: 0;
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
    cv::convertScaleAbs(orientation, adjMap, 255 / (2*CV_PI));
    cv::Mat falseColorsMap;
    cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_JET);
    cv::namedWindow("sobel orientation", cv::WINDOW_NORMAL);
    cv::imshow("sobel orientation", falseColorsMap);    



    //wait for key or timeout
    unsigned char key = cv::waitKey(args.m_wait);
    std::cout<<"key "<<int(key)<<std::endl;

    //here you can implement some looping logic using key value:
    // - pause
    // - stop
    // - step back
    // - step forward
    // - loop on the same frame


    switch(key)
    {
      case 's':
	if(stride != 1)
	  --stride;
	std::cout << "Stride: " << stride << std::endl;
	break;
      case 'S':
	++stride;
	std::cout << "Stride: " << stride << std::endl;
	break;

      case 'c':
	cv::destroyAllWindows();
	break;
      case 'p':
	std::cout << "Mat = "<< std::endl << image << std::endl;
	break;
      case 'k':
	{
	  static int sindex=0;
	  int values[]={3, 5, 7, 11 ,13};
	  ksize = values[++sindex%5];
	  std::cout << "Setting Kernel size to: " << ksize << std::endl;
	}
	break;
      case 'g':
	break;
      case 'q':
	exit(0);
	break;
    }

  }

  return 0;
   
}

int lab1(ArgumentList args)
{
    char frame_name[256];
    bool exit_loop = false;
    unsigned char key;
    
    while (!exit_loop) {
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

        sample(image, frame_name);
        openandwait("Original Image", image, false);

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

    }
}

int lab2(ArgumentList args)
{
    /*  ./simple -i images/Candela/Candela_m1.10_%06d.pgm  */
    int frame_number = 0;
    char frame_name[256];
    bool exit_loop = false;
    unsigned char key;

    int threshold = 15;
    int k = 5;
    float alpha = 0.5;
    std::vector<cv::Mat> frames(k);

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


        if (frame_number == 0) {
                for(int i = 0; i<frames.size(); i++){
                        frames[i] = cv::Mat(image.rows, image.cols, image.type(), cv::Scalar(0));
                }
        }
        cv::Mat out, out1;
        out = simpleBgSubtraction(image, threshold);
        openandwait("simpleSub", out, false);


        out1 = expRunningAverageBgSubtraction(image, threshold, alpha);
        openandwait("expRunnAve", out1, false);

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

int lab3 (ArgumentList args){

    char frame_name[256];
    bool exit_loop = false;
    unsigned char key;

    int th;
    std::cout << "Insert threshold: ";
    std::cin >> th;

    while (!exit_loop) {
       
        sprintf(frame_name, "%s", args.image_name.c_str());

        // opening file
        std::cout << "Opening " << frame_name << std::endl;

        cv::Mat image;

        image = cv::imread(frame_name, CV_8UC1);
       
        if (image.empty()) {
            std::cout << "Unable to open " << frame_name << std::endl;
            return 1;
        }

        // display image
        openandwait("Original Image", image, false);

        cv::Mat out;
        out = simpleBin(image, th);
    
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

    } 
}
//---UTILITY---

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

std::string type2str(int type)
{
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

void matTypeCV(const cv::Mat M)
{
    std::string ty =  type2str(M.type());
    printf("Matrix: %s %dx%d \n", ty.c_str(), M.cols, M.rows);
}

cv::Mat transpose(const cv::Mat image)
{
    cv::Mat img(image.cols, image.rows, image.type()); // only part of A

    for (int r = 0; r < img.rows; r++)
    {
        for (int c = 0; c < img.cols; c++)
        {
            for (int k = 0; k < img.channels(); k++)
            {
                img.data[(c + r * img.cols) * img.channels() + k] = image.data[(r + c * img.cols) * img.channels() + k];
            }
        }
    }
    return img;
}

unsigned char openandwait(const char *windowname, cv::Mat &img, const bool sera = true)
{
    cv::namedWindow(windowname, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowname, img);
    std::cout << "Image size: " << img.rows << "x" << img.cols << std::endl;
    unsigned char key = cv::waitKey();
    if (key == 'q')
        exit(EXIT_SUCCESS);
    if (sera)
        cv::destroyWindow(windowname);
    return key;
}

//---LAB 1---

cv::Mat downsampling2x(const cv::Mat image)
{
    cv::Mat out(image.rows / 2, image.cols / 2, image.type());
    for (int r = 0; r < out.rows; r++)
    {
        for (int c = 0; c < out.cols; c++)
        {
            for (int k = 0; k < out.channels(); k++)
                out.data[(c + r * out.cols) * out.channels() + k] = image.data[(c * 2 + r * 2 * image.cols) * image.channels() + k];
        }
    }
    return out;
}

cv::Mat downsampling2xVert(const cv::Mat image)
{
    cv::Mat outv(image.rows / 2, image.cols, image.type());
    for (int v = 0; v < outv.rows; v++)
    {
        for (int u = 0; u < outv.cols; u++)
        {
            for (int k = 0; k < outv.channels(); k++)
                outv.data[(u + v * outv.cols) * outv.channels() + k] = image.data[(u + v * 2 * image.cols) * image.channels() + k];
        }
    }
    return outv;
}

cv::Mat downsampling2xHoriz(const cv::Mat image)
{
    cv::Mat outh(image.rows, image.cols / 2, image.type());
    for (int v = 0; v < outh.rows; v++)
    {
        for (int u = 0; u < outh.cols; u++)
        {
            for (int k = 0; k < outh.channels(); k++)
                outh.data[(u + v * outh.cols) * outh.channels() + k] = image.data[(u * 2 + v * image.cols) * image.channels() + k];
        }
    }
    return outh;
}

cv::Mat flipHoriz(const cv::Mat image)
{
    cv::Mat outfh(image.rows, image.cols, image.type());
    int i = outfh.cols - 1;
    for (int v = 0; v < outfh.rows; v++)
    {
        for (int u = 0; u < outfh.cols; u++)
        {
            for (int k = 0; k < outfh.channels(); k++)
                outfh.data[(i - u + v * outfh.cols) * outfh.channels() + k] = image.data[(u + v * image.cols) * image.channels() + k];
        }
    }
    return outfh;
}

cv::Mat flipVert(const cv::Mat image)
{
    cv::Mat outfv(image.rows, image.cols, image.type());
    int i = outfv.rows - 1;
    for (int v = 0; v < outfv.rows; v++)
    {
        for (int u = 0; u < outfv.cols; u++)
        {
            for (int k = 0; k < outfv.channels(); k++)
                outfv.data[(u + (i - v) * outfv.cols) * outfv.channels() + k] = image.data[(u + v * image.cols) * image.channels() + k];
        }
    }
    return outfv;
}

cv::Mat crop(const cv::Mat image, int x, int y, int w, int h)
{
    cv::Mat crop(w, h, image.type());

    if (x + h < image.rows && y + w < image.cols)
    {

        for (int v = 0; v < crop.rows; v++)
        {
            for (int u = 0; u < crop.cols; u++)
            {
                for (int k = 0; k < image.channels(); k++)
                    crop.data[(u + v * crop.cols) * crop.channels() + k] = image.data[((x + u) + (y + v) * image.cols) * image.channels() + k];
            }
        }
    }
    return crop;
}

cv::Mat addPadding(const cv::Mat image, int vPadding, int hPadding)
{
    cv::Mat padded = cv::Mat(image.rows + 2 * vPadding, image.cols + 2 * hPadding, image.type(), cv::Scalar(0));

    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            for (int k = 0; k < image.channels(); k++)
                padded.data[(hPadding + c + (vPadding + r) * padded.cols) * padded.channels() + k] = image.data[(c + r * image.cols) * image.channels() + k];
        }
    }
    return padded;
}

cv::Mat splitIn4(const cv::Mat image)
{
    cv::Mat split(image.rows, image.cols, image.type());

    std::vector<std::vector<int>> tlv = {
        {0, 0}, {0, image.cols / 2}, {image.rows / 2, 0}, {image.rows / 2, image.cols / 2}};

    std::random_shuffle(tlv.begin(), tlv.end());

    for (int br = 0; br < 2; br++)
    {
        for (int bc = 0; bc < 2; bc++)
        {
            for (int r = 0; r < image.rows / 2; r++)
            {
                for (int c = 0; c < image.cols / 2; c++)
                {
                    for (int k = 0; k < split.channels(); k++)
                    {
                        // top-left position in the destination image
                        int dest_r = br * image.rows / 2;
                        int dest_c = bc * image.cols / 2;
                        // top-left position in the original image
                        int orig_r = tlv[br * 2 + bc][0];
                        int orig_c = tlv[br * 2 + bc][1];
                        split.data[((c + dest_c) + (r + dest_r) * image.cols) * split.channels() + k] = image.data[((c + orig_c) + (r + orig_r) * image.cols) * image.channels() + k];
                    }
                }
            }
        }
    }
    return split;
}

cv::Mat colorShuffle(const cv::Mat image)
{
    cv::Mat img(image.rows, image.cols, image.type()); // only part of A

    std::vector<int> channels = {0, 1, 2};
    std::random_shuffle(channels.begin(), channels.end());

    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            img.data[(c + r * img.cols) * img.channels()] = image.data[(c + r * image.cols) * image.channels() + channels[0]];
            img.data[(c + r * img.cols) * img.channels() + 1] = image.data[(c + r * image.cols) * image.channels() + channels[1]];
            img.data[(c + r * img.cols) * img.channels() + 2] = image.data[(c + r * image.cols) * image.channels() + channels[2]];
        }
    }
    return img;
}

void sample(const cv::Mat image, std::string image_name)
{
    // ese1_11 downsample, ese1_12 luminance, ese1_13 simple
    cv::Mat down(image.rows / 2, image.cols / 2, CV_8UC1);
    cv::Mat lum2(image.rows / 2, image.cols / 2, CV_8UC1);

    bool isRGGB = image_name.find("RGGB") != std::string::npos;
    bool isBGGR = image_name.find("BGGR") != std::string::npos;
    bool isGBRG = image_name.find("GBRG") != std::string::npos;

    if (isRGGB)
    {
        std::cout << "RGGB" << std::endl;
    }
    if (isBGGR)
    {
        std::cout << "BGGR" << std::endl;
    }
    if (isGBRG)
    {
        std::cout << "GBRG" << std::endl;
    }

    for (int r = 0; r < down.rows; r++)
    {
        for (int c = 0; c < down.cols; c++)
        {
            // original image pixel coordinates
            int orig_r = r * 2;
            int orig_c = c * 2;
            // channel position in the original image
            int up_left = image.data[orig_c + orig_r * image.cols];
            int up_right = image.data[orig_c + orig_r * image.cols + 1];
            int low_left = image.data[orig_c + (orig_r + 1) * image.cols];
            int low_right = image.data[orig_c + (orig_r + 1) * image.cols + 1];

            if (isRGGB)
            {
                down.data[c + r * down.cols] = (up_right + low_left) / 2;
                lum2.data[(r * down.cols + c)] = 0.3 * float(up_left) + 0.59 * float(up_right + low_left) / 2.0 + 0.11 * float(low_right);
            }
            if (isBGGR)
            {
                down.data[c + r * down.cols] = (up_right + low_left) / 2;
                lum2.data[(r * down.cols + c)] = 0.3 * float(low_right) + 0.59 * float(up_right + low_left) / 2.0 + 0.11 * float(up_left);
            }
            if (isGBRG)
            {
                down.data[c + r * down.cols] = (up_left + low_right) / 2;
                lum2.data[(r * down.cols + c)] = 0.3 * float(low_left) + 0.59 * float(up_left + low_right) / 2.0 + 0.11 * float(low_left);
            }
        }
    }

    openandwait("DOWNSAMPLE", down, false);
    openandwait("LUMINANCE2x", lum2, false);

    cv::Mat lum(image.rows, image.cols, CV_8UC1);
    cv::Mat simple(image.rows, image.cols, CV_8UC3);
    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            int up_left = image.data[c + r * image.cols];
            int up_right = image.data[c + r * image.cols + 1];
            int low_left = image.data[c + (r + 1) * image.cols];
            int low_right = image.data[c + (r + 1) * image.cols + 1];

            bool r_even = (r % 2 == 0);
            bool c_even = (c % 2 == 0);

            int blue;
            int green;
            int red;

            //////////
            // Pattern:
            //   R G
            //   G B
            //////////
            if ((isRGGB && r_even && c_even) || (isBGGR && !r_even && !c_even) || (isGBRG && !r_even && c_even))
            {
                blue = low_right;
                green = (low_left + up_right) / 2;
                red = up_left;
            }
            //////////
            // Pattern:
            //   B G
            //   G R
            //////////
            if ((isRGGB && !r_even && !c_even) || (isBGGR && r_even && c_even) || (isGBRG && r_even && !c_even))
            {
                blue = up_left;
                green = (low_left + up_right) / 2;
                red = low_right;
            }
            //////////
            // Pattern:
            //   G R
            //   B G
            //////////
            if ((isRGGB && r_even && !c_even) || (isBGGR && !r_even && c_even) || (isGBRG && !r_even && !c_even))
            {
                blue = low_left;
                green = (up_left + low_right) / 2;
                red = up_right;
            }
            //////////
            // Pattern:
            //   G B
            //   R G
            //////////
            if ((isRGGB && !r_even && c_even) || (isBGGR && r_even && !c_even) || (isGBRG && r_even && c_even))
            {
                blue = up_right;
                green = (up_left + low_right) / 2;
                red = low_left;
            }

            simple.data[(c + r * simple.cols) * simple.channels()] = blue;
            simple.data[(c + r * simple.cols) * simple.channels() + 1] = green;
            simple.data[(c + r * simple.cols) * simple.channels() + 2] = red;
            lum.data[c + r * lum.cols] = 0.3 * float(red) + 0.59 * float(green) + 0.11 * float(blue);
        }
    }
    openandwait("SIMPLE", simple, false);
    openandwait("LUMINANCE", lum, false);
}

//---LAB 1a---

cv::Mat conv(const cv::Mat &src, const cv::Mat &krn, int stride = 1)
{
    if (krn.cols % 2 == 0 || krn.rows % 2 == 0)
    {
        std::cerr << "ERROR: kernel has not odd size" << std::endl;
        exit(1);
    }
    cv::Mat out = cv::Mat(src.rows / stride, src.cols / stride, CV_32SC1);

    cv::Mat image = addPadding(src, krn.rows / 2, krn.cols / 2);

    // comode per ciclare nel kernel stesso
    int xc = krn.cols / 2;
    int yc = krn.rows / 2;

    // puntatore d'appoggio al buffer per l'uscita e per il kernel
    int *outbuffer = (int *)out.data;
    float *kernel = (float *)krn.data;

    // si cicla sempre sull'immagine destinazione

    for (int r = 0; r < out.rows; r++)
    {
        for (int c = 0; c < out.cols; c++)
        {
            // calcolo le coordinate nell'immagine originale
            int origr = r * stride + yc;
            int origc = c * stride + xc;

            // metto una variabile per calcolare il risultato di una singola sovrapposizione tra kernel e img
            float sum = 0;
            for (int kr = -yc; kr <= yc; kr++)
            {
                for (int kc = -xc; kc <= xc; kc++)
                {
                    sum += image.data[(origr + kr) * image.cols + (origc + kc)] * kernel[(kr + yc) * krn.cols + (kc + xc)];
                }
            }
            outbuffer[r * out.cols + c] = sum;
        }
    }
    return out;
}

//---LAB 2---
cv::Mat precBackground;
cv::Mat thisBackground;
cv::Mat nextBackground;

cv::Mat simpleBgSubtraction(const cv::Mat image, int threshold)
{
    if (precBackground.empty())
    {
        precBackground = cv::Mat(image.rows, image.cols, image.type(), cv::Scalar(0));
    }

    cv::Mat foreground = cv::Mat(image.rows, image.cols, CV_8UC1);
    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            int bitSubtraction = image.data[(c + r * image.cols) * image.elemSize()] - precBackground.data[(c + r * image.cols)];
            if (abs(bitSubtraction) > threshold)
            {
                foreground.data[(c + r * image.cols)] = image.data[(c + r * image.cols) * image.elemSize()];
            }
            else
            {
                foreground.data[(c + r * image.cols)] = 0;
            }
            precBackground.data[(c + r * image.cols)] = image.data[(c + r * image.cols) * image.elemSize()];
        }
    }

    return foreground;
}

cv::Mat runningAverageBgSubtraction(const cv::Mat image, std::vector<cv::Mat> &v, int threshold, int n)
{
    /*n is the number of the current frame*/
    int cursor = n % v.size();
    v[cursor] = image;

    cv::Mat foreground = cv::Mat(image.rows, image.cols, CV_8UC1);

    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            int sum = 0;
            for (int cur = 0; cur < v.size(); cur++)
            {
                sum += (int)v[cur].data[(c + r * image.cols) * image.elemSize()];
            }
            int bitSubtraction = image.data[(c + r * image.cols) * image.elemSize()] - sum / v.size(); // background.data[(c + r * image.cols)];
            if (abs(bitSubtraction) > threshold)
            {
                foreground.data[(c + r * image.cols)] = image.data[(c + r * image.cols) * image.elemSize()];
            }
            else
            {
                foreground.data[(c + r * image.cols)] = 0;
            }
        }
    }

    return foreground;
}

cv::Mat expRunningAverageBgSubtraction(const cv::Mat image, int threshold, float alpha)
{
    /*alpha exists between 0 and 1*/
    if (thisBackground.empty())
    {
        thisBackground = cv::Mat(image.rows, image.cols, image.type(), cv::Scalar(0));
        nextBackground = cv::Mat(image.rows, image.cols, image.type(), cv::Scalar(0));
    }
    cv::Mat foreground = cv::Mat(image.rows, image.cols, CV_8UC1);

    for (int r = 0; r < image.rows; r++)
    {
        for (int c = 0; c < image.cols; c++)
        {
            nextBackground.data[(c + r * image.cols)] = alpha * thisBackground.data[(c + r * image.cols)] + (1 - alpha) * image.data[(c + r * image.cols) * image.elemSize()];
            int bitSubtraction = image.data[(c + r * image.cols) * image.elemSize()] - thisBackground.data[(c + r * image.cols)];
            if (abs(bitSubtraction) > threshold)
            {
                foreground.data[(c + r * image.cols)] = image.data[(c + r * image.cols) * image.elemSize()];
            }
            else
            {
                foreground.data[(c + r * image.cols)] = 0;
            }
        }
    }

    thisBackground = nextBackground;

    return foreground;
}

//---LAB 3---

cv::Mat simpleBin(const cv::Mat &image, int threshold)
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


/* float bilinear(const cv::Mat &src, float r, float c)
{
    return 0;
}

int findPeaks(const cv::Mat &magn, const cv::Mat &src, cv::Mat &out)
{
    return 0;
}

int doubleTh(const cv::Mat &magn, cv::Mat &out, float t1, float t2)
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
 */

//---LAB 4---