#include "Functions.h"

//---UTILITY---

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

//---LAB 1---

cv::Mat downsampling2x(const cv::Mat image){
    cv::Mat out(image.rows/2, image.cols/2, image.type());
    for(int r = 0; r<out.rows; r++){
        for(int c = 0; c<out.cols; c++){
            for(int k = 0; k<out.channels(); k++)
                out.data[(c + r*out.cols)*out.channels() + k] = image.data[(c*2 + r*2*image.cols)*image.channels() + k]; 
        }
    }
    return out;
}

cv::Mat downsampling2xVert(const cv::Mat image){
    cv::Mat outv(image.rows/2, image.cols, image.type());
    for(int v = 0; v<outv.rows; v++){
        for(int u = 0; u<outv.cols; u++){
            for(int k = 0; k<outv.channels(); k++)
                outv.data[(u + v*outv.cols)*outv.channels() + k] = image.data[(u + v*2*image.cols)*image.channels() + k]; 
        }
    }
    return outv;
}

cv::Mat downsampling2xHoriz(const cv::Mat image){
    cv::Mat outh(image.rows, image.cols/2, image.type());
    for(int v = 0; v<outh.rows; v++){
        for(int u = 0; u<outh.cols; u++){
            for(int k = 0; k<outh.channels(); k++)
                outh.data[(u + v*outh.cols)*outh.channels() + k] = image.data[(u*2 + v*image.cols)*image.channels() + k]; 
        }
    }  
    return outh;
}

cv::Mat flipHoriz(const cv::Mat image){
    cv::Mat outfh(image.rows, image.cols, image.type());
    int i = outfh.cols-1;
    for(int v = 0; v<outfh.rows; v++){
        for(int u = 0; u<outfh.cols; u++){
            for(int k = 0; k<outfh.channels(); k++)
                outfh.data[(i-u + v*outfh.cols)*outfh.channels() + k] = image.data[(u + v*image.cols)*image.channels() + k]; 
        }
    }
    return outfh;
}

cv::Mat flipVert(const cv::Mat image){
    cv::Mat outfv(image.rows, image.cols, image.type());
        int i = outfv.rows-1;
        for(int v = 0; v<outfv.rows; v++){
            for(int u = 0; u<outfv.cols; u++){
                for(int k = 0; k<outfv.channels(); k++)
                    outfv.data[(u + (i-v)*outfv.cols)*outfv.channels() + k] = image.data[(u + v*image.cols)*image.channels() + k]; 
        }
    }
    return outfv;
}

cv::Mat crop(const cv::Mat image, int x, int y, int w, int h){
    cv::Mat crop(w, h, image.type());
    
    if(x+h < image.rows && y+w< image.cols){
    
        for(int v = 0; v<crop.rows; v++){
            for(int u = 0; u<crop.cols; u++){
                for(int k = 0; k<image.channels(); k++)
                    crop.data[(u + v*crop.cols)*crop.channels() + k] = image.data[((x+u) + (y+v)*image.cols)*image.channels() + k]; 
            }
        }
    }
    return crop;
}

cv::Mat addPadding(const cv::Mat image, int vPadding, int hPadding){
    cv::Mat padded = cv::Mat(image.rows+2*vPadding, image.cols+2*hPadding, image.type(), cv::Scalar(0));
      
    for(int r = 0; r<image.rows; r++){
        for(int c = 0; c<image.cols; c++){
            for(int k = 0; k<image.channels(); k++)
                padded.data[(hPadding+c + (vPadding+r)*padded.cols)*padded.channels() + k] = image.data[(c + r*image.cols)*image.channels() + k]; 
        }
    }
    return padded;
}

cv::Mat splitIn4(const cv::Mat image){
    cv::Mat split(image.rows, image.cols, image.type());
    
    std::vector<std::vector <int>> tlv = {
        {0,0}, {0, image.cols/2}, {image.rows/2, 0}, {image.rows/2, image.cols/2}
    };

    std::random_shuffle(tlv.begin(), tlv.end());

    for(int br = 0; br<2; br++){
        for(int bc = 0; bc<2; bc++){
            for(int r = 0; r<image.rows/2; r++){
                for(int c = 0; c<image.cols/2; c++){
                    for(int k = 0; k<split.channels(); k++){
                        //top-left position in the destination image
                        int dest_r = br*image.rows/2;
                        int dest_c = bc*image.cols/2;
                        //top-left position in the original image
                        int orig_r = tlv[br*2 + bc][0];
                        int orig_c = tlv[br*2 + bc][1];
                        split.data[((c+dest_c) + (r+dest_r)*image.cols)*split.channels() + k] = image.data[((c+orig_c) + (r+orig_r)*image.cols)*image.channels() + k]; 
                    }
                }
            }
        }
    }
    return split;
}

cv::Mat colorShuffle(const cv::Mat image){
    cv::Mat img(image.rows, image.cols, image.type()); // only part of A

    std::vector<int> channels = {0,1,2};
    std::random_shuffle(channels.begin(), channels.end());

    for(int r = 0; r<image.rows; r++){
        for(int c = 0; c<image.cols; c++){
            img.data[(c + r*img.cols)*img.channels()] = image.data[(c + r*image.cols)*image.channels() + channels[0]]; 
            img.data[(c + r*img.cols)*img.channels() + 1] = image.data[(c + r*image.cols)*image.channels() + channels[1]]; 
            img.data[(c + r*img.cols)*img.channels() + 2] = image.data[(c + r*image.cols)*image.channels() + channels[2]]; 
        }
    }
    return img;
}

void sample(const cv::Mat image, std::string image_name){
    //ese1_11 downsample, ese1_12 luminance, ese1_13 simple
    cv::Mat down(image.rows/2, image.cols/2, CV_8UC1);
    cv::Mat lum2(image.rows/2, image.cols/2, CV_8UC1);
    
    bool isRGGB = image_name.find("RGGB")!=std::string::npos;
    bool isBGGR = image_name.find("BGGR")!=std::string::npos;
    bool isGBRG = image_name.find("GBRG")!=std::string::npos;

    if(isRGGB){
        std::cout << "RGGB" << std::endl;
    }
    if(isBGGR){
        std::cout << "BGGR" << std::endl;
    }
    if(isGBRG){
        std::cout << "GBRG" << std::endl;       
    }

    for(int r = 0; r<down.rows; r++){
        for(int c = 0; c<down.cols; c++){
            //original image pixel coordinates
            int orig_r = r*2;
            int orig_c = c*2;
            //channel position in the original image
            int up_left = image.data[orig_c + orig_r*image.cols];
            int up_right = image.data[orig_c + orig_r*image.cols + 1];
            int low_left = image.data[orig_c + (orig_r + 1)*image.cols];
            int low_right = image.data[orig_c + (orig_r + 1)*image.cols + 1];

            if(isRGGB){
                down.data[c+r*down.cols] = (up_right+low_left)/2;
                lum2.data[(r*down.cols+c)] = 0.3*float(up_left) + 0.59*float(up_right+low_left)/2.0 + 0.11*float(low_right);
            }
            if(isBGGR){
                down.data[c+r*down.cols] = (up_right+low_left)/2;
                lum2.data[(r*down.cols+c)] = 0.3*float(low_right) + 0.59*float(up_right+low_left)/2.0 + 0.11*float(up_left);
            }
            if(isGBRG){
                down.data[c+r*down.cols] = (up_left+low_right)/2;
                lum2.data[(r*down.cols+c)] = 0.3*float(low_left) + 0.59*float(up_left+low_right)/2.0 + 0.11*float(low_left);
            }
        }
    }

    openandwait("DOWNSAMPLE", down, false);
    openandwait("LUMINANCE2x", lum2, false);

    cv::Mat lum(image.rows, image.cols, CV_8UC1);
    cv::Mat simple(image.rows, image.cols, CV_8UC3);
    for(int r = 0; r<image.rows; r++){
        for(int c = 0; c<image.cols; c++){
            int up_left = image.data[c + r*image.cols];
            int up_right = image.data[c + r*image.cols + 1];
            int low_left = image.data[c + (r + 1)*image.cols];
            int low_right = image.data[c + (r + 1)*image.cols + 1];

            bool r_even = (r%2 == 0);
            bool c_even = (c%2 == 0);

            int blue;
            int green;
            int red;

            //////////
            //Pattern:
            //  R G
            //  G B
            //////////
            if((isRGGB && r_even && c_even) || (isBGGR && !r_even && !c_even) || (isGBRG && !r_even && c_even)){
                blue = low_right;
                green = (low_left+up_right)/2;
                red = up_left; 
            }
            //////////
            //Pattern:
            //  B G
            //  G R
            //////////
            if((isRGGB && !r_even && !c_even) || (isBGGR && r_even && c_even) || (isGBRG && r_even && !c_even)){
                blue = up_left;
                green = (low_left+up_right)/2;
                red = low_right; 
            }
            //////////
            //Pattern:
            //  G R
            //  B G
            //////////
            if((isRGGB && r_even && !c_even) || (isBGGR && !r_even && c_even) || (isGBRG && !r_even && !c_even)){
                blue = low_left;
                green = (up_left+low_right)/2;
                red = up_right; 
            }
            //////////
            //Pattern:
            //  G B
            //  R G
            //////////
            if((isRGGB && !r_even && c_even) || (isBGGR && r_even && !c_even) || (isGBRG && r_even && c_even)){
                blue = up_right;
                green = (up_left+low_right)/2;
                red = low_left; 
            }

            simple.data[(c + r*simple.cols)*simple.channels()] = blue;
            simple.data[(c + r*simple.cols)*simple.channels() + 1] = green;
            simple.data[(c + r*simple.cols)*simple.channels() + 2] = red; 
            lum.data[c + r*lum.cols] = 0.3*float(red) + 0.59*float(green) + 0.11*float(blue);
        }
    }
    openandwait("SIMPLE", simple, false);
    openandwait("LUMINANCE", lum, false);    
    
}

//---LAB 1a---

cv::Mat myfilter2D(const cv::Mat src, const cv::Mat& krn, cv::Mat& out,  int stride=1){
    if(krn.cols%2 == 0 || krn.rows%2 == 0){
        std::cerr << "ERROR: kernel has not odd size" << std::endl;
        exit(1);
    }
    out = cv::Mat(src.rows/stride, src.cols/stride, CV_32SC1);

    cv::Mat image = addPadding(src, krn.rows/2, krn.cols/2);
    
    //comode per ciclare nel kernel stesso
    int xc = krn.cols/2;
    int yc = krn.rows/2;

    //puntatore d'appoggio al buffer per l'uscita e per il kernel
    int *outbuffer = (int *) out.data;
    float *kernel = (float *) krn.data;

    //si cicla sempre sull'immagine destinazione

    for(int r = 0; r < out.rows; r++){
        for(int c = 0 ; c < out.cols; c++){
            //calcolo le coordinate nell'immagine originale
            int origr = r*stride + yc;
            int origc = c*stride + xc;

            //metto una variabile per calcolare il risultato di una singola sovrapposizione tra kernel e img
            float sum = 0;
            for(int kr = -yc; kr <= yc; kr++){
                for(int kc = -xc; kc <= xc; kc++){
                    sum += image.data[(origr+kr)*image.cols + (origc+kc)] * kernel[(kr+yc)*krn.cols + (kc+xc)];
                }
            }
            outbuffer[r*out.cols+c]=sum;
        }
    }
    return out;
}

//---LAB 3---

//---LAB 4---