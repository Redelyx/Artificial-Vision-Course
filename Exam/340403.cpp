// OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <fstream>
#include <iostream>
#include <string>

//#define DEBUG

void addPadding(const cv::Mat& src, cv::Mat& padded, int vPadding, int hPadding, int color = 0)
{
    padded = cv::Mat(src.rows + 2 * vPadding, src.cols + 2 * hPadding, src.type(), cv::Scalar(color));
    for (int r = 0; r < src.rows; r++) {
        for (int c = 0; c < src.cols; c++) {
            for (int k = 0; k < src.channels(); k++)
                padded.data[(hPadding + c + (vPadding + r) * padded.cols) * padded.channels() + k] = src.data[(c + r * src.cols) * src.channels() + k];
        }
    }
}

void conv(const cv::Mat& src, const cv::Mat& krn, cv::Mat& out, int stride = 1)
{
    if (krn.cols % 2 == 0 || krn.rows % 2 == 0) {
        std::cerr << "ERROR: kernel has not odd size" << std::endl;
        exit(1);
    }
    out = cv::Mat(src.rows / stride, src.cols / stride, CV_32SC1);

    cv::Mat image;

    addPadding(src, image, krn.rows / 2, krn.cols / 2);

    // comode per ciclare nel kernel stesso
    int xc = krn.cols / 2;
    int yc = krn.rows / 2;

    // puntatore d'appoggio al buffer per l'uscita e per il kernel
    int* outbuffer = (int*)out.data;
    float* kernel = (float*)krn.data;

    // si cicla sempre sull'immagine destinazione
    for (int r = 0; r < out.rows; r++) {
        for (int c = 0; c < out.cols; c++) {
            // calcolo le coordinate nell'immagine originale
            int origr = r * stride + yc;
            int origc = c * stride + xc;

            // metto una variabile per calcolare il risultato di una singola sovrapposizione tra kernel e img
            float sum = 0;
            for (int kr = -yc; kr <= yc; kr++) {
                for (int kc = -xc; kc <= xc; kc++) {
                    sum += image.data[(origr + kr) * image.cols + (origc + kc)] * kernel[(kr + yc) * krn.cols + (kc + xc)];
                }
            }
            outbuffer[r * out.cols + c] = sum;
        }
    }
}
void gaussianKernel(cv::Mat& kernel, int rows, int cols, float sigma)
{
    kernel = cv::Mat(rows, cols, CV_32FC1);
    int x_center = cols / 2;
    int y_center = rows / 2;
    float sum = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float point = pow((j - x_center), 2) + pow((i - y_center), 2);
            float gaussian = exp(-point / (2 * pow(sigma, 2)));
            kernel.at<float>(i, j) = gaussian;
            sum += gaussian;
        }
    }
    kernel /= sum;
}
bool computeNMS(cv::Mat input, int row, int col){
	return ( input.at<float>(row,col) > input.at<float>(row-1,col-1)   &&
           input.at<float>(row,col) > input.at<float>(row-1,col+1)   &&
           input.at<float>(row,col) > input.at<float>(row,col+1)     &&
           input.at<float>(row,col) > input.at<float>(row,col-1)     &&
           input.at<float>(row,col) > input.at<float>(row+1,col)     &&
           input.at<float>(row,col) > input.at<float>(row+1,col-1)   &&
           input.at<float>(row,col) > input.at<float>(row-1,col)     &&
           input.at<float>(row,col) > input.at<float>(row+1,col+1));
}
void floatConv(const cv::Mat input, const cv::Mat& kernel, cv::Mat& out){
	out=cv::Mat(input.rows,input.cols,CV_32FC1, cv::Scalar(0));
	float* out_data=(float *)out.data;
	float* kernel_data=(float *)kernel.data;
	float* input_data=(float *)input.data;
	for(int v=kernel.rows/2;v<out.rows-kernel.rows/2;v++){
		for(int u=kernel.cols/2;u<out.cols-kernel.cols/2;u++){
			float somma=0;
			for(int k_v=0;k_v<kernel.rows;k_v++){
				for(int k_u=0;k_u<kernel.cols;k_u++){
					somma+=((float)input_data[((u-kernel.cols/2+k_u)+(v-kernel.rows/2+k_v)*input.cols)]*kernel_data[k_u+k_v*kernel.cols]);
				}
			}
			out_data[u+v*out.cols]=somma;
		}
	}
}

struct ArgumentList {
    std::string image_name; //!< image file name
    int wait_t; //!< waiting time
    float alpha; //!< alpha parameter
    double th; //!< harris threshold
};

bool ParseInputs(ArgumentList& args, int argc, char** argv);

int main(int argc, char** argv)
{
    char frame_name[256];

    std::cout << "Simple program." << std::endl;

    //////////////////////
    // parse argument list:
    //////////////////////
    ArgumentList args;
    if (!ParseInputs(args, argc, argv)) {
        exit(0);
    }

    while (true) {
        sprintf(frame_name, "%s", args.image_name.c_str());

        // opening file
        std::cout << "Opening " << frame_name << std::endl;

        cv::Mat image = cv::imread(frame_name, cv::IMREAD_GRAYSCALE); // image to be processed
        cv::Mat image_color = cv::imread(frame_name, cv::IMREAD_COLOR); // only for display purposes
        if (image.empty()) {
            std::cout << "Unable to open " << frame_name << std::endl;
            return 1;
        }

        std::cout << "The image has " << image.channels() << " channels, the size is " << image.rows << "x" << image.cols << " pixels "
                  << " the type is " << image.type() << " the pixel size is " << image.elemSize() << " and each channel is " << image.elemSize1() << (image.elemSize1() > 1 ? " bytes" : " byte") << std::endl;

        // apply a little smoothing
        cv::GaussianBlur(image, image, cv::Size(5, 5), 0, 0);

        //////////////////////
        // processing code here
        cv::Mat d_Ix;
        cv::Mat d_Iy;
        cv::Mat Ix;
        cv::Mat Iy;
        cv::Mat blur_oriz;
        cv::Mat blur_bidim;
        cv::Mat kernel_gauss_oriz;
        cv::Mat t_x;
        cv::Mat t_y;
		cv::Mat magn;
		float harrisTh = 6000.0f;
		float alpha = 0.04;

        float gradiente_data[3] = { -1, 0, 1 };
        // 1. APPLY SOBEL AND COMPUTE Ix & Iy
        //    HINT: convert to CV_32F the results

		//sobel3x3(image, magn, image);
        cv::Mat gradiente_oriz = cv::Mat(1, 3, CV_32FC1, gradiente_data);
        conv(image, gradiente_oriz, Ix);
        Ix.convertTo(d_Ix, CV_32F);
		#ifdef DEBUG
		cv::imshow("Ix", d_Ix);
		#endif

        cv::Mat gradiente_vert = gradiente_oriz.t();
        conv(image, gradiente_vert, Iy);
        Iy.convertTo(d_Iy, CV_32F);
		#ifdef DEBUG
        cv::imshow("Iy", d_Iy);
		#endif

        // 2. COMPUTE A matrix elements
        //    Q: when do we need to apply a Gaussian filter?
        cv::Mat d_Ixy = cv::Mat(image.rows, image.cols, CV_32FC1);
        d_Ixy = d_Ix.mul(d_Iy);
		#ifdef DEBUG
        cv::imshow("Ixy", d_Ixy);
		#endif

        int raggio = 3;
        float sigma = 47.0f;

        // Calcolo g(I_x).
        gaussianKernel(kernel_gauss_oriz, raggio, raggio, sigma);

        // Calcolo g(I_y).
        cv::Mat kernel_gauss_vert = kernel_gauss_oriz.t();

        // Calcolo di g(I_x*I_y).
        floatConv(d_Ixy, kernel_gauss_oriz, blur_oriz);
        floatConv(blur_oriz, kernel_gauss_vert, blur_bidim);

        // Calcolo di d(I_x)^2 e d(I_y)^2
        cv::Mat d_Ix_2 = cv::Mat(d_Ix.rows, d_Ix.cols, d_Ix.type());
        cv::Mat d_Iy_2 = cv::Mat(d_Iy.rows, d_Iy.cols, d_Iy.type());
        d_Ix_2 = d_Ix.mul(d_Ix);
        d_Iy_2 = d_Iy.mul(d_Iy);

        cv::Mat g_d_Ix_2;
        cv::Mat g_d_Iy_2;

        // Calcolo di g(I_x^2)
        floatConv(d_Ix_2, kernel_gauss_oriz, t_x);
        floatConv(t_x, kernel_gauss_vert, g_d_Ix_2);

        // Calcolo di g(I_y^2)
        floatConv(d_Iy_2, kernel_gauss_oriz, t_y);
        floatConv(t_y, kernel_gauss_vert, g_d_Iy_2);

        // 3. COMPUTE THETA AND THRESHOLD IT
        cv::Mat theta = cv::Mat(image.rows, image.cols, image.type());
        theta = g_d_Ix_2.mul(g_d_Iy_2) - blur_bidim.mul(blur_bidim) - alpha * ((g_d_Ix_2 + g_d_Iy_2).mul(g_d_Ix_2 + g_d_Iy_2));
		#ifdef DEBUG
        cv::imshow("theta", theta);
		#endif
	
        // 4. CONSIDER A NMS ON A 3x3 WINDOW
        // 5. FOR EACH SURVIVING THETA VALUE SAVE THE POINT IN A LIST

		std::vector<cv::KeyPoint> keypoints;

        for (int i = 0; i < theta.rows; i++) {
            for (int j = 0; j < theta.cols; j++) {
                if (theta.at<float>(i, j) > harrisTh && computeNMS(theta, i, j))
                    keypoints.push_back(cv::KeyPoint(float(j), float(i), 3));
					#ifdef DEBUG
					std::cout << "corner detected at: " << i << ", " << j << std::endl;
					#endif
            }
        }

        cv::Mat output;
        cv::drawKeypoints(image_color, keypoints, output, cv::Scalar(0, 0, 255));
        cv::namedWindow("Result", cv::WINDOW_NORMAL);
        cv::imshow("Result", output);

        // wait for key or timeout
        unsigned char key = cv::waitKey(args.wait_t);
        std::cout << "key " << int(key) << std::endl;

        // here you can implement some looping logic using key value:
        //  - pause
        //  - stop
        //  - step back
        //  - step forward
        //  - loop on the same frame

        switch (key) {
        case 'p':
            std::cout << "Mat = " << std::endl
                      << image << std::endl;
            break;
        case 'q':
            exit(0);
            break;
        }
    }

    return 0;
}

#include <unistd.h>
bool ParseInputs(ArgumentList& args, int argc, char** argv)
{
    int c;
    args.alpha = 0.05;
    args.th = 50000.0;
    args.wait_t = 0;

    while ((c = getopt(argc, argv, "hi:t:a:")) != -1)
        switch (c) {
        case 't':
            args.th = atof(optarg);
            break;
        case 'a':
            args.alpha = atof(optarg);
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
