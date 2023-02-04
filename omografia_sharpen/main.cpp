//OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

//std:
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <iterator>

// eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>


//////////////////////////////////////////////
/// EX1
//
// Nota la posizione dei 4 angoli della copertina del libro nell'immagine "input.jpg"
// generare la corrispondente immagine vista dall'alto, senza prospettiva.
//
// Si tratta di trovare l'opportuna trasformazione che fa corrispondere la patch di immagine
// input.jpg corrispondente alla copertina del libro con la vista dall'alto della stessa.
//
// Che tipo di trasformazione e'? Come si puo' calcolare con i dati forniti?
//
// E' possibile utilizzare alcune funzioni di OpenCV
//
void WarpBookCover(const cv::Mat & image, cv::Mat & output, const std::vector<cv::Point2f> & corners_src)
{
	std::vector<cv::Point2f> corners_out;

	/*
	* YOUR CODE HERE
	*
	*
	*/

	/* nota la posizione dei 4 angoli della copertina in input, questi sono i corrispondenti
	 * nell'immagine di uscita.
	 * E' importante che l'ordine sia rispettato, quindi top-left, top-right, bottom-right, bottom-left
	 */
	corners_out = { cv::Point2f(0,0), cv::Point2f(output.cols-1,0), cv::Point2f(output.cols-1,output.rows-1), cv::Point2f(0,output.rows-1)};

	//calcolo l'omografia corrispondente
	cv::Mat H = cv::findHomography(cv::Mat(corners_out), cv::Mat(corners_src));
	std::cout<<"H:"<<std::endl<<H<<std::endl;

	for(int r=0;r<output.rows;++r)
	{
		for(int c=0;c<output.cols;++c)
		{
			cv::Mat p  = (cv::Mat_<double>(3, 1) << c, r, 1);

			//trasformo da immagine di destinazione a immagine di input
			cv::Mat pp = H*p;

			//da omogenee a eucliedee
			pp/=pp.at<double>(2,0);

			//verifico che la posizione ottenuta sia contenuta nell'immagine di input
			if(round(pp.at<double>(0,0))>=0 && round(pp.at<double>(0,0))<image.cols && round(pp.at<double>(1,0))>=0 && round(pp.at<double>(1,0))<image.rows)
			{
				//prelevo il pixel dall'immagine di input e lo copio in quella di output
				output.at<cv::Vec3b>(r,c) = image.at<cv::Vec3b>(round(pp.at<double>(1,0)), round(pp.at<double>(0,0)));
			}
		}
	}
}
/////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////
/// EX2
//
// Applicare il filtro di sharpening visto a lezione
//
// Per le convoluzioni potete usare le funzioni sviluppate per il primo assegnamento
//
//
/* Single channel convolution
 *
 * - output: always float
 * - input: generic single channel
 *
 */
template <class T>
int convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out)
{
    out = cv::Mat(image.rows, image.cols, CV_32FC1, cv::Scalar(0));

    int h_kh = (int) std::floor(kernel.rows/ 2);
    int h_kw = (int) std::floor(kernel.cols/ 2);

    for (int r = h_kh; r < out.rows - h_kh; r++)
    {
        for (int c = h_kw; c < out.cols - h_kw; c++)
        {
            for (int rr = -h_kh; rr <= h_kh; rr++)
            {
                for (int cc = -h_kw; cc <= h_kw; cc++)
                {
                	//senza usare at<>
                	//
//                    *((float*) &out.data[(c + r * out.cols) * out.elemSize()]) +=
//                    		*((T*)&image.data[(c + cc + (r + rr) * image.cols) * image.elemSize()]) *
//                            *((float*) &kernel.data[(cc + h_kw + (rr + h_kh) * kernel.cols) * kernel.elemSize()]);

                    out.at<float>(r,c) = float(image.at<T>(r+rr,c+cc)) * kernel.at<float>(rr + h_kh, cc + h_kw);

                }
            }
        }
    }

    return 0;
}

void sharpening(const cv::Mat & image, cv::Mat & output, float alpha)
{
	output = cv::Mat(image.rows, image.cols, image.type(), cv::Scalar(0));

    cv::Mat LoG_conv_I;

	/*
	* YOUR CODE HERE
	*
	*
	*/

    //versione semplificata del Laplacian of Gausiann
    cv::Mat LoG = (cv::Mat_<float>(3, 3) << 0,  1, 0,
                                            1, -4, 1,
                                            0,  1, 0);

    //convoluzione in float
    convFloat<uchar>(image, LoG, LoG_conv_I);

    for(int r=0;r<output.rows;++r)
    {
        for(int c=0;c<output.cols;++c)
        {
        	//funzione di sharpening
        	float value = float(image.at<uchar>(r,c)) - 0.8*LoG_conv_I.at<float>(r,c);

        	//saturo i valori ottenuti agli estremi dell'immagine di uscita, che assumiamo uchar
        	value = std::min(value, 255.0f);
        	value = std::max(value, 0.0f);
        	output.at<uchar>(r,c) = uchar(value);
        }
    }
}
//////////////////////////////////////////////


int main(int argc, char **argv) {
    
    if (argc != 2)
    {
        std::cerr << "Usage ./prova <image_filename>" << std::endl;
        return 0;
    }
    
    //images
    cv::Mat input;

    // load image from file
    input = cv::imread(argv[1]);
	if(input.empty())
	{
		std::cout<<"Error loading input image "<<argv[1]<<std::endl;
		return 1;
	}




    //////////////////////////////////////////////
    /// EX1
    //
    // Creare un'immagine contenente la copertina del libro input come vista "dall'alto" (senza prospettiva)
    //
    //
	//

	// Dimensioni note e fissate dell'immagine di uscita (vista dall'alto):
	constexpr int outwidth = 431;
	constexpr int outheight = 574;
	cv::Mat outwarp(outheight, outwidth, input.type(), cv::Scalar(0));

	//posizioni note e fissate dei quattro corner della copertina nell'immagine input
    std::vector<cv::Point2f> pts_src = { cv::Point2f(274,189), //top left
    		                             cv::Point2f(631,56), //top right
    									 cv::Point2f(1042,457), //bottom right
										 cv::Point2f(722,764)};//bottom left

    WarpBookCover(input, outwarp, pts_src);
    //////////////////////////////////////////////







    //////////////////////////////////////////////
    /// EX2
    //
    // Applicare uno sharpening all'immagine cover
    //
    // Immagine = Immagine - alfa(LoG * Immagine)
    //
    //
    // alfa e' una costante float, utilizziamo 0.5
    //
    //
    // LoG e' il Laplaciano del Gaussiano. Utilizziamo l'approssimazione 3x3 vista a lezione
    //
    //
    // In questo caso serve fare il contrast stratching nelle convoluzioni?
    //
    //

    //immagine di uscita sharpened
	cv::Mat sharpened(input.rows, input.cols, CV_8UC1);

	//convertiamo l'immagine della copertina a toni di grigio, per semplicita'
	cv::Mat inputgray(input.rows, input.cols, CV_8UC1);
	cv::cvtColor(input, inputgray, cv::COLOR_BGR2GRAY);

	sharpening(inputgray, sharpened, 0.8);
    //////////////////////////////////////////////






    ////////////////////////////////////////////
    /// WINDOWS
    cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input", input);
    
    cv::Mat outimage_win(std::max(input.rows, outwarp.rows), input.cols+outwarp.cols, input.type(), cv::Scalar(0));
    input.copyTo(outimage_win(cv::Rect(0,0,input.cols, input.rows)));
    outwarp.copyTo(outimage_win(cv::Rect(input.cols,0,outwarp.cols, outwarp.rows)));

    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    cv::imshow("Output", outimage_win);

    cv::namedWindow("Input Gray", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input Gray", inputgray);

    cv::namedWindow("Input Gray Sharpened", cv::WINDOW_AUTOSIZE);
    cv::imshow("Input Gray Sharpened", sharpened);

    cv::waitKey();

    return 0;
}





