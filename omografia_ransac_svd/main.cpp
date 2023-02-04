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
#include <algorithm>
#include <random>
#include <iterator>

// eigen
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

void gaussianKernel(float sigma, int radius, cv::Mat& kernel){

	int kernel_size = (2*radius)+1;
	kernel = cv::Mat(1,kernel_size,CV_32FC1);
	float* kernel_data = (float*) kernel.data;
	float sum = 0.0;

	for(int i = 0; i < kernel_size; i++){
		kernel_data[i] = exp(-0.5 * ((i-radius)/sigma) * ((i-radius)/sigma));
		sum += kernel_data[i];
	}
	for(int i = 0; i < kernel_size; i++){
		kernel_data[i] /= sum;
	}
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

void convFloat(const cv::Mat& image, const cv::Mat& kernel, cv::Mat& out){    //ok

	out=cv::Mat(image.rows,image.cols,CV_32FC1, cv::Scalar(0));
	float* out_data=(float *)out.data;
	float* kernel_data=(float *)kernel.data;
	for(int v=kernel.rows/2;v<out.rows-kernel.rows/2;v++){
		for(int u=kernel.cols/2;u<out.cols-kernel.cols/2;u++){
			float somma=0;
			for(int k_v=0;k_v<kernel.rows;k_v++){
				for(int k_u=0;k_u<kernel.cols;k_u++){
					somma+=((float)image.data[((u-kernel.cols/2+k_u)+(v-kernel.rows/2+k_v)*image.cols)*image.elemSize()]*kernel_data[k_u+k_v*kernel.cols]);
				}
			}
			out_data[u+v*out.cols]=somma;
		}
	}
}

bool maxLoc(cv::Mat input, int row, int col){
	return ( input.at<float>(row,col) > input.at<float>(row-1,col-1)   &&
           input.at<float>(row,col) > input.at<float>(row-1,col+1)   &&
           input.at<float>(row,col) > input.at<float>(row,col+1)     &&
           input.at<float>(row,col) > input.at<float>(row,col-1)     &&
           input.at<float>(row,col) > input.at<float>(row+1,col)     &&
           input.at<float>(row,col) > input.at<float>(row+1,col-1)   &&
           input.at<float>(row,col) > input.at<float>(row-1,col)     &&
           input.at<float>(row,col) > input.at<float>(row+1,col+1));
}

void myHarrisCornerDetector(const cv::Mat image, std::vector<cv::KeyPoint> & keypoints0, float alpha, float harrisTh)
{
  /**********************************
   *
   * PLACE YOUR CODE HERE
   *
   *
   *
   * E' ovviamente viatato utilizzare un detector di OpenCv....
   *
   */
  cv::Mat d_Ix;
	cv::Mat d_Iy;
	cv::Mat blur_oriz;
	cv::Mat blur_bidim;
 	cv::Mat kernel_gauss_oriz;
  cv::Mat t_x;
	cv::Mat t_y;
	float gradiente_data[3] = {-1,0,1};

  // Calcolo I_x
	cv::Mat gradiente_oriz = cv::Mat(1,3,CV_32FC1,gradiente_data);
	convFloat(image,gradiente_oriz,d_Ix);

  // Calcolo I_y
	cv::Mat gradiente_vert = gradiente_oriz.t();
	convFloat(image,gradiente_vert,d_Iy);

  // Calcolo I_xy
	cv::Mat d_Ixy = cv::Mat(image.rows,image.cols,CV_32FC1);
	d_Ixy = d_Ix.mul(d_Iy);

  int raggio = 1;
	float sigma = 47.0f;

	//Calcolo g(I_x).
	gaussianKernel(sigma,raggio,kernel_gauss_oriz);

	//Calcolo g(I_y).
	cv::Mat kernel_gauss_vert = kernel_gauss_oriz.t();

	//Calcolo di g(I_x*I_y).
	floatConv(d_Ixy, kernel_gauss_oriz, blur_oriz);
	floatConv(blur_oriz,kernel_gauss_vert,blur_bidim);

	//Calcolo di d(I_x)^2 e d(I_y)^2
	cv::Mat d_Ix_2 = cv::Mat(d_Ix.rows,d_Ix.cols,d_Ix.type());
	cv::Mat d_Iy_2 = cv::Mat(d_Iy.rows,d_Iy.cols,d_Iy.type());
	d_Ix_2 = d_Ix.mul(d_Ix);
	d_Iy_2 = d_Iy.mul(d_Iy);


	cv::Mat g_d_Ix_2;
	cv::Mat g_d_Iy_2;

	//Calcolo di g(I_x^2)
	floatConv(d_Ix_2, kernel_gauss_oriz, t_x);
	floatConv(t_x, kernel_gauss_vert, g_d_Ix_2);

	//Calcolo di g(I_y^2)
	floatConv(d_Iy_2, kernel_gauss_oriz, t_y);
	floatConv(t_y, kernel_gauss_vert, g_d_Iy_2);


  ////////////////////////////////////////////////////////
  /// HARRIS CORNER
  //

  //Calcolo il teta
	cv::Mat teta = cv::Mat(image.rows,image.cols,image.type());
	teta = g_d_Ix_2.mul(g_d_Iy_2)-blur_bidim.mul(blur_bidim) - alpha*((g_d_Ix_2 + g_d_Iy_2).mul(g_d_Ix_2 + g_d_Iy_2));

  for(int i = 0; i < teta.rows; i++){
		for(int j = 0; j < teta.cols; j++){
			if(teta.at<float>(i,j) > harrisTh && maxLoc(teta, i, j))
				keypoints0.push_back(cv::KeyPoint (float (j), float (i), float (1)));
		}
	}

  // Disegnate tutti i risultati intermendi per capire se le cose funzionano
  //
  // Per la response di Harris:
      cv::Mat adjMap;
      cv::Mat falseColorsMap;
      double minr,maxr;
  
      cv::minMaxLoc(teta, &minr, &maxr);
      cv::convertScaleAbs(teta, adjMap, 255 / (maxr-minr));
      cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
      cv::namedWindow("response1", cv::WINDOW_NORMAL);
      cv::imshow("response1", falseColorsMap);

  // HARRIS CORNER END
  ////////////////////////////////////////////////////////
}

void myFindHomographySVD(const std::vector<cv::Point2f> & points1, const std::vector<cv::Point2f> & points0, cv::Mat & H)
{
  cv::Mat A(points1.size()*2,9, CV_64FC1, cv::Scalar(0));

  /**********************************
   *
   * PLACE YOUR CODE HERE
   *
   *
   * Utilizzate la funzione:
   * cv::SVD::compute(A,D, U, Vt);
   *
   * In pratica dovete costruire la matrice A opportunamente e poi prendere l'ultima colonna di V
   *
   */
  
	//Definizione delle matrici necessarie.
	cv::Mat D, U , Vt;
	//Calcolo A
	for(int i = 0; i < int(points1.size()); i++){
		A.at<double>(2*i+1,0) = 0;
		A.at<double>(2*i+1,1) = 0;
		A.at<double>(2*i+1,2) = 0;
		A.at<double>(2*i+1,3) = -points1[i].x;;
		A.at<double>(2*i+1,4) = -points1[i].y;
		A.at<double>(2*i+1,5) = -1;
		A.at<double>(2*i+1,6) = points1[i].x * points0[i].y;
		A.at<double>(2*i+1,7) = points1[i].y * points0[i].y;
		A.at<double>(2*i+1,8) = points0[i].y;
    A.at<double>(2*i,0) = -points1[i].x;
		A.at<double>(2*i,1) = -points1[i].y;
		A.at<double>(2*i,2) = -1;
		A.at<double>(2*i,3) = 0;
		A.at<double>(2*i,4) = 0;
		A.at<double>(2*i,5) = 0;
		A.at<double>(2*i,6) = points1[i].x * points0[i].x;
		A.at<double>(2*i,7) = points1[i].y * points0[i].x;
		A.at<double>(2*i,8) = points0[i].x;
	}

	cv::SVD::compute(A, D, U, Vt, cv::SVD::FULL_UV);
	
  cv::Mat V = Vt.t();
	//Prendo l'ultima colonna di V.
	for(int i = 0; i < 9; i++){
		H.at<double>(i / 3, i % 3) = V.at<double>(i, V.cols - 1);
	}
	
  // ricordatevi di normalizzare alla fine
  H /= H.at<double>(2,2);
	A.release();
	D.release();
	U.release();
	V.release();
	Vt.release();
}

void myFindHomographyRansac(const std::vector<cv::Point2f> & points1, const std::vector<cv::Point2f> & points0,
        const std::vector<cv::DMatch> & matches, int N, float epsilon, int sample_size, cv::Mat & H,
        std::vector<cv::DMatch> & matchesInlierBest)
{
  srand(time(NULL));
  std::vector<cv::Point2f> inlier_0, inlier_1;
	std::vector<cv::Point2f> inliers_top_0, inliers_top_1;
	std::vector<cv::DMatch> matches_inl;
	H = cv::Mat(3, 3, CV_64FC1, cv::Scalar(0));

	cv::Mat hp1 = cv::Mat(3, 1, CV_64FC1, cv::Scalar(0));
	cv::Mat p0 = cv::Mat(2, 2, CV_64FC1, cv::Scalar(0));
	cv::Mat p1 = cv::Mat(2, 2, CV_64FC1, cv::Scalar(0));

	//N iterazioni RANSAC
	for(int i=0;i<N;i++){
		std::vector<cv::Point2f> rand_0, rand_1;
		//Evitiamo di selezionare lo stesso match piú volte.
    int visited[points0.size()] = {0};
		bool not_visited;
		//4 match random
		for(int j=0;j<sample_size;j++){
			not_visited = false;
			while(!not_visited) {
				int random = rand()%points0.size();
				if(visited[random]==0){
					visited[random]=1;
					rand_0.push_back(points0[random]);
					rand_1.push_back(points1[random]);
					not_visited = true;
				}
			}
		}

		//Calcolo della matrice H con match selezionati a caso.
		myFindHomographySVD(rand_1, rand_0, H);

		inlier_0.clear();
		inlier_1.clear();
		matches_inl.clear();

    // ricalcola H utilizzando i match del set di inliers più numeroso.
		for(int k=0;k<int(points0.size());k++){
			//H*p1.
			for(int j=0;j<3;j++){
				hp1.at<double>(j,0) = (H.at<double>(j,0) * points1[k].x) + 
                              (H.at<double>(j,1) * points1[k].y) + H.at<double>(j,2);
			}

			p0.at<double>(0,0) = points0[k].x;
			p0.at<double>(1,0) = points0[k].y;

			//Coordinate euclidee.
			p1.at<double>(0,0)=hp1.at<double>(0,0)/hp1.at<double>(2,0);
			p1.at<double>(1,0)=hp1.at<double>(1,0)/hp1.at<double>(2,0);

			double norma = cv::norm(cv::Mat(p0),cv::Mat(p1));
			//Verfico che la norma sia minore di epsilon(errore).
			if(norma < epsilon){
				inlier_0.push_back(points0[k]);
				inlier_1.push_back(points1[k]);
				matches_inl.push_back(matches[k]);
			}
		}
		//Aggiorno i top inlier.
		if(inlier_0.size()>inliers_top_0.size()){
			inliers_top_0.clear();
			inliers_top_1.clear();
			matchesInlierBest.clear();
			
      int dim_inliers=int(inlier_0.size());
			
      for(int j=0;j<dim_inliers;j++){
				inliers_top_0.push_back(inlier_0[j]);
				inliers_top_1.push_back(inlier_1[j]);
				matchesInlierBest.push_back(matches_inl[j]);
			}
		}
	}
	
	myFindHomographySVD(inliers_top_1, inliers_top_0, H);
}


int main(int argc, char **argv) {

  if (argc < 4) 
  {
    std::cerr << "Usage prova <image_filename> <book_filename> <alternative_cover_filename>" << std::endl;
    return 0;
  }

  // images
  cv::Mat input, cover, newcover;

  // load image from file
  input = cv::imread(argv[1], CV_8UC1);
  if(input.empty())
  {
    std::cout<<"Error loading input image "<<argv[1]<<std::endl;
    return 1;
  }

  // load image from file
  cover = cv::imread(argv[2], CV_8UC1);
  if(cover.empty())
  {
    std::cout<<"Error loading book image "<<argv[2]<<std::endl;
    return 1;
  }

  //load image from file
	newcover = cv::imread(argv[3], CV_8UC1);
	if(newcover.empty())
	{
		std::cout<<"Error loading newcover image "<<argv[3]<<std::endl;
		return 1;
	}

  ////////////////////////////////////////////////////////
  /// HARRIS CORNER
  //
  float alpha = 0.04;
  float harrisTh = 6000000.0f;    //da impostare in base alla propria implementazione!!!!!

  std::vector<cv::KeyPoint> keypoints0, keypoints1;

  // FASE 1
  //
  // Qui sotto trovate i corner di Harris di OpenCV
  //
  // Da commentare e sostituire con la propria implementazione
  //
   /* {
    std::vector<cv::Point2f> corners;
    int maxCorners = 0;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = true;
    double k = 0.04;

    cv::goodFeaturesToTrack( input,corners,maxCorners,qualityLevel,minDistance,cv::noArray(),blockSize,useHarrisDetector,k ); // estrae strong feature (k -> alpha)
    std::transform(corners.begin(), corners.end(), std::back_inserter(keypoints0), [](const cv::Point2f & p){ return cv::KeyPoint(p.x,p.y,3.0);} ); // applica funzione a range vector e memorizza in altro range 3->size del keypoint

    corners.clear();
    cv::goodFeaturesToTrack( cover, corners, maxCorners, qualityLevel, minDistance, cv::noArray(), blockSize, useHarrisDetector, k );
    std::transform(corners.begin(), corners.end(), std::back_inserter(keypoints1), [](const cv::Point2f & p){ return cv::KeyPoint(p.x,p.y,3.0);} );
  } */
  //
  //
  //
  // Abilitare il proprio detector una volta implementato
  //
  //
  myHarrisCornerDetector(input, keypoints0, alpha, harrisTh);
  myHarrisCornerDetector(cover, keypoints1, alpha, harrisTh);
  //
  //
  //


  std::cout<<"keypoints0 "<<keypoints0.size()<<std::endl;
  std::cout<<"keypoints1 "<<keypoints1.size()<<std::endl;
  //
  //
  ////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////
  /// CALCOLO DESCRITTORI E MATCHES
  //
  int briThreshl=30;
  int briOctaves = 3;
  int briPatternScales = 1.0;
  cv::Mat descriptors0, descriptors1;

  //dichiariamo un estrattore di features di tipo BRISK
  cv::Ptr<cv::DescriptorExtractor> extractor = cv::BRISK::create(briThreshl, briOctaves, briPatternScales);
  //calcoliamo il descrittore di ogni keypoint
  extractor->compute(input, keypoints0, descriptors0);
  extractor->compute(cover, keypoints1, descriptors1);

  //associamo i descrittori tra me due immagini
  std::vector<std::vector<cv::DMatch> > matches;
  std::vector<cv::DMatch> matchesDraw;
  cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING); // brute force matcher (TODO aggiornare con .create()), usa hamming distance tra vettori
  //matcher.radiusMatch(descriptors0, descriptors1, matches, input.cols*2.0);
  matcher.match(descriptors0, descriptors1, matchesDraw);

  //copio i match dentro a dei semplici vettori oint2f
  std::vector<cv::Point2f> points[2];
  for(unsigned int i=0; i<matchesDraw.size(); ++i)
  {
    points[0].push_back(keypoints0.at(matchesDraw.at(i).queryIdx).pt);
    points[1].push_back(keypoints1.at(matchesDraw.at(i).trainIdx).pt);
  }
  ////////////////////////////////////////////////////////


  ////////////////////////////////////////////////////////
  // CALCOLO OMOGRAFIA
  //
  //
  // E' obbligatorio implementare RANSAC.
  //
  // Per testare i corner di Harris inizialmente potete utilizzare findHomography di opencv, che include gia' RANSAC
  //
  // Una volta che avete verificato che i corner funzionano, passate alla vostra implementazione di RANSAC
  //
  //
  cv::Mat H;                                  //omografia finale
  std::vector<cv::DMatch> matchesInliersBest; //match corrispondenti agli inliers trovati
  std::vector<cv::Point2f> corners_cover;     //coordinate dei vertici della cover sull'immagine di input
  bool have_match=false;                      //verra' messo a true in caso ti match

  //
  // Verifichiamo di avere almeno 4 inlier per costruire l'omografia
  //
  //
  if(points[0].size()>=4)
  {
    //
    // Soglie RANSAC
    //
    // Piuttosto critiche, da adattare in base alla propria implementazione
    //
    int N=50000;            //numero di iterazioni di RANSAC
    float epsilon = 3;      //distanza per il calcolo degli inliers


    // Dimensione del sample per RANSAC, quiesto e' fissato
    //
    int sample_size = 4;    //dimensione del sample di RANSAC

    //////////
    // FASE 2
    //
    //
    //
    // Inizialmente utilizzare questa chiamata OpenCV, che utilizza RANSAC, per verificare i vostri corner di Harris
    //
    //
    /* cv::Mat mask;
    H = cv::findHomography( cv::Mat(points[1]), cv::Mat(points[0]), cv::RANSAC, 3, mask);
    for(std::size_t i=0;i<matchesDraw.size();++i)
      if(mask.at<uchar>(0,i) == 1) matchesInliersBest.push_back(matchesDraw[i]); */
    //
    //
    //
    // Una volta che i vostri corner di Harris sono funzionanti, commentare il blocco sopra e abilitare la vostra myFindHomographyRansac
    //
    myFindHomographyRansac(points[1], points[0], matchesDraw, N, epsilon, sample_size, H, matchesInliersBest);
    //
    //
    //

    std::cout<<std::endl<<"Risultati Ransac: "<<std::endl;
    std::cout<<"Num inliers / match totali  "<<matchesInliersBest.size()<<" / "<<matchesDraw.size()<<std::endl;

    std::cout<<"H"<<std::endl<<H<<std::endl;

    //
    // Facciamo un minimo di controllo sul numero di inlier trovati
    //
    //
    float match_kpoints_H_th = 0.1;
    if(matchesInliersBest.size() > matchesDraw.size()*match_kpoints_H_th)
    {
      std::cout<<"MATCH!"<<std::endl;
      have_match = true;


      // Calcoliamo i bordi della cover nell'immagine di input, partendo dai corrispondenti nell'immagine target
      //
      //
      cv::Mat p  = (cv::Mat_<double>(3, 1) << 0, 0, 1);
      cv::Mat pp = H*p;
      pp/=pp.at<double>(2,0);
      std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
      if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
      {
	corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
      }

      p  = (cv::Mat_<double>(3, 1) << cover.cols-1, 0, 1);
      pp = H*p;
      pp/=pp.at<double>(2,0);
      std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
      if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
      {
	corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
      }

      p  = (cv::Mat_<double>(3, 1) << cover.cols-1, cover.rows-1, 1);
      pp = H*p;
      pp/=pp.at<double>(2,0);
      std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
      if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
      {
	corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
      }

      p  = (cv::Mat_<double>(3, 1) << 0,cover.rows-1, 1);
      pp = H*p;
      pp/=pp.at<double>(2,0);
      std::cout<<std::endl<<p<<"->"<<pp<<std::endl;
      if(pp.at<double>(0,0)>=0 && pp.at<double>(0,0)<input.cols && pp.at<double>(1,0)>=0 && pp.at<double>(1,0)<input.rows)
      {
	corners_cover.push_back(cv::Point2f(pp.at<double>(0,0),pp.at<double>(1,0)));
      }
    }
    else
    {
      std::cout<<"Pochi inliers! "<<matchesInliersBest.size()<<"/"<<matchesDraw.size()<<std::endl;
    }


  }
  else
  {
    std::cout<<"Pochi match! "<<points[0].size()<<"/"<<keypoints0.size()<<std::endl;
  }
  ////////////////////////////////////////////////////////

  ////////////////////////////////////////////
  /// WINDOWS
  cv::Mat inputKeypoints;
  cv::Mat coverKeypoints;
  cv::Mat outMatches;
  cv::Mat outInliers;

  cv::drawKeypoints(input, keypoints0, inputKeypoints);
  cv::drawKeypoints(cover, keypoints1, coverKeypoints);

  cv::drawMatches(input, keypoints0, cover, keypoints1, matchesDraw, outMatches);
  cv::drawMatches(input, keypoints0, cover, keypoints1, matchesInliersBest, outInliers);


  // se abbiamo un match, disegniamo sull'immagine di input i contorni della cover
  if(have_match)
  {
    for(unsigned int i = 0;i<corners_cover.size();++i)
    {
      cv::line(input, cv::Point(corners_cover[i].x , corners_cover[i].y ), cv::Point(corners_cover[(i+1)%corners_cover.size()].x , corners_cover[(i+1)%corners_cover.size()].y ), cv::Scalar(255), 2, 8, 0);
    }
  }

  // copertina del libro sostitutiva.
  cv::Point2f corner_newcover;
  std::vector<cv::Point2f> vet;
  cv::Mat vet_corn_cover;
  cv::Mat img_rotate;
  cv::Mat input_newcover = input.clone();
  
  corner_newcover.x = 0;
  corner_newcover.y = 0;
  vet.push_back(corner_newcover);
  corner_newcover.x = newcover.cols-1;
  corner_newcover.y = 0;
  vet.push_back(corner_newcover);
  corner_newcover.x = newcover.cols-1;
  corner_newcover.y = newcover.rows-1;
  vet.push_back(corner_newcover);
  corner_newcover.x = 0;
  corner_newcover.y = newcover.rows-1;
  vet.push_back(corner_newcover);

  vet_corn_cover = cv::getPerspectiveTransform(vet,corners_cover);
  
  cv::warpPerspective(newcover,img_rotate,vet_corn_cover,input.size(),cv::INTER_LINEAR,cv:: BORDER_CONSTANT);
  cv::imshow("img_rotate",img_rotate);

  for(int i=0;i<input_newcover.rows;i++){
    for(int j=0;j<input_newcover.cols;j++){
      if(img_rotate.data[(j + i*img_rotate.cols)*img_rotate.elemSize()] != 0)
        input_newcover.data[(j + i*input_newcover.cols)*input_newcover.elemSize()] = img_rotate.data[(j + i*img_rotate.cols)*img_rotate.elemSize()];
    }
  }

  cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);
  cv::imshow("Input", input);

  cv::namedWindow("NewCover", cv::WINDOW_AUTOSIZE);
  cv::imshow("NewCover", input_newcover);

  cv::namedWindow("BookCover", cv::WINDOW_AUTOSIZE);
  cv::imshow("BookCover", cover);

  cv::namedWindow("inputKeypoints", cv::WINDOW_AUTOSIZE);
  cv::imshow("inputKeypoints", inputKeypoints);

  cv::namedWindow("coverKeypoints", cv::WINDOW_AUTOSIZE);
  cv::imshow("coverKeypoints", coverKeypoints);

  cv::namedWindow("Matches", cv::WINDOW_AUTOSIZE);
  cv::imshow("Matches", outMatches);

  cv::namedWindow("Matches Inliers", cv::WINDOW_AUTOSIZE);
  cv::imshow("Matches Inliers", outInliers);

  cv::waitKey();

  return 0;
}