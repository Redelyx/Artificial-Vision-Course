//./main images/left.pgm images/right.pgm images/dsi.bin
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//std:
#include <fstream>
#include <iostream>
#include <string>

#define USE_OPENCVVIZ
#ifdef USE_OPENCVVIZ
#include <opencv2/viz.hpp>

void PointsToMat(const std::vector<cv::Point3f> &points, cv::Mat &mat)
{
	mat = cv::Mat(1, 3 * points.size(), CV_32FC3);
	for (unsigned int i = 0, j = 0; i < points.size(); ++i, j += 3)
	{
		mat.at<float>(j) = points[i].x;
		mat.at<float>(j + 1) = points[i].y;
		mat.at<float>(j + 2) = points[i].z;
	}
}
#endif
#define DMAX 127

//Not used, used as test
int computeSAD(const cv::Mat &l, const cv::Mat &r, unsigned short size, int d, int col_left, int row_left)
{
	int SAD = 0;
	for (int i = -size / 2; i <= size / 2; i++)
	{
		for (int j = -size / 2; j <= size / 2; j++)
		{
			SAD += abs(l.at<unsigned char>(row_left + i, col_left + j) - r.at<unsigned char>(row_left + i, col_left + j - d));
		}
	}
	return SAD;
}

int computeFirstRow(const cv::Mat &l, const cv::Mat &r, unsigned short size, int d, int col_left, std::vector<int> &cy_buffer)
{
	int row_left = size / 2;
	int SAD = 0;
	for (int i = -size / 2; i <= size / 2; i++)
	{
		float cy = 0;
		for (int j = -size / 2; j <= size / 2; j++)
		{
			cy += abs(l.at<unsigned char>(row_left + i, col_left + j) - r.at<unsigned char>(row_left + i, col_left + j - d));
		}
		SAD += cy;
		cy_buffer[(row_left - i) * l.cols * DMAX + col_left * DMAX + d] = cy;
	}
	return SAD;
}

int calculateCyD(const cv::Mat &l, const cv::Mat &r, unsigned short size, std::vector<int> &cy_buffer, int col_left, int row_left, int d)
{
	int topRow = cy_buffer[(row_left - 1 - size / 2) * l.cols * DMAX + col_left * DMAX + d];
	int botRow = 0;
	for (int col = -size / 2; col <= size / 2; col++)
	{
		int pixelL = l.at<unsigned char>(row_left + size / 2, col_left + col);
		int pixelR = r.at<unsigned char>(row_left + size / 2, col_left + col - d);
		botRow += abs(pixelL - pixelR);
	}
	cy_buffer[(row_left + size / 2) * l.cols * DMAX + col_left * DMAX + d] = botRow;
	return -topRow + botRow;
}

// USE SEMI INCREMENTAL SAD
void disp(const cv::Mat &l, const cv::Mat &r, unsigned short size, cv::Mat &out)
{
	out = cv::Mat::zeros(l.size(), CV_8UC1);
	/// SAD
	// for (int row_left = size / 2 + 1; row_left < l.rows - size / 2; row_left++)
	// {
	// 	for (int col_left = size / 2; col_left < l.cols - size / 2; col_left++)
	// 	{
	// 		int minSAD = INT_MAX;
	// 		int minD = 0;
	// 		for (int d = 0; d < DMAX && (col_left - d) >= size / 2; d++)
	// 		{
	// 			int SAD = computeSAD(l, r, size, d, col_left, row_left);
	// 			if (SAD < minSAD)
	// 			{
	// 				minSAD = SAD;
	// 				minD = d;
	// 			}
	// 		}
	// 		out.data[(col_left + row_left * out.cols) * out.elemSize()] = minD;
	// 	}
	// }

	std::vector<int> cy_buffer(l.rows * l.cols * DMAX);
	std::vector<int> SAD_buffer(l.cols * DMAX);

	int minD = 0;
	int row_left = size / 2;
	for (int col_left = size / 2; col_left < l.cols - size / 2; col_left++)
	{
		int minSAD = INT_MAX;
		for (int d = 0; d < DMAX && (col_left - d) >= size / 2; d++)
		{
			int SAD = computeFirstRow(l, r, size, d, col_left, cy_buffer);
			if (SAD < minSAD)
			{
				minSAD = SAD;
				minD = d;
			}
			SAD_buffer[col_left * DMAX + d] = SAD;
		}
		out.data[(col_left + row_left * l.cols) * l.elemSize()] = minD;
	}

	for (row_left = size / 2 + 1; row_left < l.rows - size / 2; row_left++)
	{
		for (int col_left = size / 2; col_left < l.cols - size / 2; col_left++)
		{
			int minSAD = INT_MAX;
			int minD = 0;
			for (int d = 0; d < DMAX && (col_left - d) >= size / 2; d++)
			{
				int SAD = SAD_buffer[col_left * DMAX + d] + calculateCyD(l, r, size, cy_buffer, col_left, row_left, d);
				if (SAD < minSAD)
				{
					minSAD = SAD;
					minD = d;
				}
				SAD_buffer[col_left * DMAX + d] = SAD;
			}
			out.data[(col_left + row_left * out.cols) * out.elemSize()] = minD;
		}
	}
}

/*
 * Piano passante per tre punti:
 *
 * https://en.wikipedia.org/wiki/Plane_(geometry)
 *
 * Equazione del piano usata: ax + by +cz + d = 0
 */
//
// DO NOT TOUCH
void plane3points(cv::Point3f p1, cv::Point3f p2, cv::Point3f p3, float &a, float &b, float &c, float &d)
{
	cv::Point3f p21 = p2 - p1;
	cv::Point3f p31 = p3 - p1;

	a = p21.y * p31.z - p21.z * p31.y;
	b = p21.x * p31.z - p21.z * p31.x;
	c = p21.x * p31.y - p21.y * p31.x;

	d = -(a * p1.x - b * p1.y + c * p1.z);
}

/*
 * Distanza punto piano.
 *
 * https://en.wikipedia.org/wiki/Plane_(geometry)
 */
//
// DO NOT TOUCH
float distance_plane_point(cv::Point3f p, float a, float b, float c, float d)
{
	return fabs((a * p.x + b * p.y + c * p.z + d)) / (sqrt(a * a + b * b + c * c));
}

/////////////////////////////////////////////////////////////////////
//	EX1
//
//	 Calcolare le coordinate 3D x,y,z per ogni pixel, nota la disparita'
//
//	 Si vedano le corrispondenti formule per il calcolo (riga, colonna, disparita') -> (x,y,z)
//
//	 Utilizzare i parametri di calibrazione forniti
//
//	 I valori di disparita' sono contenuti nell'immagine disp
void compute3Dpoints(const cv::Mat &disp, std::vector<cv::Point3f> &points, std::vector<cv::Point2i> &rc)
{
	//parametri di calibrazione predefiniti
	// DO NOT TOUCH
	constexpr float focal = 657.475;
	constexpr float baseline = 0.3;
	constexpr float u0 = 509.5;
	constexpr float v0 = 247.15;

	for (int r = 0; r < disp.rows; ++r)
	{
		for (int c = 0; c < disp.cols; ++c)
		{

			if (disp.at<float>(r, c) > 1)
			{
				int disparity = disp.at<float>(r, c);
				float x, y, z;

				/*
				 * YOUR CODE HERE
				 *
				 * calcolare le coordinate x,y,z a partire dalla disparita' e da riga/colonna
				 */
				x = ((c - u0) * baseline) / disparity;
				y = ((r - v0) * baseline) / disparity;
				z = (baseline * focal) / disparity;

				// salvo tutti i punti 3D con z entro i 30m, per semplicita'
				if (z < 30)
				{
					points.push_back(cv::Point3f(x, y, z));
					rc.push_back(cv::Point2i(c, r));
				}
			}
		}
	}
}
//////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
//	 EX. 2
//
//	 Calcolare con RANSAC il piano che meglio modella i punti forniti
//
//	 Si tratta di implementare un plane fitting con RANSAC:
//
//	 1) scelgo a caso 3 punti da points
//	 2) calcolo il piano con la funzione fornita
//	 3) calcolo la distanza di tutti i punto dal piano, con la funzione fornita
//   4) calcolo gli inliers del modello attuale, salvando punti e coordinate immagine corrispondenti
//
//	 Mi devo salvare il modello che ha piu' inliers
//
//	 Una volta ottenuto il piano, generare un'immagine dei soli inliers
void computePlane(const std::vector<cv::Point3f> &points, const std::vector<cv::Point2i> &uv, std::vector<cv::Point3f> &inliers_best_points, std::vector<cv::Point2i> &inliers_best_uv)
{
	// Parametri di RANSAC:
	int N = 10000;		 // numero di iterazioni
	float epsilon = 0.2; //errore massimo di un inliers

	std::cout << "RANSAC" << std::endl;
	std::cout << "Points: " << points.size() << std::endl;

	/*
	 * YOUR CODE HERE
	 *
	 * Ciclo di RANSAC
	 */
	int numberOfInliers = 0;

	for (int iteration = 0; iteration < N; iteration++)
	{
		int p1 = 0;
		int p2 = 0;
		int p3 = 0;
		//All three points must be different
		while ((p1 == p2) || (p1 == p3) || (p2 == p3))
		{
			p1 = rand() % points.size();
			p2 = rand() % points.size();
			p3 = rand() % points.size();
		}
		// std::cout << p1 << " " << p2 << " " << p3 << std::endl;
		float a;
		float b;
		float c;
		float d;
		plane3points(points[p1], points[p2], points[p3], a, b, c, d);

		int inliers = 0;
		std::vector<cv::Point3f> inliers_best_points_tmp;
		std::vector<cv::Point2i> inliers_best_uv_tmp;

		for (int i = 0; i < points.size(); i++)
		{
			float distance = distance_plane_point(points[i], a, b, c, d);

			if (distance < epsilon)
			{
				inliers++;
				inliers_best_points_tmp.push_back(points[i]);
				inliers_best_uv_tmp.push_back(uv[i]);
			}
		}
		if (inliers > numberOfInliers)
		{
			inliers_best_points = inliers_best_points_tmp;
			inliers_best_uv = inliers_best_uv_tmp;
			numberOfInliers = inliers;
		}
	}
	std::cout << numberOfInliers << std::endl;
}

int main(int argc, char **argv)
{

	//////////////////////////////////////////////////////////////////
	// Parse argument list:
	//
	// DO NOT TOUCH
	if (argc < 4 && argc > 5)
	{
		std::cerr << "Usage ./main <left_image_filename> <right_image_filename> <dsi_filename> [<disparity_window_size>]" << std::endl;
		return 0;
	}

	//opening left file
	std::cout << "Opening " << argv[1] << std::endl;
	cv::Mat left_image = cv::imread(argv[1], CV_8UC1);
	if (left_image.empty())
	{
		std::cout << "Unable to open " << argv[1] << std::endl;
		return 1;
	}

	//opening right file
	std::cout << "Opening " << argv[2] << std::endl;
	cv::Mat right_image = cv::imread(argv[2], CV_8UC1);
	if (right_image.empty())
	{
		std::cout << "Unable to open " << argv[2] << std::endl;
		return 1;
	}

	unsigned short w_size = 11;
	if (argc == 5)
		w_size = atoi(argv[4]);

	std::cout << "Computing disparity" << std::endl;

	cv::Mat disparity;
	disp(left_image, right_image, w_size, disparity);

	cv::namedWindow("My Disparity", cv::WINDOW_NORMAL);
	cv::imshow("My Disparity", disparity);

	//////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////
	// Lettura delle disparita'
	//
	// DO NOT TOUCH
	cv::Mat imgDisparity16U(left_image.rows, left_image.cols, CV_16U, cv::Scalar(0));
	cv::Mat imgDisparityF32(left_image.rows, left_image.cols, CV_32FC1, cv::Scalar(0));

	// Leggiamo la dsi gia' PRECALCOLATA da file
	std::ifstream dsifile(argv[3], std::ifstream::binary);
	if (!dsifile.is_open())
	{
		std::cout << "Unable to open " << argv[3] << std::endl;
		return 1;
	}
	dsifile.seekg(0, std::ios::beg);
	dsifile.read((char *)imgDisparity16U.data, imgDisparity16U.rows * imgDisparity16U.cols * 2);
	dsifile.close();

	imgDisparity16U.convertTo(imgDisparityF32, CV_32FC1);
	imgDisparityF32 /= 16.0;
	//////////////////////////////////////////////////////////////////

	cv::Mat dsi;
	imgDisparityF32.convertTo(dsi, CV_8UC1);
	cv::namedWindow("dsi.bin", cv::WINDOW_NORMAL);
	cv::imshow("dsi.bin", dsi);

	/////////////////////////////////////////////////////////////////////
	//	EX1
	//
	//	 Calcolare le coordinate 3D x,y,z per ogni pixel, nota la disparita'
	//
	//	 Si vedano le corrispondenti formule per il calcolo (riga, colonna, disparita') -> (x,y,z)
	//
	//	 Utilizzare i parametri di calibrazione forniti
	//
	//	 I valori di disparita' sono contenuti nell'immagine imgDisparityF32

	//vettore dei punti 3D calcolati a partire disparita'
	std::vector<cv::Point3f> points;
	//vettore delle corrispondenti righe,colonne
	std::vector<cv::Point2i> rc;

	compute3Dpoints(imgDisparityF32, points, rc);
	/////////////////////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////
	//	 EX. 2
	//
	//	 Calcolare con RANSAC il piano che meglio modella i punti forniti
	//
	//	 Si tratta di implementare un plane fitting con RANSAC.
	//

	// vettore degli inliers del modello miglioe
	std::vector<cv::Point3f> inliers_best;

	//vettore delle coordinate (r,c) degli inliers miglioi
	std::vector<cv::Point2i> inliers_best_rc;

	computePlane(points, rc, inliers_best, inliers_best_rc);

	/*
	 * Creare un'immagine formata dai soli pixel inliers
	 *
	 * Nella parte di RANSAC precedente dovro' quindi calcolare, oltre ai punti 3D inliers, anche le loro coordinate riga colonna corrispondenti
	 *
	 * Salvare queste (r,c) nel vettore inliers_best_rc ed utilizzarlo adesso per scrivere l'immagine out con i soli pixel inliers
	 */

	//immagine di uscita che conterra' i soli pixel inliers
	cv::Mat out(left_image.rows, left_image.cols, CV_8UC1, cv::Scalar(0));
	for (cv::Point2i p : inliers_best_rc)
	{
		out.data[(p.x + p.y * out.cols) * out.elemSize()] = left_image.at<unsigned char>(p.y, p.x);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	//display images
	//
	// DO NOT TOUCH
#ifdef USE_OPENCVVIZ
	cv::viz::Viz3d win("3D view");
	win.setWindowSize(cv::Size(800, 600));

	cv::Mat points_mat;
	PointsToMat(points, points_mat);
	win.showWidget("cloud", cv::viz::WCloud(points_mat));

	std::cout << "Press q to exit" << std::endl;
	win.spin();
#endif

	cv::namedWindow("left image", cv::WINDOW_NORMAL);
	cv::imshow("left image", left_image);

	cv::namedWindow("left image out", cv::WINDOW_NORMAL);
	cv::imshow("left image out", out);

	//wait for key
	cv::waitKey();
	///////////////////////////////////////////////////////////////////////////////////////////////////////

	return 0;
}
