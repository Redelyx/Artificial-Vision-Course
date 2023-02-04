// OpneCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// std:
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

// #define DEBUG
//#define USE_OPENCVVIZ
#ifdef USE_OPENCVVIZ
#include <opencv2/viz.hpp>

void PointsToMat(const std::vector<cv::Point3f>& points, cv::Mat& mat)
{
    mat = cv::Mat(1, 3 * points.size(), CV_32FC3);
    for (unsigned int i = 0, j = 0; i < points.size(); ++i, j += 3) {
        mat.at<float>(j) = points[i].x;
        mat.at<float>(j + 1) = points[i].y;
        mat.at<float>(j + 2) = points[i].z;
    }
}
#endif

/*
 * Piano passante per tre punti:
 *
 * https://en.wikipedia.org/wiki/Plane_(geometry)
 *
 * Equazione del piano usata: ax + by +cz + d = 0
 */
//
// DO NOT TOUCH
void plane3points(cv::Point3f p1, cv::Point3f p2, cv::Point3f p3, float& a, float& b, float& c, float& d)
{
    cv::Point3f p21 = p2 - p1;
    cv::Point3f p31 = p3 - p1;

    a = p21.y * p31.z - p21.z * p31.y;
    b = p21.x * p31.z - p21.z * p31.x;
    c = p21.x * p31.y - p21.y * p31.x;

    d = -(a * p1.x + b * p1.y + c * p1.z);
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
void compute3Dpoints(const cv::Mat& disp, std::vector<cv::Point3f>& points, std::vector<cv::Point2i>& rc)
{
    // parametri di calibrazione predefini
    //  DO NOT TOUCH
    constexpr float focal = 657.475;
    constexpr float baseline = 0.3;
    constexpr float u0 = 509.5;
    constexpr float v0 = 247.15;

    for (int r = 0; r < disp.rows; ++r) {
        for (int c = 0; c < disp.cols; ++c) {
            if (disp.at<float>(r, c) > 1) {
                /*
                 * YOUR CODE HERE
                 *
                 * calcolare le coordinate x,y,z a partire dalla disparita' e da riga/colonna
                 */
                float x, y, z;

                x = ((c - u0) * baseline) / disp.at<float>(r, c);
                y = ((r - v0) * baseline) / disp.at<float>(r, c);
                z = baseline * focal / disp.at<float>(r, c);
                /*
                 *
                 */

                // salvo tutti i punti 3D con z entro i 30m, per semplicita'
                if (z < 30) {
                    points.push_back(cv::Point3f(x, y, z));
                    rc.push_back(cv::Point2i(r, c));
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

void computePlane(const std::vector<cv::Point3f>& points, const std::vector<cv::Point2i>& uv, std::vector<cv::Point3f>& inliers_best_points, std::vector<cv::Point2i>& inliers_best_uv)
{
    // Parametri di RANSAC:
    int N = 10000; // numero di iterazioni
    float epsilon = 0.2; // errore massimo di un inliers

    std::cout << "RANSAC" << std::endl;
    std::cout << "Points: " << points.size() << std::endl;
	std::cout << "uv: " << uv.size() << std::endl;

	int best_n_inliers = 0;

    /*
     * YOUR CODE HERE
     *
     * Ciclo di RANSAC
     */
    for (int i = 0; i < N; i++) {
		//scelgo tre punti random
		float a, b, c, d;
		std::vector<cv::Point3f> chosen_p;
		std::vector<cv::Point2i> chosen_uv;
		std::vector<cv::Point3f> temp_inliers_point;
		std::vector<cv::Point2i> temp_inliers_uv;
        std::set<int> chosenIndexes;

		int temp_n_inliers = 0;

        while (chosenIndexes.size() < 3) {
            int randomIndex = rand() % points.size();
            if (chosenIndexes.find(randomIndex) == chosenIndexes.end()) {
                chosenIndexes.insert(randomIndex);
                chosen_p.push_back(points[randomIndex]);
            }
        }

		#ifdef DEBUG
		std::cout<< "points: " << chosen_p[0] << ", " << chosen_p[1] << ", " << chosen_p[2] << std::endl;
		#endif
		//calcolo il piano passante per i 3 punti
        plane3points(chosen_p[0], chosen_p[1], chosen_p[2], a, b, c, d);

		//calcolo quanti inliers ha il modello appena calcolato
		for(int j = 0; j < points.size(); j++){
			/* a = -0.00603338;
			b = -0.765999; 
			c = -0.0501304;
			d = 0.806783; */

			float temp_dist = distance_plane_point(points[j], a, b, c, d);
			if(temp_dist < epsilon){
				temp_n_inliers++;
				temp_inliers_point.push_back(points[j]);
				temp_inliers_uv.push_back(uv[j]);
			}
		}
		if(temp_n_inliers > best_n_inliers){
			#ifdef DEBUG
			std::cout<< "update inliers: " << best_n_inliers << " -> " << temp_n_inliers << " p= " << chosen_p << " uv = " << chosen_uv << std::endl;
			#endif
			best_n_inliers = temp_n_inliers;
			inliers_best_points = temp_inliers_point;
			inliers_best_uv = temp_inliers_uv;
			std::cout << "update inliers: " << best_n_inliers << std::endl;
			std::cout << "a: " << a << " b: " << b << " c: " << c << " d: " << d << std::endl;
		}
    }
	#ifdef DEBUG
	std::cout << "Num inliers: " << best_n_inliers << std::endl;
	std::cout << "Num points: " << inliers_best_points.size()<< std::endl;
	std::cout << "Num uv: " << inliers_best_uv.size() << std::endl;
	#endif
}

int main(int argc, char** argv)
{

    //////////////////////////////////////////////////////////////////
    // Parse argument list:
    //
    // DO NOT TOUCH
    if (argc != 3) {
        std::cerr << "Usage ./prova <left_image_filename> <dsi_filename>" << std::endl;
        return 0;
    }

    // opening left file
    std::cout << "Opening " << argv[1] << std::endl;
    cv::Mat left_image = cv::imread(argv[1], CV_8UC1);
    if (left_image.empty()) {
        std::cout << "Unable to open " << argv[1] << std::endl;
        return 1;
    }
    //////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////
    // Lettura delle disparita'
    //
    // DO NOT TOUCH
    cv::Mat imgDisparity16U(left_image.rows, left_image.cols, CV_16U, cv::Scalar(0));
    cv::Mat imgDisparityF32(left_image.rows, left_image.cols, CV_32FC1, cv::Scalar(0));

    // Leggiamo la dsi gia' PRECALCOLATA da file
    std::ifstream dsifile(argv[2], std::ifstream::binary);
    if (!dsifile.is_open()) {
        std::cout << "Unable to open " << argv[2] << std::endl;
        return 1;
    }
    dsifile.seekg(0, std::ios::beg);
    dsifile.read((char*)imgDisparity16U.data, imgDisparity16U.rows * imgDisparity16U.cols * 2);
    dsifile.close();

    imgDisparity16U.convertTo(imgDisparityF32, CV_32FC1);
    imgDisparityF32 /= 16.0;
    //////////////////////////////////////////////////////////////////

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

    // vettore dei punti 3D calcolati a partire disparita'
    std::vector<cv::Point3f> points;
    // vettore delle corrispondenti righe,colonne
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

    // vettore delle coordinate (r,c) degli inliers migliori
    std::vector<cv::Point2i> inliers_best_rc;

    computePlane(points, rc, inliers_best, inliers_best_rc);

    /*
     * Creare un'immagine formata dai soli pixel inliers
     *
     * Nella parte di RANSAC precedente dovro' quindi calcolare, oltre ai punti 3D inliers, anche le loro coordinate riga colonna corrispondenti
     *
     * Salvare queste (r,c) nel vettore inliers_best_rc ed utilizzarlo adesso per scrivere l'immagine out con i soli pixel inliers
     */

    // immagine di uscita che conterra' i soli pixel inliers
    cv::Mat out(left_image.rows, left_image.cols, CV_8UC1, cv::Scalar(0));
    /*
     * YOUR CODE HERE
     *
     * Costruzione immagine di soli inliers
     *
     * Si tratta di copia gli inliers dentro out
     */
    /////////////////////////////////////////////////////////////////////
	for (cv::Point2i p : inliers_best_rc)
	{
		out.at<uchar>(p.x, p.y) = left_image.at<uchar>(p.x, p.y);
	}
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // display images
    //
    //  DO NOT TOUCH
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

    // wait for key
    cv::waitKey();
    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    return 0;
}
