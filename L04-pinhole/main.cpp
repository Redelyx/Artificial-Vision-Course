/*L04-PINHOLE-SOLVED*/
// std
#include <iostream>
#include <fstream>

// opencv
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>

// eigen
#include <eigen3/Eigen/Core>

// utils
#include "utils.h"

using namespace cv;

void Project(const std::vector<cv::Point3f>& points, const CameraParams& params, std::vector<cv::Point2f>& uv_points);

std::string im_win_name = "Image";
std::string im_win_name_loop = "Image_loop";
std::string im_win_name_spiral = "Image_spiral";

int main(int argc, char **argv) {

  if (argc < 3) 
  {
    std::cerr << "Usage lab4 <points_filename> <camera_params_filename>" << std::endl;
    return 0;
  }

  // load point cloud from file
  std::vector<cv::Point3f> points;
  LoadPoints(argv[1], points);

  // load camera params from file
  CameraParams params;
  LoadCameraParams(argv[2], params);

#ifdef USE_OPENCVVIZ
  cv::Mat cloud;
  PointsToMat(points, cloud);

  // 3d visualization
  cv::viz::Viz3d win = Viz3D(params);

  win.showWidget("cloud", cv::viz::WCloud(cloud));

  std::cout << "Press q to exit" << std::endl;
  win.spin();
#endif

  // project 3d points on image
  std::vector<Point2f> uv_points;

  Project(points, params, uv_points);

  // draw image
  Mat image;
  image = Mat::zeros(params.h, params.w, CV_32FC1);
  DrawPixels(uv_points, image);

  namedWindow(im_win_name, WINDOW_AUTOSIZE );
  imshow(im_win_name, image);
  waitKey(0);

  // rotazione intorno all'edificio
  //
  // Provare ad implementare un loop di 16 posizioni sul piano XZ equidistanti dal baricentro dell'edificio, raggio 30m.
  // Per mantenere l'edificio al centro della visuale dobbiamo ruotare l'orientazione della camera di (2*M_PI/16) ad ogni step
  //
  //
  // centro del palazzo sul piano XZ, Y costante

  //baricentro dell'edificio ad altezza fissata
  float bx=0.0, by=-5.0, bz=0;

  for(unsigned int i=0;i<points.size();++i)
  {
    bx+=points[i].x;
    bz+=points[i].z;
  }

  bx/=points.size();
  bz/=points.size();

  std::cout<<"Building center "<<bx<<" "<<by<<" "<<bz<<std::endl;


  // L'idea e' di muoversi lungo una circonferenza di raggio radius e centrato nel baricentro dell'edificio
  //
  // Supponiamo di volerci spostare su 16 posizioni equidistanti, possiamo utilizzare una variable angle che fa da
  // 0 a 2PI per step costanti (2*M_PI/16), e quindi calcolare la deltaX e deltaZ con seno e coseno.
  //
  float radius = 30.0;
  float angle=0.0;
  int steps = 16;
  namedWindow(im_win_name_loop, WINDOW_AUTOSIZE );
  int i = 0;
  while(1)
  {
    /**
     * YOUR CODE HERE:
     *
     * Calcolare i params opportuni per spostare il punto di vista lungo la circonferenza
     * mantenendo l'orientazione che punti verso l'edificio
     *
     * Utilizzare la funzione PoseToAffine fornita per calcolare i nuovi params
     */

    float rx, ry, rz, tx, ty, tz;

    tx = bx - sin(angle)*radius;
    ty = by; //non modifichiamo l'altezza
    tz = bz - cos(angle)*radius;

    rx = 0;      //nessuna rotazione intorno a X
    ry = angle;  //rotazione intorno ad Y per orientarci verso l'edificio
    rz = 0;      //nessuna rotazione intorno a Z

    PoseToAffine(rx, ry ,rz, tx, ty, tz, params.RT);

    // project 3d points on image
    uv_points.clear();
    Project(points, params, uv_points);

    // draw image
    Mat image_loop;
    image_loop = Mat::zeros(params.h, params.w, CV_32FC1);
    DrawPixels(uv_points, image_loop);

    imshow(im_win_name_loop, image_loop);
    waitKey(200);

    angle+=2*M_PI/float(steps);

    ++i;
  }

  return 0;
}

/*	- const std::vector< Point3f >& points - sono le coordinate euclidee del punto 3D letto dal file
	- const CameraParams& params - parametri della camera
	- std::vector< Point2f >& uv_points - punto 2D in coordinate euclidee
*/
void Project(const std::vector< Point3f >& points, const CameraParams& params, std::vector< Point2f >& uv_points)
{
  Affine3f RT_inv = params.RT.inv(); // attenzione: nei parametri di calibrazione c'e' orientazione e posizione della camera rispetto al mondo, quindi la RT che otteniamo a partire da quelli punti camera in punti mondo

  // inv() non fa una vera inversa ma inverte la trasformazione 

  Eigen::Matrix<float, 4, 4> RT;
  RT << RT_inv.matrix(0,0), RT_inv.matrix(0,1), RT_inv.matrix(0,2), RT_inv.matrix(0,3), 
     RT_inv.matrix(1,0), RT_inv.matrix(1,1), RT_inv.matrix(1,2), RT_inv.matrix(1,3), 
     RT_inv.matrix(2,0), RT_inv.matrix(2,1), RT_inv.matrix(2,2), RT_inv.matrix(2,3),
     0,                  0,                  0,                  1;

  Eigen::Matrix<float, 3, 4> K;
  K << params.ku,         0, params.u0, 0,
               0, params.kv, params.v0, 0,
               0,         0,         1, 0;

  /**
   * YOUR CODE HERE: project points from 3D to 2D
   * hint: p' = K*RT*P'
   */

  uv_points.resize(points.size());
  for (unsigned int i = 0; i < points.size(); ++i)
  {
    Eigen::Vector4f point; //coordinate omogenee punto 3D
    //trasformazione del punto 3D da coo. euclidee a omogenee
    point.x() = points[i].x;
    point.y() = points[i].y;
    point.z() = points[i].z;
    point.w() = 1.0;

    Eigen::Vector3f uv_point; //coordinate omogenee punto 2D
    uv_point = K * RT * point;

	//trasformazione del punto 2D da coo. omogenee a euclidee
    uv_points[i].x = uv_point.x() / uv_point.z(); 
    uv_points[i].y = uv_point.y() / uv_point.z();
  }
}


