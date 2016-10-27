#include "ueye_camera.h"

/********************* my functions ****************************/

// INITIALIZE CAMERA
// -----------------------------------------------------------------------------
bool init_camera(CUeye_Camera& cam)
{
  // Initialize camera
  cout << endl << "[Camera test]: Trying to open camera " + std::to_string(cam.params_.cameraid) << endl;
  try
  {
    cam.init_camera();
  }
  catch (CUeyeCameraException & e)
  {
    cout << e.what () << endl;
    return false;
  }
  catch (CUeyeFeatureException & e)
  {
    cout << e.what () << endl;
  }

  return true;
}

// LIST AVAILABLE CAMERAS
// -----------------------------------------------------------------------------
bool list_cameras(CUeye_Camera& cam)
{
  try
  {
    cam.list_cameras();
  }
  catch (CUeyeCameraException & e)
  {
    cout << e.what () << endl;
    return false;
  }
  catch (CUeyeFeatureException & e)
  {
    cout << e.what () << endl;
  }
  return true;
}


/*
// GET IMAGE
//-----------------------------------------------------------------------------
cv::Mat* get_img(CUeye_Camera& cam)
{
  // Aacquire a single image from the camera
  bool image_ok = false;
  try
  {
    image_ok = cam.get_image();
  }
  catch (CUeyeCameraException & e)
  {
    cout << e.what () << endl;
    return 0;
  }
  catch (CUeyeFeatureException & e)
  {
    cout << e.what () << endl;
  }

  int type;
  if(cam.params_.img_bpp ==8)
	type=CV_8UC1;
  else if(cam.params_.img_bpp ==24 || cam.params_.img_bpp==32)
	type=CV_8UC3;

  cv::Mat* image = NULL;

  if(image_ok)
  {
    image = new cv::Mat(cam.params_.img_height, cam.params_.img_width, type);

    for (int jj = 0; jj < cam.img_data_size_; ++jj)
      image->at<unsigned char>(jj) = (unsigned char)cam.image_data_.at(jj);
  }

  return image;
}
*/


// GET IMAGE for multithread
//-----------------------------------------------------------------------------
void get_img_thread(CUeye_Camera& cam, cv::Mat** ppImage)
{
  // Acquire a single image from the camera
  bool image_ok = false;
  try
  {
    image_ok = cam.get_image();
  }
  catch (CUeyeCameraException & e)
  {
    cout << e.what () << endl;
    return;
  }
  catch (CUeyeFeatureException & e)
  {
    cout << e.what () << endl;
  }

  int type;
  if(cam.params_.img_bpp ==8)
	type=CV_8UC1;
  else if(cam.params_.img_bpp ==24 || cam.params_.img_bpp==32)
	type=CV_8UC3;

  cv::Mat* image = NULL;

  if(image_ok)
  {
    image = new cv::Mat(cam.params_.img_height, cam.params_.img_width, type);

	if (image != NULL){
        for (int i = 0; i < cam.img_data_size_; ++i)
            image->at<unsigned char>(i) = (unsigned char)cam.image_data_.at(i);
    }
  }

  *ppImage = image;

  return;
}



// Capture Image for calibration
//----------------------------------------------------
void image_save(const cv::Mat &img1, const cv::Mat &img2){
string str;
static int count_f = 0;
count_f++;
string result;
ostringstream convert;
convert << count_f;
result = convert.str();
string filename1 = "/home/haibo/Desktop/Calib/Left/" + result + "_1.png";
string filename2 = "/home/haibo/Desktop/Calib/Right/" + result + "_2.png";
cv::imwrite(filename1.c_str(), img1);
cv::imwrite(filename2.c_str(), img2);
return;
}

void setCameraParams(CUeye_Camera& cam){
  static int cameraCount = 1;

  cam.params_.cameraid     		=cameraCount++; // first time 1, second time 2, etc

  cam.params_.exposure              =10;
  cam.params_.img_width             =752;
  cam.params_.img_height            =480;
  cam.params_.img_left      		=-1;
  cam.params_.img_top       		=-1;
  cam.params_.fps           		=20;
  cam.params_.param_mode      		=0;
  cam.params_.file_str       		="";
  cam.params_.pixel_clock   		=20;
  cam.params_.mirror_updown  		=false;
  cam.params_.mirror_leftright 	=false;

//  ros::param::get("~image_width", ueye.params_.img_width);
//  ros::param::get("~image_height", ueye.params_.img_height);
//  ros::param::get("~exposure", ueye.params_.exposure);

}

bool readCalibParams(const string calibFile, cv::Mat& M1l, cv::Mat& M2l, cv::Mat& M1r, cv::Mat& M2r){
  // Load settings related to stereo calibration
  cv::FileStorage fsSettings(calibFile, cv::FileStorage::READ);

  if(!fsSettings.isOpened()){
    cerr << "ERROR: Wrong path to settings" << endl;
    return false;
  }

  cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
  fsSettings["LEFT.K"] >> K_l;
  fsSettings["RIGHT.K"] >> K_r;
  fsSettings["LEFT.P"] >> P_l;
  fsSettings["RIGHT.P"] >> P_r;
  fsSettings["LEFT.R"] >> R_l;
  fsSettings["RIGHT.R"] >> R_r;
  fsSettings["LEFT.D"] >> D_l;
  fsSettings["RIGHT.D"] >> D_r;

  int rows_l = fsSettings["LEFT.height"];
  int cols_l = fsSettings["LEFT.width"];
  int rows_r = fsSettings["RIGHT.height"];
  int cols_r = fsSettings["RIGHT.width"];

  if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
        rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0){
    cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
    return false;
  }

  cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
  cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);
  return true;
}
