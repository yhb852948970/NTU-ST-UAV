#ifndef _my_ueye_functions_H
#define _my_ueye_functions_H

//#include <uEye.h>

/*
 *----------------------------------------------------------------------
 * My own function prototypes
 *----------------------------------------------------------------------
 */
//cv::Mat* get_img(CUeye_Camera& cam); // used in single thread
void get_img_thread(CUeye_Camera& cam, cv::Mat** ppImage); // used in multi-thread
bool init_camera(CUeye_Camera& cam);
bool list_cameras(CUeye_Camera& cam);
void image_save(const cv::Mat &img1, const cv::Mat &img2);
void setCameraParams(CUeye_Camera& cam);
bool readCalibParams(const string calibFile, cv::Mat& M1l, cv::Mat& M2l, cv::Mat& M1r, cv::Mat& M2r);

#endif
