/*
	GPU SGM stereo matching with PCL in ROS
*/

#ifndef _STEREOGPU_H
#define _STEREOGPU_H

#include "libsgm.h"
#include "internal.h"

// ROS headers
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Quaternion.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/ros/conversions.h>
#include <pcl_ros/point_cloud.h>  // for the conversion

//OpenCV headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/contrib/contrib.hpp>

//PCL headers
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
//#include <pcl/filters/statistical_outlier_removal.h>

// system headers
#include <math.h>
#include <iterator>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <stdio.h>

using namespace std;

class stereoGPU
{
private:
  ros::NodeHandle nh;
  image_transport::ImageTransport it;
  image_transport::Publisher disparity_pub;
  ros::Publisher cloud_pub;

// original image size
  int height, width;
// image size after cutting
  int height_cut, width_cut;
// amount of pixels cut from the up and bottom
  int height_up_cut;
  int height_down_cut;
  int width_left_cut;
  int width_right_cut;

  //cv::Size img_size;
  //cv::Size img_size_cut;
  cv::Rect disparity_roi;
  //cv::Mat  disparity;

  int disp_size = 64;
  int cbCounter = 0;
  string calibFile;
  cv::Mat map11, map12, map21, map22;
  cv::Mat Q;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_last_frame;

  tf::TransformBroadcaster br;
  tf::Transform transform;

  /*
    Function to read the stereo cameras calibration parameters from a yaml file
    calibFile yaml file where the calibration parameters are stored.
    map11, map12, map21, map22 for remapping
  */
  //bool readCalibParams(const string calibFile, cv::Mat& m11, cv::Mat& map12, cv::Mat& map21, cv::Mat& map22);
  bool readCalibParams();


  /*
    Function that converting a depth map to pointcloud msg
  */
  pcl::PointCloud<pcl::PointXYZ>::Ptr convert2XYZPointCloud (const cv::Mat& depth_32F, const double maxDepth, const double minDepth) const;


  /*
    Function that filtering the row pointcloud, which made it sparse but with less noise
  */
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) const;


public:
  stereoGPU(ros::NodeHandle nh_, string calibFile);
  ~stereoGPU();

  /*
    Function actually do the stereo matching
    msg1: left image
    msg2: right image
  */
  void process(const sensor_msgs::ImageConstPtr& msg1, const sensor_msgs::ImageConstPtr& msg2);

};	// for the class

#endif
