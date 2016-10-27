
//#include <typeinfo>

#include "libsgm.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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


#define WIDTH 752
#define HEIGHT 480

double r_data[9] = {	0.999869085748125, -0.005390380207778, 0.015256315624197, 0.005380773755080, 0.999985298644059, 6.706486037032410e-04, -0.015259706386630, -5.884700235331059e-04, 0.999883390733152};
double t_data[3] = {-2.997493527781123e+02,-0.652091851538177,2.153642301837196};
double M_1[9] = {4.229179410595286e+02, 0.014313559116982, 3.657902748321415e+02, 0, 4.245942224546713e+02, 2.378417928424143e+02, 0, 0, 1 };
double M_2[9] = {4.243355006950268e+02, 0.162619548606052, 3.762657866679956e+02, 0, 4.261813872638455e+02, 2.361768388068374e+02, 0, 0, 1};
double D_1[5] = {-0.310266312281967,0.123131099544115,5.140978207209692e-04,4.359861887949068e-04, -0.026795203831160};
double D_2[5] = {-0.317295992415919,0.135591053622665, 4.095224889332796e-05,3.238947661482718e-04, -0.032648223811825};

class stereoGPU
{
  ros::NodeHandle nh;
  image_transport::ImageTransport it;
  image_transport::Subscriber img_combine;
  image_transport::Publisher img_disparity;
  image_transport::Publisher img_depth;
  ros::Publisher cloud_pub;

  cv::Size img_size;
  cv::Rect roi_left;
  cv::Rect roi_right;
  cv::Rect roi_half;

  cv::Mat img_left, img_right;
  cv::Mat img_left_rect, img_right_rect;
  cv::Mat img_left_half, img_right_half;

  cv::Mat M1, D1, M2, D2;
  cv::Mat R, T;

  cv::Mat map11, map12, map21, map22;
  cv::Mat depth_32F;
  cv::Mat bgr[3];

  cv::Rect roi1, roi2;
  cv::Mat Q;
  cv::Mat R1, P1, R2, P2;
  cv::Mat output, output_show;

  int disp_size;
  float depth_center;

  tf::TransformBroadcaster br;
  tf::Transform transform;

  int cbCounter;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_old;

int count_imwrite;

  public:
  stereoGPU() : 	it(nh) {
	//output = cv::Mat::zeros(HEIGHT/2,WIDTH, CV_8UC1); // half image
	output = cv::Mat::zeros(HEIGHT,WIDTH, CV_8UC1); // original image
	roi_left = cv::Rect(0, 0, WIDTH, HEIGHT);
	roi_right = cv::Rect(WIDTH, 0, WIDTH, HEIGHT);
	roi_half = cv::Rect(0, HEIGHT/4, WIDTH, HEIGHT/2);
	img_size = cv::Size(WIDTH, HEIGHT);

  	M1= cv::Mat(3, 3, CV_64FC1, &M_1);
  	M2 = cv::Mat(3, 3, CV_64FC1, &M_2);
  	D1 = cv::Mat(1, 4, CV_64FC1, &D_1);
  	D2 = cv::Mat(1, 4, CV_64FC1, &D_2);
  	R = cv::Mat(3, 3, CV_64FC1, &r_data);
  	T = cv::Mat(3, 1, CV_64FC1, &t_data);

	disp_size = 64;
	//depth_center = 0;
	count_imwrite = 0;

	cv::stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0, img_size, &roi1, &roi2 );
	cv::initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
	cv::initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
	//std::cout << std::endl << "Q = " << Q << std::endl;
	//std::cout << std::endl << "R1 = " << R1 << std::endl;
	//std::cout << std::endl << "R2 = " << R2 << std::endl;
	//std::cout << std::endl << "P1 = " << P1 << std::endl;
	//std::cout << std::endl << "P2 = " << P2 << std::endl;

	img_combine = it.subscribe("/camera/image_combine", 1, &stereoGPU::imageCallback, this);
	img_disparity = it.advertise("/stereo/disparity", 1);
	img_depth = it.advertise("/stereo/depth", 1);
  	cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/stereo/cloud_raw", 1);
  	//cloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("/stereo/cloud_raw", 1);
	cbCounter = 0;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_old ( new pcl::PointCloud<pcl::PointXYZ>);
  }

  ~stereoGPU(){
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr
  convert2XYZPointCloud (const cv::Mat& depth_32F, const double maxDepth, const double minDepth) const
  {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    	cloud->height = depth_32F.rows;
    	cloud->width = depth_32F.cols;
    	cloud->is_dense = false;
    	cloud->header.frame_id = "pcl_frame";

	//cv::Mat diff_Z = cv::Mat::zeros(cloud->height, cloud->width, CV_8UC1) ;


   	 pcl::PointXYZ point;
   	 cv::Point3f pointOcv;
    	//cloud->points.resize (cloud->height * cloud->width);

    	for (int y = 0; y < cloud->height; y++){ //iteration along y on 2D image plane
      	for (int x = 0; x < cloud->width; x++){ //iteration along x on 2D image plane

		pointOcv = depth_32F.at<cv::Point3f>(y, x);

            	if (pointOcv.z < maxDepth && pointOcv.z > minDepth){
              		point.x = (float)pointOcv.x / 1000;
              		point.y = (float)pointOcv.y / 1000;
              		point.z = (float)pointOcv.z / 1000;
            	}

            	else{
              		point.x = 0.0;
              		point.y = 0.0;
              		point.z = 0.0;
            	}

            	cloud->points.push_back(point);

            	//std::cout << cloud->points.size() << std::endl;


     	}
    	}
    	return cloud;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) const{

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_Filtered (new pcl::PointCloud<pcl::PointXYZ>);
	*cloud_Filtered = *cloud;
	int diff_Z;

	for (int y = 0; y < cloud->height; y++){ //iteration along y on 2D image plane
      	for (int x = 0; x < cloud->width; x++){ //iteration along x on 2D image plane
		if (this->cbCounter > 1 && cloud_old->points[cloud->width*y+x].z != 0 ){
			diff_Z = abs(cloud_old->points[cloud->width*y+x].z - cloud->points[cloud->width*y+x].z);
			if (diff_Z > 2){
				cloud_Filtered->points[cloud->width*y+x].x = 0;
				cloud_Filtered->points[cloud->width*y+x].y = 0;
				cloud_Filtered->points[cloud->width*y+x].z = 0;

			}
		}
		else {
			cloud_Filtered->points[cloud->width*y+x].x = 0;
			cloud_Filtered->points[cloud->width*y+x].y = 0;
			cloud_Filtered->points[cloud->width*y+x].z = 0;
		}
	}
	}
	return cloud_Filtered;
  }


  void imageCallback(const sensor_msgs::ImageConstPtr& msg){
	cbCounter++;
	//int64 t = cv::getTickCount();
     	cv_bridge::CvImagePtr cv_ptr;	// opencv Mat pointer;

      	try{
        	cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC1);
      	}

      	catch (cv_bridge::Exception& e) {
      	  	ROS_ERROR("cv_bridge exception: %s", e.what());
          	return;
    	}

      	// cv_ptr->image   ----> Mat image in opencv
      	img_left = cv_ptr->image(roi_left);
      	img_right = cv_ptr->image(roi_right);
      	cv::remap(img_left, img_left_rect, map11, map12, cv::INTER_LINEAR);
      	cv::remap(img_right, img_right_rect, map21, map22, cv::INTER_LINEAR);

	/* use half image  */
	//img_left_half = img_left_rect(roi_half);
      	//img_right_half = img_right_rect(roi_half);

	/* use full image  */
	img_left_half = img_left_rect;
	img_right_half = img_right_rect;

/* //imwrite to save rectified images
count_imwrite++;
std::string Result;
std::ostringstream convert;
convert << count_imwrite;
Result = convert.str();
std::string filename1 = "/home/haibo/Desktop/Rectify/Left/" + Result + "_1.png";
std::string filename2 = "/home/haibo/Desktop/Rectify/Right/" + Result + "_2.png";
cv::imwrite(filename1.c_str(), img_left_half);
cv::imwrite(filename2.c_str(), img_right_half);
//end of imwrite */

	sgm::StereoSGM ssgm(img_left_half.cols, img_left_half.rows, disp_size, sgm::EXECUTE_INOUT_HOST2HOST);
       	ssgm.execute(img_left_half.data, img_right_half.data, (void**)&output.data);

       	output.convertTo(output_show, CV_8U, 255/disp_size);
       	cv::reprojectImageTo3D(output, depth_32F, Q);
       	//cv::split(depth_32F, bgr);
       	//bgr[2] = bgr[2] / 1000;
       	//depth_center = bgr[2].at<float>(HEIGHT/4, WIDTH/2);
       	//ROS_INFO_STREAM("depth of the center point is " << depth_center << " m." );

       pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr = convert2XYZPointCloud(depth_32F, 30000., 1500);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_Fil_ptr = cloudFiltering(cloud_ptr);
	this->cloud_old = cloud_ptr;

       transform.setOrigin( tf::Vector3(0.0, 0.0, 0.0) );
       tf::Quaternion q;
       q.setRPY(0., 0., 0.);
       transform.setRotation(q);
       br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "pcl_frame"));

       sensor_msgs::ImagePtr disparity = cv_bridge::CvImage(std_msgs::Header(), "mono8", output_show).toImageMsg();
       img_disparity.publish(disparity);
       sensor_msgs::ImagePtr depth = cv_bridge::CvImage(std_msgs::Header(), "32FC1",bgr[2]).toImageMsg();
       img_depth.publish(depth);
       sensor_msgs::PointCloud2 cloud_msg;
       pcl::toROSMsg (*cloud_Fil_ptr, cloud_msg);
       cloud_pub.publish(cloud_msg);

       //	t = cv::getTickCount() - t;
       //ROS_INFO_STREAM("Stereo Matching time: " << t*1000/cv::getTickFrequency() << " miliseconds.");

  }	// for imageCallback
};	// for the class


int main(int argc, char** argv){

	ros::init(argc, argv, "stereo_gpu");	// The third argument is the cpp filename
	stereoGPU stereo;
	ros::spin();
	return 0;
}
