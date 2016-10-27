
#include <stereoGPU.h>

using namespace std;


stereoGPU::stereoGPU(ros::NodeHandle nh_, string calibFile) : nh(nh_), it(nh_), calibFile(calibFile), cloud_last_frame ( new pcl::PointCloud<pcl::PointXYZ>) {

	//img_size = cv::Size(width, height);
  //img_size_cut = cv::Size(width_cut, height_cut);
  //disparity_full = cv::Mat::zeros(height, width, CV_8UC1);
  //disparity_cut = cv::Mat::zeros(height_cut, width_cut, CV_8UC1);
	disparity_pub = it.advertise("/stereo/disparity", 1);
  cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/stereo/cloud_raw", 1);
  //cloud_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZ> >("/stereo/cloud_raw", 1);

  bool rectify = readCalibParams();

  if (!rectify){
    cerr << "Loading of calibration parameters failed, please provide a valid path to the calibration parameters." << endl;
  }
}


stereoGPU::~stereoGPU(){
}


bool stereoGPU::readCalibParams(){
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

	fsSettings["Camera.Q"] >> Q;
	//cout << "Q: " << Q << endl;

  int rows_l = fsSettings["LEFT.height"];
  int cols_l = fsSettings["LEFT.width"];
  int rows_r = fsSettings["RIGHT.height"];
  int cols_r = fsSettings["RIGHT.width"];

  width = fsSettings["Camera.width"];
  height = fsSettings["Camera.height"];
  height_up_cut = fsSettings["Camera.up"];
  height_down_cut = fsSettings["Camera.down"];
  width_left_cut = fsSettings["Camera.left"];
  width_right_cut = fsSettings["Camera.right"];
  height_cut = height - height_up_cut - height_down_cut;
  width_cut = width - width_left_cut - width_right_cut;
	disparity_roi = cv::Rect(width_left_cut, height_up_cut, width_cut, height_cut);

  if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
        rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0){
    cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
    return false;
  }

  cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F, map11, map12);
  cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F, map21, map22);

	// cout << endl << "K_l: " << K_l << endl;
	// cout << endl << "K_r: " << K_r << endl;
	// cout << endl << "D_l: " << D_l << endl;
	// cout << endl << "D_r: " << D_r << endl;
	// cout << endl << "R_l: " << R_l << endl;
	// cout << endl << "R_r: " << R_r << endl;
	// cout << endl << "P_l: " << P_l << endl;
	// cout << endl << "P_r: " << P_r << endl;

  return true;
}


void stereoGPU::process(const sensor_msgs::ImageConstPtr& msg1, const sensor_msgs::ImageConstPtr& msg2){
	cbCounter++;
	//int64 t = cv::getTickCount();
  cv_bridge::CvImagePtr cv_ptr1;	// opencv Mat pointer;
  cv_bridge::CvImagePtr cv_ptr2;

  cv::Mat img_left_raw, img_right_raw;
  cv::Mat img_left_rect, img_right_rect;
  cv::Mat img_left_cut, img_right_cut;
  cv::Mat img_disparity, img_disparity_show;
  img_disparity = cv::Mat::zeros(height_cut, width_cut, CV_8UC1);
  img_disparity_show = cv::Mat::zeros(height_cut, width_cut, CV_8UC1);
  cv::Mat depth_32F;
  //cv::Mat bgr[3];

  try{
        cv_ptr1 = cv_bridge::toCvCopy(msg1, sensor_msgs::image_encodings::TYPE_8UC1);
  }

  catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
  }

  try{
        cv_ptr2 = cv_bridge::toCvCopy(msg2, sensor_msgs::image_encodings::TYPE_8UC1);
  }

  catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
  }

  // cv_ptr->image   ----> Mat image in opencv
  img_left_raw = cv_ptr1->image;
  img_right_raw = cv_ptr2->image;
  // rectify for the full image
  cv::remap(img_left_raw, img_left_rect, map11, map12, cv::INTER_LINEAR);
  cv::remap(img_right_raw, img_right_rect, map21, map22, cv::INTER_LINEAR);

	img_left_cut = img_left_rect(disparity_roi);
	img_right_cut = img_right_rect(disparity_roi);

	sgm::StereoSGM ssgm(width_cut, height_cut, disp_size, sgm::EXECUTE_INOUT_HOST2HOST);
  ssgm.execute(img_left_cut.data, img_right_cut.data, (void**)&img_disparity.data);

  img_disparity.convertTo(img_disparity_show, CV_8U, 255/disp_size);
  cv::reprojectImageTo3D(img_disparity, depth_32F, Q);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr = convert2XYZPointCloud(depth_32F, 30000., 1500.);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_Fil_ptr = cloudFiltering(cloud_ptr);
	cloud_last_frame = cloud_ptr;

  transform.setOrigin( tf::Vector3(0.0, 0.0, 0.0) );
  tf::Quaternion q;
  q.setRPY(0., 0., 0.);
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "pcl_frame"));

  sensor_msgs::ImagePtr disparity_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", img_disparity_show).toImageMsg();
  disparity_pub.publish(disparity_msg);
  //sensor_msgs::ImagePtr depth = cv_bridge::CvImage(std_msgs::Header(), "32FC1",bgr[2]).toImageMsg();
  //img_depth.publish(depth);
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg (*cloud_Fil_ptr, cloud_msg);
  cloud_pub.publish(cloud_msg);

  //	t = cv::getTickCount() - t;
  //ROS_INFO_STREAM("Stereo Matching time: " << t*1000/cv::getTickFrequency() << " miliseconds.");
}


pcl::PointCloud<pcl::PointXYZ>::Ptr stereoGPU::convert2XYZPointCloud(
    const cv::Mat& depth_32F, const double maxDepth, const double minDepth) const{

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
    }// for loop
  }// for loop
    	return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr stereoGPU::cloudFiltering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) const{

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_Filtered (new pcl::PointCloud<pcl::PointXYZ>);
	*cloud_Filtered = *cloud;
	int diff_Z;

	for (int y = 0; y < cloud->height; y++){ //iteration along y on 2D image plane
      for (int x = 0; x < cloud->width; x++){ //iteration along x on 2D image plane
		      if (this->cbCounter > 1 && cloud_last_frame->points[cloud->width*y+x].z != 0 ){
			         diff_Z = abs(cloud_last_frame->points[cloud->width*y+x].z - cloud->points[cloud->width*y+x].z);
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
