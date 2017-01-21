cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#add_definitions(-DCOMPILEDWITHC11)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/ORB_SLAM2/cmake_modules)

find_package(OpenCV 2.4.10 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  tf
)

#catkin_destinations()
catkin_package()

include_directories(
${catkin_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/ORB_SLAM2/
${PROJECT_SOURCE_DIR}/ORB_SLAM2/include
${Pangolin_INCLUDE_DIRS}
)

set(LIBS 
${catkin_LIBRARIES}
${OpenCV_LIBS} 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/ORB_SLAM2/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/ORB_SLAM2/Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/ORB_SLAM2/lib/libORB_SLAM2.so
)

# Node for stereo camera
add_executable(ORB_SLAM2_node
src/ros_stereo.cc
)

target_link_libraries(ORB_SLAM2_node
${LIBS}
)

