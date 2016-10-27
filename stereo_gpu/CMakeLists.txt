cmake_minimum_required(VERSION 2.8)

project(stereo_gpu)

find_package(CUDA REQUIRED)
unset(CUDA_ARCH CACHE)
set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH}")

find_package(OpenCV REQUIRED core)
if(OpenCV_VERSION VERSION_LESS 3.0)
	find_package(OpenCV REQUIRED core highgui imgproc calib3d contrib)  # 2.x
else()
	find_package(OpenCV REQUIRED core highgui imgproc imgcodecs videoio) # 3.x
endif()

find_package(PCL REQUIRED COMPONENTS common io)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  cv_bridge
  image_transport
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  tf
)

catkin_package(
   INCLUDE_DIRS include
   LIBRARIES stereo_gpu
   CATKIN_DEPENDS common_msgs cv_bridge geometry image_transport pcl_ros roscpp sensor_msgs std_msgs tf
   DEPENDS system_lib
)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

SET(DOTCU
	src/census.cu
	src/matching_cost.cu
	src/scan_scost.cu
	src/winner_takes_all.cu
	src/median_filter.cu
	src/check_consistency.cu
	src/cuda_utils.cu
)

CUDA_ADD_LIBRARY(sgm src/stereo_sgm.cpp ${DOTCU} STATIC)

ADD_LIBRARY(nppc_static STATIC IMPORTED)
SET_TARGET_PROPERTIES(nppc_static PROPERTIES
IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/libnppc_static.a)

ADD_LIBRARY(nppi_static STATIC IMPORTED)
SET_TARGET_PROPERTIES(nppi_static PROPERTIES
IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/libnppi_static.a)

CUDA_ADD_EXECUTABLE(stereo_gpu_node ./src/main.cpp )

TARGET_LINK_LIBRARIES(stereo_gpu_node sgm nppc_static nppi_static ${OpenCV_LIBS} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

