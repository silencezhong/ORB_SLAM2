cmake_minimum_required (VERSION 2.8) 
project (emo)

set (CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unknown-pragmas -Wall -std=c++11 -msse4")  #-Wall
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wno-unknown-pragmas -Wall -msse4")  #-Wall
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

set (EMO_LIB_TYPE SHARED)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

##find opencv
FIND_PACKAGE(OpenCV REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
# # # UNCOMMENT THIS IF YOU WANT TO USE OPENMP PARALLELIZATION
# add_definitions(-DWITH_OPENMP=true)
# FIND_PACKAGE( OpenMP REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
set(SRC_LISTS
        src/Converter.cc
        src/Frame.cc
        src/Initializer.cc
        src/KeyFrame.cc
        src/Map.cc
        src/MapPoint.cc
        src/Optimizer.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/EgoMotion.cpp)

#generate lib
ADD_LIBRARY(${PROJECT_NAME} ${EMO_LIB_TYPE} ${SRC_LISTS})

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${PROJECT_SOURCE_DIR}/../../Thirdparty/g2o/lib/libg2o.so
        ${PROJECT_SOURCE_DIR}/../camera/lib/libcamera.so)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/inc
        ${PROJECT_SOURCE_DIR}/../
        ${PROJECT_SOURCE_DIR}/../../
        ${EIGEN3_INCLUDE_DIR})

# Load two images, cacluate ORB feature and correspondency
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
set(SOURCE_FILES ${PROJECT_SOURCE_DIR}/test/test_emo.cpp)
add_executable(test test/test_emo.cpp)
target_link_libraries(test
        ${PROJECT_NAME}
        ${PROJECT_SOURCE_DIR}/../camera/lib/libcamera.so)
target_include_directories(test PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/inc
        ${PROJECT_SOURCE_DIR}/../)

#unit test
add_executable(unit_test test/unit_test.cpp)
target_link_libraries(unit_test
        gtest
        ${PROJECT_NAME}
        ${PROJECT_SOURCE_DIR}/../camera/lib/libcamera.so)
target_include_directories(unit_test PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/inc
        ${PROJECT_SOURCE_DIR}/../)



