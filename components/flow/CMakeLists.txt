cmake_minimum_required (VERSION 2.8) 
project (flow)

set (CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-unknown-pragmas -Wall -std=c++11 -msse4")  #-Wall
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wno-unknown-pragmas -Wall -msse4")  #-Wall
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

set (FLOW_LIB_TYPE SHARED)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

##find opencv
FIND_PACKAGE(OpenCV REQUIRED)

##find eigen
SET(EIGEN3_INCLUDE_DIR ${FLOW_EIGEN3_INCLUDE})
FIND_PACKAGE(Eigen3 3.1.0 REQUIRED)
IF(EIGEN3_FOUND)
  SET(FLOW_EIGEN3_INCLUDE ${EIGEN3_INCLUDE_DIR} CACHE PATH "Directory of Eigen3")
ELSE(EIGEN3_FOUND)
  SET(FLOW_EIGEN3_INCLUDE "" CACHE PATH "Directory of Eigen3")
ENDIF(EIGEN3_FOUND)
MESSAGE("Eigen include DIR ${FLOW_EIGEN3_INCLUDE}")

# # # UNCOMMENT THIS IF YOU WANT TO USE OPENMP PARALLELIZATION
# add_definitions(-DWITH_OPENMP=true)
# FIND_PACKAGE( OpenMP REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# # 
# # ENABLE PARALLEL FLOW AGGREGATION, CAN LEAD TO DATA RACES, BUT IN PRACTICE HAS ONLY A WEAK NEGATIVE EFFECT ON THE RESULT, [affects only PatGridClass::AggregateFlowDense() ]
# # add_definitions(-DUSE_PARALLEL_ON_FLOWAGGR)
INCLUDE_DIRECTORIES(
        ${flow_SOURCE_DIR}/inc
        ${FLOW_EIGEN3_INCLUDE})

set(SRC_LISTS
        src/oflow.cpp
        src/run_flow.cpp
        src/patchgrid.cpp
        src/refine_variational.cpp
        src/image.c
        src/patch.cpp
        src/opticalflow_aux.c
        src/solver.c)

#generate lib
ADD_LIBRARY(${PROJECT_NAME} ${FLOW_LIB_TYPE} ${SRC_LISTS})

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
)
set_target_properties (${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "SELECTMODE=1")
set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY COMPILE_DEFINITIONS "SELECTCHANNEL=1") # use grey-valued image

# GrayScale, Optical Flow
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
add_executable(test test/test_flow.cpp)
set_target_properties (test PROPERTIES COMPILE_DEFINITIONS "SELECTMODE=1")
set_property(TARGET test APPEND PROPERTY COMPILE_DEFINITIONS "SELECTCHANNEL=1") # use grey-valued image
target_link_libraries(test ${PROJECT_NAME})



