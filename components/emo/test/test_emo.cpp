
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <sys/time.h>
#include <fstream>
#include <assert.h>
    
#include "Frame.h"
#include "ORBextractor.h"
#include "camera/inc/PinholeCamera.h"

using namespace std;
using namespace emo;

int main( int argc, char** argv )
{
    struct timeval tv_start_all, tv_end_all;
    gettimeofday(&tv_start_all, NULL);

     // *** Parse and load input images
    char *imgfile_ao = argv[1];
    char *imgfile_bo = argv[2];
    int incoltype = CV_LOAD_IMAGE_GRAYSCALE;
    int rpyrtype = CV_32FC1;
    int nochannels = 1;

    cv::Mat img_ao_mat = cv::imread(imgfile_ao, incoltype);   // Read the file
    cv::Mat img_bo_mat = cv::imread(imgfile_bo, incoltype);   // Read the file
    cv::Mat img_ao_fmat, img_bo_fmat;
    cv::Size sz = img_ao_mat.size();
    const int width_org = sz.width;   // unpadded original image size
    const int height_org = sz.height;  // unpadded original image size

    // init camera
    cv::FileStorage l_Settings( argv[3], cv::FileStorage::READ);
    assert(l_Settings.isOpened());
    float fx, fy, cx, cy;

    // Camera calibration parameters
    cv::FileNode node = l_Settings["Camera.fx"];
    assert(!node.empty() && node.isReal());
    fx = node.real();
    node = l_Settings["Camera.fy"];
    assert(!node.empty() && node.isReal());
    fy = node.real();
    node = l_Settings["Camera.cx"];
    assert(!node.empty() && node.isReal());
    cx = node.real();
    node = l_Settings["Camera.cy"];
    assert(!node.empty() && node.isReal());
    cy = node.real();
    vector<float> vCamCalib{fx,fy,cx,cy};
    auto l_pinholeCamera = new camera::Pinhole(vCamCalib);

    // Distortion parameters
    cv::Mat l_DistCoef;
    node = l_Settings["Camera.k1"];
    assert(!node.empty() && node.isReal());
    l_DistCoef.at<float>(0) = node.real();
    node = l_Settings["Camera.k2"];
    assert(!node.empty() && node.isReal());
    l_DistCoef.at<float>(1) = node.real();
    node = l_Settings["Camera.p1"];
    assert(!node.empty() && node.isReal());
    l_DistCoef.at<float>(2) = node.real();
    node = l_Settings["Camera.p2"];
    assert(!node.empty() && node.isReal());
    l_DistCoef.at<float>(3) = node.real();
    //

    // load ORB setting
    int nFeatures = l_Settings["ORBextractor.nFeatures"];
    float fScaleFactor = l_Settings["ORBextractor.scaleFactor"];
    int nLevels = l_Settings["ORBextractor.nLevels"];
    int fIniThFAST = l_Settings["ORBextractor.iniThFAST"];
    int fMinThFAST = l_Settings["ORBextractor.minThFAST"];
    auto* l_orbExtractor_p = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

//  // *** Parse rest of parameters, See oflow.h for definitions.
//  int lv_f, lv_l, maxiter, miniter, patchsz, patnorm, costfct, tv_innerit, tv_solverit, verbosity;
//  float mindprate, mindrrate, minimgerr, poverl, tv_alpha, tv_gamma, tv_delta, tv_sor;
//  bool usefbcon, usetvref;
//  //bool hasinfile; // initialization flow file
//  //char *infile = nullptr;
//
//  if (argc<=5)  // Use operation point X, set scales automatically
//  {
//    mindprate = 0.05; mindrrate = 0.95; minimgerr = 0.0;
//    usefbcon = 0; patnorm = 1; costfct = 0;
//    tv_alpha = 10.0; tv_gamma = 10.0; tv_delta = 5.0;
//    tv_innerit = 1; tv_solverit = 3; tv_sor = 1.6;
//    verbosity = 2; // Default: Plot detailed timings
//
//    int fratio = 5; // For automatic selection of coarsest scale: 1/fratio * width = maximum expected motion magnitude in image. Set lower to restrict search space.
//
//    int sel_oppoint = 2; // Default operating point
//    if (argc==5)         // Use provided operating point
//      sel_oppoint=atoi(argv[4]);
//
//    switch (sel_oppoint)
//    {
//      case 1:
//        patchsz = 8; poverl = 0.3;
//        lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
//        lv_l = std::max(lv_f-2,0); maxiter = 16; miniter = 16;
//        usetvref = 0;
//        break;
//      case 3:
//        patchsz = 12; poverl = 0.75;
//        lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
//        lv_l = std::max(lv_f-4,0); maxiter = 16; miniter = 16;
//        usetvref = 1;
//        break;
//      case 4:
//        patchsz = 12; poverl = 0.75;
//        lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
//        lv_l = std::max(lv_f-5,0); maxiter = 128; miniter = 128;
//        usetvref = 1;
//        break;
//      case 2:
//      default:
//        patchsz = 8; poverl = 0.4;
//        lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
//        lv_l = std::max(lv_f-2,0); maxiter = 12; miniter = 12;
//        usetvref = 1;
//        break;
//
//    }
//  }
//  else //  Parse explicitly provided parameters
//  {
//    int acnt = 4; // Argument counter
//    lv_f = atoi(argv[acnt++]);
//    lv_l = atoi(argv[acnt++]);
//    maxiter = atoi(argv[acnt++]);
//    miniter = atoi(argv[acnt++]);
//    mindprate = atof(argv[acnt++]);
//    mindrrate = atof(argv[acnt++]);
//    minimgerr = atof(argv[acnt++]);
//    patchsz = atoi(argv[acnt++]);
//    poverl = atof(argv[acnt++]);
//    usefbcon = atoi(argv[acnt++]);
//    patnorm = atoi(argv[acnt++]);
//    costfct = atoi(argv[acnt++]);
//    usetvref = atoi(argv[acnt++]);
//    tv_alpha = atof(argv[acnt++]);
//    tv_gamma = atof(argv[acnt++]);
//    tv_delta = atof(argv[acnt++]);
//    tv_innerit = atoi(argv[acnt++]);
//    tv_solverit = atoi(argv[acnt++]);
//    tv_sor = atof(argv[acnt++]);
//    verbosity = atoi(argv[acnt++]);
//    //hasinfile = (bool)atoi(argv[acnt++]);   // initialization flow file
//    //if (hasinfile) infile = argv[acnt++];
//  }
//
//
//
//  // *** Pad image such that width and height are restless divisible on all scales (except last)
//  int padw=0, padh=0;
//  int scfct = pow(2,lv_f); // enforce restless division by this number on coarsest scale
//  //if (hasinfile) scfct = pow(2,lv_f+1); // if initialization file is given, make sure that size is restless divisible by 2^(lv_f+1) !
//  int div = sz.width % scfct;
//  if (div>0) padw = scfct - div;
//  div = sz.height % scfct;
//  if (div>0) padh = scfct - div;
//  if (padh>0 || padw>0)
//  {
//    copyMakeBorder(img_ao_mat,img_ao_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
//    copyMakeBorder(img_bo_mat,img_bo_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
//  }
//  sz = img_ao_mat.size();  // padded image size, ensures divisibility by 2 on all scales (except last)
//
//  // Timing, image loading
//  if (verbosity > 1)
//  {
//    gettimeofday(&tv_end_all, NULL);
//    double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
//    printf("TIME (Image loading     ) (ms): %3g\n", tt);
//    gettimeofday(&tv_start_all, NULL);
//  }
//
//
//
//
//  //  *** Generate scale pyramides
//  img_ao_mat.convertTo(img_ao_fmat, CV_32F); // convert to float
//  img_bo_mat.convertTo(img_bo_fmat, CV_32F);
//
//  const float* img_ao_pyr[lv_f+1];
//  const float* img_bo_pyr[lv_f+1];
//  const float* img_ao_dx_pyr[lv_f+1];
//  const float* img_ao_dy_pyr[lv_f+1];
//  const float* img_bo_dx_pyr[lv_f+1];
//  const float* img_bo_dy_pyr[lv_f+1];
//
//  cv::Mat img_ao_fmat_pyr[lv_f+1];
//  cv::Mat img_bo_fmat_pyr[lv_f+1];
//  cv::Mat img_ao_dx_fmat_pyr[lv_f+1];
//  cv::Mat img_ao_dy_fmat_pyr[lv_f+1];
//  cv::Mat img_bo_dx_fmat_pyr[lv_f+1];
//  cv::Mat img_bo_dy_fmat_pyr[lv_f+1];
//
//  ConstructImgPyramide(img_ao_fmat, img_ao_fmat_pyr, img_ao_dx_fmat_pyr, img_ao_dy_fmat_pyr, img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);
//  ConstructImgPyramide(img_bo_fmat, img_bo_fmat_pyr, img_bo_dx_fmat_pyr, img_bo_dy_fmat_pyr, img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);
//
//  // Timing, image gradients and pyramid
//  if (verbosity > 1)
//  {
//    gettimeofday(&tv_end_all, NULL);
//    double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
//    printf("TIME (Pyramide+Gradients) (ms): %3g\n", tt);
//  }
//
//
//  //  *** Run main optical flow / depth algorithm
//  float sc_fct = pow(2,lv_l);
//  #if (SELECTMODE==1)
//  cv::Mat flowout(sz.height / sc_fct , sz.width / sc_fct, CV_32FC2); // Optical Flow
//  #else
//  cv::Mat flowout(sz.height / sc_fct , sz.width / sc_fct, CV_32FC1); // Depth
//  #endif
//
//  OFC::OFClass ofc(img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr,
//                    img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr,
//                    patchsz,  // extra image padding to avoid border violation check
//                    (float*)flowout.data,   // pointer to n-band output float array
//                    nullptr,  // pointer to n-band input float array of size of first (coarsest) scale, pass as nullptr to disable
//                    sz.width, sz.height,
//                    lv_f, lv_l, maxiter, miniter, mindprate, mindrrate, minimgerr, patchsz, poverl,
//                    usefbcon, costfct, nochannels, patnorm,
//                    usetvref, tv_alpha, tv_gamma, tv_delta, tv_innerit, tv_solverit, tv_sor,
//                    verbosity);
//
//  if (verbosity > 1) gettimeofday(&tv_start_all, NULL);
//
//
//
//  // *** Resize to original scale, if not run to finest level
//  if (lv_l != 0)
//  {
//    flowout *= sc_fct;
//    cv::resize(flowout, flowout, cv::Size(), sc_fct, sc_fct , cv::INTER_LINEAR);
//  }
//
//  // If image was padded, remove padding before saving to file
//  flowout = flowout(cv::Rect((int)floor((float)padw/2.0f),(int)floor((float)padh/2.0f),width_org,height_org));
//
//  // Save Result Image
//  #if (SELECTMODE==1)
//  SaveFlowFile(flowout, outfile);
//  #else
//  SavePFMFile(flowout, outfile);
//  #endif
//
//  if (verbosity > 1)
//  {
//    gettimeofday(&tv_end_all, NULL);
//    double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
//    printf("TIME (Saving flow file  ) (ms): %3g\n", tt);
//  }
//
//  return 0;
}


    


