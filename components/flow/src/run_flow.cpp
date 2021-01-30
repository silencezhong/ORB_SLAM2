
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "oflow.h"

void ConstructImgPyramide(const cv::Mat & img_ao_fmat, cv::Mat * img_ao_fmat_pyr, cv::Mat * img_ao_dx_fmat_pyr, cv::Mat * img_ao_dy_fmat_pyr, const float ** img_ao_pyr, const float ** img_ao_dx_pyr, const float ** img_ao_dy_pyr, const int lv_f, const int lv_l, const int rpyrtype, const bool getgrad, const int imgpadding, const int padw, const int padh)
{
    for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
    {
      if (i==0) // At finest scale: copy directly, for all other: downscale previous scale by .5
      {
        #if (SELECTCHANNEL==1 | SELECTCHANNEL==3)  // use RGB or intensity image directly
        img_ao_fmat_pyr[i] = img_ao_fmat.clone();
        #elif (SELECTCHANNEL==2)   // use gradient magnitude image as input
        cv::Mat dx,dy,dx2,dy2,dmag;
        cv::Sobel( img_ao_fmat, dx, CV_32F, 1, 0, 3, 1/8.0, 0, cv::BORDER_DEFAULT );
        cv::Sobel( img_ao_fmat, dy, CV_32F, 0, 1, 3, 1/8.0, 0, cv::BORDER_DEFAULT );
        dx2 = dx.mul(dx);
        dy2 = dy.mul(dy);
        dmag = dx2+dy2;
        cv::sqrt(dmag,dmag);
        img_ao_fmat_pyr[i] = dmag.clone();
        #endif
      }
      else
        cv::resize(img_ao_fmat_pyr[i-1], img_ao_fmat_pyr[i], cv::Size(), .5, .5, cv::INTER_LINEAR);
	      
      img_ao_fmat_pyr[i].convertTo(img_ao_fmat_pyr[i], rpyrtype);
	
      if ( getgrad ) 
      {
        cv::Sobel( img_ao_fmat_pyr[i], img_ao_dx_fmat_pyr[i], CV_32F, 1, 0, 3, 1/8.0, 0, cv::BORDER_DEFAULT );
        cv::Sobel( img_ao_fmat_pyr[i], img_ao_dy_fmat_pyr[i], CV_32F, 0, 1, 3, 1/8.0, 0, cv::BORDER_DEFAULT );
        img_ao_dx_fmat_pyr[i].convertTo(img_ao_dx_fmat_pyr[i], CV_32F);
        img_ao_dy_fmat_pyr[i].convertTo(img_ao_dy_fmat_pyr[i], CV_32F);
      }
    }
    
    // pad images
    for (int i=0; i<=lv_f; ++i)  // Construct image and gradient pyramides
    {
      copyMakeBorder(img_ao_fmat_pyr[i],img_ao_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_REPLICATE);  // Replicate border for image padding
      img_ao_pyr[i] = (float*)img_ao_fmat_pyr[i].data;

      if ( getgrad ) 
      {
        copyMakeBorder(img_ao_dx_fmat_pyr[i],img_ao_dx_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT , 0); // Zero padding for gradients
        copyMakeBorder(img_ao_dy_fmat_pyr[i],img_ao_dy_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT , 0);

        img_ao_dx_pyr[i] = (float*)img_ao_dx_fmat_pyr[i].data;
        img_ao_dy_pyr[i] = (float*)img_ao_dy_fmat_pyr[i].data;      
      }
    }
}

int AutoFirstScaleSelect(int imgwidth, int fratio, int patchsize)
{
  return std::max(0,(int)std::floor(log2((2.0f*(float)imgwidth) / ((float)fratio * (float)patchsize))));
}

int main( int argc, char** argv )
{
    // run  "./test in1.png in2.png out.flo 5 3 12 12 0.05 0.95 0 8 0.40 0 1 0 1 10 10 5 1 3 1.6 2" to test the code
  // *** Parse and load input images
  char *f_imgfile_ao_p = argv[1];
  char *f_imgfile_bo_p = argv[2];

  cv::Mat img_ao_mat, img_bo_mat, img_tmp;
  int rpyrtype, nochannels, incoltype;
  #if (SELECTCHANNEL==1 | SELECTCHANNEL==2) // use Intensity or Gradient image      
  incoltype = CV_LOAD_IMAGE_GRAYSCALE;        
  rpyrtype = CV_32FC1;
  nochannels = 1;
  #elif (SELECTCHANNEL==3) // use RGB image
  incoltype = CV_LOAD_IMAGE_COLOR;
  rpyrtype = CV_32FC3;
  nochannels = 3;      
  #endif
  img_ao_mat = cv::imread(f_imgfile_ao_p, incoltype);   // Read the file
  img_bo_mat = cv::imread(f_imgfile_bo_p, incoltype);   // Read the file
  cv::Mat img_ao_fmat, img_bo_fmat;
  cv::Size sz = img_ao_mat.size();
  int width_org = sz.width;   // unpadded original image size
  int height_org = sz.height;  // unpadded original image size

  // *** Parse rest of parameters, See oflow.h for definitions.
  int lv_f, lv_l, maxiter, miniter, patchsz, patnorm, costfct, tv_innerit, tv_solverit, verbosity;
  float mindprate, mindrrate, minimgerr, poverl, tv_alpha, tv_gamma, tv_delta, tv_sor;
  bool usefbcon, usetvref;

  if (argc<=5)  // Use operation point X, set scales automatically
  {
    mindprate = 0.05; mindrrate = 0.95; minimgerr = 0.0;    
    usefbcon = 0; patnorm = 1; costfct = 0; 
    tv_alpha = 10.0; tv_gamma = 10.0; tv_delta = 5.0;
    tv_innerit = 1; tv_solverit = 3; tv_sor = 1.6;
    verbosity = 2; // Default: Plot detailed timings
        
    int fratio = 5; // For automatic selection of coarsest scale: 1/fratio * width = maximum expected motion magnitude in image. Set lower to restrict search space.
    
    int sel_oppoint = 2; // Default operating point
    if (argc==5)         // Use provided operating point
      sel_oppoint=atoi(argv[4]);
      
    switch (sel_oppoint)
    {
      case 1:
        patchsz = 8; poverl = 0.3; 
        lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
        lv_l = std::max(lv_f-2,0); maxiter = 16; miniter = 16; 
        usetvref = 0; 
        break;
      case 3:
        patchsz = 12; poverl = 0.75; 
        lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
        lv_l = std::max(lv_f-4,0); maxiter = 16; miniter = 16; 
        usetvref = 1; 
        break;
      case 4:
        patchsz = 12; poverl = 0.75; 
        lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
        lv_l = std::max(lv_f-5,0); maxiter = 128; miniter = 128; 
        usetvref = 1; 
        break;        
      case 2:
      default:
        patchsz = 8; poverl = 0.4; 
        lv_f = AutoFirstScaleSelect(width_org, fratio, patchsz);
        lv_l = std::max(lv_f-2,0); maxiter = 12; miniter = 12; 
        usetvref = 1; 
        break;
    }
  }
  else //  Parse explicitly provided parameters
  {
    int acnt = 4; // Argument counter
    lv_f = atoi(argv[acnt++]);
    lv_l = atoi(argv[acnt++]);
    maxiter = atoi(argv[acnt++]);
    miniter = atoi(argv[acnt++]);
    mindprate = atof(argv[acnt++]);
    mindrrate = atof(argv[acnt++]);
    minimgerr = atof(argv[acnt++]);
    patchsz = atoi(argv[acnt++]);
    poverl = atof(argv[acnt++]);
    usefbcon = atoi(argv[acnt++]);
    patnorm = atoi(argv[acnt++]);
    costfct = atoi(argv[acnt++]);
    usetvref = atoi(argv[acnt++]);
    tv_alpha = atof(argv[acnt++]);
    tv_gamma = atof(argv[acnt++]);
    tv_delta = atof(argv[acnt++]);
    tv_innerit = atoi(argv[acnt++]);
    tv_solverit = atoi(argv[acnt++]);
    tv_sor = atof(argv[acnt++]);    
    verbosity = atoi(argv[acnt++]);
  }
  
  // *** Pad image such that width and height are restless divisible on all scales (except last)
  int padw=0, padh=0;
  int scfct = pow(2,lv_f); // enforce restless division by this number on coarsest scale
  int div = sz.width % scfct;
  if (div>0) padw = scfct - div;
  div = sz.height % scfct;
  if (div>0) padh = scfct - div;          
  if (padh>0 || padw>0)
  {
    copyMakeBorder(img_ao_mat,img_ao_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
    copyMakeBorder(img_bo_mat,img_bo_mat,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
  }
  sz = img_ao_mat.size();  // padded image size, ensures divisibility by 2 on all scales (except last)
  
  //  *** Generate scale pyramides
  img_ao_mat.convertTo(img_ao_fmat, CV_32F); // convert to float
  img_bo_mat.convertTo(img_bo_fmat, CV_32F);
  
  const float* img_ao_pyr[lv_f+1];
  const float* img_bo_pyr[lv_f+1];
  const float* img_ao_dx_pyr[lv_f+1];
  const float* img_ao_dy_pyr[lv_f+1];
  const float* img_bo_dx_pyr[lv_f+1];
  const float* img_bo_dy_pyr[lv_f+1];
  
  cv::Mat img_ao_fmat_pyr[lv_f+1];
  cv::Mat img_bo_fmat_pyr[lv_f+1];
  cv::Mat img_ao_dx_fmat_pyr[lv_f+1];
  cv::Mat img_ao_dy_fmat_pyr[lv_f+1];
  cv::Mat img_bo_dx_fmat_pyr[lv_f+1];
  cv::Mat img_bo_dy_fmat_pyr[lv_f+1];
  
  ConstructImgPyramide(img_ao_fmat, img_ao_fmat_pyr, img_ao_dx_fmat_pyr, img_ao_dy_fmat_pyr, img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);
  ConstructImgPyramide(img_bo_fmat, img_bo_fmat_pyr, img_bo_dx_fmat_pyr, img_bo_dy_fmat_pyr, img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, lv_f, lv_l, rpyrtype, 1, patchsz, padw, padh);

  //  *** Run main optical flow / depth algorithm
  float sc_fct = pow(2,lv_l);
  #if (SELECTMODE==1)
  cv::Mat flowout(sz.height / sc_fct , sz.width / sc_fct, CV_32FC2); // Optical Flow
  #else
  cv::Mat flowout(sz.height / sc_fct , sz.width / sc_fct, CV_32FC1); // Depth
  #endif       
  
  OFC::OFClass ofc(img_ao_pyr, img_ao_dx_pyr, img_ao_dy_pyr, 
                    img_bo_pyr, img_bo_dx_pyr, img_bo_dy_pyr, 
                    patchsz,  // extra image padding to avoid border violation check
                    (float*)flowout.data,   // pointer to n-band output float array
                    nullptr,  // pointer to n-band input float array of size of first (coarsest) scale, pass as nullptr to disable
                    sz.width, sz.height, 
                    lv_f, lv_l, maxiter, miniter, mindprate, mindrrate, minimgerr, patchsz, poverl, 
                    usefbcon, costfct, nochannels, patnorm, 
                    usetvref, tv_alpha, tv_gamma, tv_delta, tv_innerit, tv_solverit, tv_sor,
                    verbosity);    

  // *** Resize to original scale, if not run to finest level
  if (lv_l != 0)
  {
    flowout *= sc_fct;
    cv::resize(flowout, flowout, cv::Size(), sc_fct, sc_fct , cv::INTER_LINEAR);
  }
  
  // If image was padded, remove padding before saving to file
  flowout = flowout(cv::Rect((int)floor((float)padw/2.0f),(int)floor((float)padh/2.0f),width_org,height_org));

  return 0;
}


    


