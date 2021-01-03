
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <sys/time.h>
#include <fstream>
    
#include "oflow.h"


using namespace std;
#define MAXWHEELCOLS 60

void setcols(int* colorwheel, int r, int g, int b, int k)
{
    colorwheel[k*3+0] = r;
    colorwheel[k*3+1] = g;
    colorwheel[k*3+2] = b;
}

int makecolorwheel(int* colorwheel)
{
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    int ncols = RY + YG + GC + CB + BM + MR;
    //printf("ncols = %d\n", ncols);
    if (ncols > MAXWHEELCOLS){
        printf("Too Many Columns in ColorWheel!\n");
        //exit(1);
    }
    int i;
    int k = 0;
    for (i = 0; i < RY; i++) setcols(colorwheel, 255,	   255*i/RY,	 0,	       k++);
    for (i = 0; i < YG; i++) setcols(colorwheel, 255-255*i/YG, 255,		 0,	       k++);
    for (i = 0; i < GC; i++) setcols(colorwheel, 0,		   255,		 255*i/GC,     k++);
    for (i = 0; i < CB; i++) setcols(colorwheel, 0,		   255-255*i/CB, 255,	       k++);
    for (i = 0; i < BM; i++) setcols(colorwheel, 255*i/BM,	   0,		 255,	       k++);
    for (i = 0; i < MR; i++) setcols(colorwheel, 255,	   0,		 255-255*i/MR, k++);

    return ncols;
}

void computeColor(double fx, double fy, unsigned char *pix, int* colorwheel, int ncols)
{
    double rad = sqrt(fx * fx + fy * fy);
    double a = atan2(-fy, -fx) / M_PI;
    double fk = (a + 1.0) / 2.0 * (ncols-1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % ncols;
    double f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    for (int b = 0; b < 3; b++) {
        double col0 = colorwheel[k0*3+b] / 255.0;
        double col1 = colorwheel[k1*3+b] / 255.0;
        double col = (1 - f) * col0 + f * col1;
        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range
        pix[2 - b] = (int)(255.0 * col);
    }
    pix[3] = 0xff; // alpha channel, only for alignment
}

float MotionToColor(unsigned char* fillPix, const cv::Mat& f_flowImg_r, int w, int h, float range = 0)
{
    // determine motion range:
    float maxrad;
    cv::Size szt = f_flowImg_r.size();
    int width = szt.width, height = szt.height;

    if (range > 0) {
        maxrad = range;
    }else{	// obtain the motion range according to the max flow
        float maxu = -999, maxv = -999;
        float minu = 999, minv = 999;
        maxrad = -1;
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                const auto flowU = f_flowImg_r.at<cv::Vec2f>(y,x)[0];
                const auto flowV = f_flowImg_r.at<cv::Vec2f>(y,x)[1];
                maxu = std::max(maxu, flowU);
                maxv = std::max(maxv, flowV);
                minu = std::min(minu, flowU);
                minv = std::min(minv, flowV);
                float rad = sqrt(flowU * flowU + flowV * flowV);
                maxrad = std::max(maxrad, rad);
            }
        }
        if (maxrad == 0) // if flow == 0 everywhere
            maxrad = 1;
    }

    //printf("max motion: %.2f  motion range: u = [%.2f,%.2f];  v = [%.2f,%.2f]\n",
    //	maxrad, minu, maxu, minv, maxv);

    int colorwheel[MAXWHEELCOLS*3];
    int ncols = makecolorwheel(colorwheel);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y*w+x;
            const auto flowU = f_flowImg_r.at<cv::Vec2f>(y,x)[0];
            const auto flowV = f_flowImg_r.at<cv::Vec2f>(y,x)[1];
            float dx = std::min(std::max(flowU / maxrad, -1.0f), 1.0f);
            float dy = std::min(std::max(flowV / maxrad, -1.0f), 1.0f);
            computeColor(dx, dy, (unsigned char*)(fillPix + idx * 4), colorwheel, ncols);
        }
    }

    return maxrad;
}

// Save a Depth/OF/SF as .flo file
void SaveFlowFile(cv::Mat& img, const char* filename)
{
    cv::Size szt = img.size();

    int width = szt.width, height = szt.height;
    int nc = img.channels();
    float tmp[nc];

    cv::Mat test(cv::Size(width, height), img.type());

    FILE *stream = fopen(filename, "wb");
    if (stream == 0)
        cout << "WriteFile: could not open file" << endl;

    // write the header
    fprintf(stream, "PIEH");
    if ((int)fwrite(&width,  sizeof(int),   1, stream) != 1 ||
        (int)fwrite(&height, sizeof(int),   1, stream) != 1)
        cout << "WriteFile: problem writing header" << endl;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (nc==1) // depth
                tmp[0] = img.at<float>(y,x);
            else if (nc==2) // Optical Flow
            {
                tmp[0] = img.at<cv::Vec2f>(y,x)[0];
                tmp[1] = img.at<cv::Vec2f>(y,x)[1];
            }
            else if (nc==4) // Scene Flow
            {
                tmp[0] = img.at<cv::Vec4f>(y,x)[0];
                tmp[1] = img.at<cv::Vec4f>(y,x)[1];
                tmp[2] = img.at<cv::Vec4f>(y,x)[2];
                tmp[3] = img.at<cv::Vec4f>(y,x)[3];
            }

            if ((int)fwrite(tmp, sizeof(float), nc, stream) != nc)
                cout << "WriteFile: problem writing data" << endl;
        }
    }

    cv::Mat flowImg(height, width, CV_8UC4);
    float maxFlow = MotionToColor(flowImg.data, img, width, height);
    cv::imwrite("/home/kevin/SLAM/others/OF_DIS-master/build/test.jpg", flowImg);
    cv::imshow("test", flowImg);
    cv::waitKey(0);
    fclose(stream);
}

// Save a depth as .pfm file
void SavePFMFile(cv::Mat& img, const char* filename)
{
  cv::Size szt = img.size();
  
  FILE *stream = fopen(filename, "wb");
  if (stream == 0)
    cout << "WriteFile: could not open file" << endl;

  // write the header
  fprintf(stream, "Pf\n%d %d\n%f\n", szt.width, szt.height, (float)-1.0f);    
  
  for (int y = szt.height-1; y >= 0 ; --y) 
  {
    for (int x = 0; x < szt.width; ++x) 
    {
      float tmp = -img.at<float>(y,x);
      if ((int)fwrite(&tmp, sizeof(float), 1, stream) != 1)
        cout << "WriteFile: problem writing data" << endl;         
    }
  }  
  fclose(stream);
}

// Read a depth/OF/SF as file
void ReadFlowFile(cv::Mat& img, const char* filename)
{
  FILE *stream = fopen(filename, "rb");
  if (stream == 0)
    cout << "ReadFile: could not open %s" << endl;
  
  int width, height;
  float tag;
  int nc = img.channels();
  float tmp[nc];  

  if ((int)fread(&tag,    sizeof(float), 1, stream) != 1 ||
      (int)fread(&width,  sizeof(int),   1, stream) != 1 ||
      (int)fread(&height, sizeof(int),   1, stream) != 1)
        cout << "ReadFile: problem reading file %s" << endl;

  for (int y = 0; y < height; y++) 
  {
    for (int x = 0; x < width; x++) 
    {
      if ((int)fread(tmp, sizeof(float), nc, stream) != nc)
        cout << "ReadFile(%s): file is too short" << endl;

      if (nc==1) // depth
        img.at<float>(y,x) = tmp[0];
      else if (nc==2) // Optical Flow
      {
        img.at<cv::Vec2f>(y,x)[0] = tmp[0];
        img.at<cv::Vec2f>(y,x)[1] = tmp[1];
      }
      else if (nc==4) // Scene Flow
      {
        img.at<cv::Vec4f>(y,x)[0] = tmp[0];
        img.at<cv::Vec4f>(y,x)[1] = tmp[1];
        img.at<cv::Vec4f>(y,x)[2] = tmp[2];
        img.at<cv::Vec4f>(y,x)[3] = tmp[3];
      }
    }
  }

  if (fgetc(stream) != EOF)
    cout << "ReadFile(%s): file is too long" << endl;

  fclose(stream);
}

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
  struct timeval tv_start_all, tv_end_all;
  gettimeofday(&tv_start_all, NULL);
  
  
  
  // *** Parse and load input images
  char *imgfile_ao = argv[1];
  char *imgfile_bo = argv[2];
  char *outfile = argv[3];
   
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
  img_ao_mat = cv::imread(imgfile_ao, incoltype);   // Read the file
  img_bo_mat = cv::imread(imgfile_bo, incoltype);   // Read the file    
  cv::Mat img_ao_fmat, img_bo_fmat;
  cv::Size sz = img_ao_mat.size();
  int width_org = sz.width;   // unpadded original image size
  int height_org = sz.height;  // unpadded original image size 
  
  
  
  
  // *** Parse rest of parameters, See oflow.h for definitions.
  int lv_f, lv_l, maxiter, miniter, patchsz, patnorm, costfct, tv_innerit, tv_solverit, verbosity;
  float mindprate, mindrrate, minimgerr, poverl, tv_alpha, tv_gamma, tv_delta, tv_sor;
  bool usefbcon, usetvref;
  //bool hasinfile; // initialization flow file
  //char *infile = nullptr;
  
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
    //hasinfile = (bool)atoi(argv[acnt++]);   // initialization flow file
    //if (hasinfile) infile = argv[acnt++];  
  }

  
  
  // *** Pad image such that width and height are restless divisible on all scales (except last)
  int padw=0, padh=0;
  int scfct = pow(2,lv_f); // enforce restless division by this number on coarsest scale
  //if (hasinfile) scfct = pow(2,lv_f+1); // if initialization file is given, make sure that size is restless divisible by 2^(lv_f+1) !
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
  
  // Timing, image loading
  if (verbosity > 1)
  {
    gettimeofday(&tv_end_all, NULL);
    double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
    printf("TIME (Image loading     ) (ms): %3g\n", tt);
    gettimeofday(&tv_start_all, NULL);
  }
  
  
  
  
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

  // Timing, image gradients and pyramid
  if (verbosity > 1)
  {
    gettimeofday(&tv_end_all, NULL);
    double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
    printf("TIME (Pyramide+Gradients) (ms): %3g\n", tt);
  }

  
//     // Read Initial Truth flow (if available)
//     float * initptr = nullptr;
//     cv::Mat flowinit;
//     if (hasinfile)
//     {
//       #if (SELECTMODE==1)
//       flowinit.create(height_org, width_org, CV_32FC2);
//       #else
//       flowinit.create(height_org, width_org, CV_32FC1);
//       #endif
//       
//       ReadFlowFile(flowinit, infile);
//         
//       // padding to ensure divisibility by 2
//       if (padh>0 || padw>0)
//         copyMakeBorder(flowinit,flowinit,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
//       
//       // resizing to coarsest scale - 1, since the array is upsampled at .5 in the code
//       float sc_fct = pow(2,-lv_f-1);
//       flowinit *= sc_fct;
//       cv::resize(flowinit, flowinit, cv::Size(), sc_fct, sc_fct , cv::INTER_AREA); 
//       
//       initptr = (float*)flowinit.data;
//     }

  
  
  
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

  if (verbosity > 1) gettimeofday(&tv_start_all, NULL);
      
  
  
  // *** Resize to original scale, if not run to finest level
  if (lv_l != 0)
  {
    flowout *= sc_fct;
    cv::resize(flowout, flowout, cv::Size(), sc_fct, sc_fct , cv::INTER_LINEAR);
  }
  
  // If image was padded, remove padding before saving to file
  flowout = flowout(cv::Rect((int)floor((float)padw/2.0f),(int)floor((float)padh/2.0f),width_org,height_org));

  // Save Result Image    
  #if (SELECTMODE==1)
  SaveFlowFile(flowout, outfile);
  #else
  SavePFMFile(flowout, outfile);      
  #endif

  if (verbosity > 1)
  {
    gettimeofday(&tv_end_all, NULL);
    double tt = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f + (tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
    printf("TIME (Saving flow file  ) (ms): %3g\n", tt);
  }
    
  return 0;
}


    


