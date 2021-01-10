
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <sys/time.h>
#include <fstream>
#include <assert.h>
    
#include "Frame.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "ORBmatcher.h"
#include "camera/inc/PinholeCamera.h"

using namespace std;
using namespace emo;

int main( int argc, char** argv )
{
    struct timeval tv_start_all, tv_end_all;
    gettimeofday(&tv_start_all, NULL);

     // *** Parse and load input images
    char *l_img1Name_p = argv[1];
    char *l_img2Name_p = argv[2];

    cv::Mat l_fisrsImg = cv::imread(l_img1Name_p, CV_LOAD_IMAGE_UNCHANGED);   // Read the file
    cv::Mat l_2ndImg = cv::imread(l_img2Name_p, CV_LOAD_IMAGE_UNCHANGED);   // Read the file
    cv::Mat l_visImag = l_fisrsImg;
    if(l_fisrsImg.channels()==3) {
        cvtColor(l_fisrsImg, l_fisrsImg, CV_RGB2GRAY);
        cvtColor(l_2ndImg, l_2ndImg, CV_RGB2GRAY);
    }
    cv::Size sz = l_fisrsImg.size();

    // init camera
    const cv::FileStorage l_Settings( argv[3], cv::FileStorage::READ);
    assert(l_Settings.isOpened());
    float fx, fy, cx, cy;

    cv::FileNode node = l_Settings["PerspectiveCamera.fx"];
    assert(!node.empty() && node.isReal());
    fx = node.real();
    node = l_Settings["PerspectiveCamera.fy"];
    assert(!node.empty() && node.isReal());
    fy = node.real();
    node = l_Settings["PerspectiveCamera.cx"];
    assert(!node.empty() && node.isReal());
    cx = node.real();
    node = l_Settings["PerspectiveCamera.cy"];
    assert(!node.empty() && node.isReal());
    cy = node.real();
    vector<float> vCamCalib{fx,fy,cx,cy};
    auto l_pinholeCamera = new camera::Pinhole(vCamCalib);

    // Distortion parameters
    cv::Mat l_DistCoef(4,1,CV_32F);
    node = l_Settings["PerspectiveCamera.k1"];
    assert(!node.empty() && node.isReal());
    l_DistCoef.at<float>(0) = node.real();
    node = l_Settings["PerspectiveCamera.k2"];
    assert(!node.empty() && node.isReal());
    l_DistCoef.at<float>(1) = node.real();
    node = l_Settings["PerspectiveCamera.p1"];
    assert(!node.empty() && node.isReal());
    l_DistCoef.at<float>(2) = node.real();
    node = l_Settings["PerspectiveCamera.p2"];
    assert(!node.empty() && node.isReal());
    l_DistCoef.at<float>(3) = node.real();

    // load ORB setting
    int nFeatures = l_Settings["ORBextractor.nFeatures"];
    float fScaleFactor = l_Settings["ORBextractor.scaleFactor"];
    int nLevels = l_Settings["ORBextractor.nLevels"];
    int fIniThFAST = l_Settings["ORBextractor.iniThFAST"];
    int fMinThFAST = l_Settings["ORBextractor.minThFAST"];
    auto l_orbExtractor = ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // init frames
    Frame m_firstFrame = Frame(l_fisrsImg, 1, l_orbExtractor,*l_pinholeCamera, l_pinholeCamera->toK(), l_DistCoef);
    std::vector<cv::Point2f> mvbPrevMatched;
    mvbPrevMatched.resize(m_firstFrame.mvKeys.size());
    for(size_t i=0; i<m_firstFrame.mvKeys.size(); i++)
        mvbPrevMatched[i]=m_firstFrame.mvKeys[i].pt;

    Frame m_2ndFrame = Frame(l_2ndImg, 1, l_orbExtractor, *l_pinholeCamera, l_pinholeCamera->toK(), l_DistCoef);
    Initializer l_Initializer = Initializer(m_firstFrame,1.0,200);
    ORBmatcher  l_orbMatcher  = ORBmatcher(1.0, true);

    // match the ORB features
    auto num = m_firstFrame.mvKeys.size();
    assert(m_firstFrame.mvKeys.size() > 100);
    assert(m_2ndFrame.mvKeys.size() > 100);
    std::vector<int> mvIniMatches;
    fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
    int nmatches = l_orbMatcher.SearchForInitialization(m_firstFrame, m_2ndFrame, mvbPrevMatched,mvIniMatches,100);
    assert(nmatches > 70);

    // show flow
    std::vector<pair<int, int>> l_Matches12Vec;
    for(size_t i=0, iend=mvIniMatches.size();i<iend; i++)
    {
        if(mvIniMatches[i]>=0)
        {
            l_Matches12Vec.push_back(make_pair(i,mvIniMatches[i]));
        }
    }
    cv::Mat mask = cv::Mat::zeros(l_visImag.size(), l_visImag.type());
    for(int l_idx_i = 0; l_idx_i < l_Matches12Vec.size(); ++l_idx_i)
    {
        cv::KeyPoint l_point1 = m_firstFrame.mvKeys[l_Matches12Vec[l_idx_i].first];
        cv::KeyPoint l_point2 = m_2ndFrame.mvKeys[l_Matches12Vec[l_idx_i].second];

        cv::line(mask, l_point1.pt, l_point2.pt, cv::Scalar(0, 0,256),2);
    }
//
//    cv::FileStorage file("l_fisrsImg.ext", cv::FileStorage::WRITE);
//    file << "matName" << l_fisrsImg;
    cv::Mat img;
    add(l_visImag, mask, img);
    imshow("test", img);
    cvWaitKey(0);
    // calculate the motion between two images by decomposing F(Fundamental matrix) or H(Homography matrix)
    cv::Mat Rcw; // Current Camera Rotation
    cv::Mat tcw; // Current Camera Translation
    vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
    std::vector<cv::Point3f> mvIniP3D;
    if(l_Initializer.Initialize(m_firstFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
    {
        for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
        {
            if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
            {
                mvIniMatches[i] = -1;
                nmatches--;
            }
        }

        // Set Frame Poses
        m_firstFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
        cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
        Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
        tcw.copyTo(Tcw.rowRange(0, 3).col(3));
        m_2ndFrame.SetPose(Tcw);
    }
}


    


