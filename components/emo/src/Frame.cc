/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"
#include "ORBmatcher.h"
#include <thread>

namespace emo
{

    long unsigned int Frame::nNextId=0;
    bool Frame::mbInitialComputations=true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

    //Copy Constructor
    Frame::Frame(const Frame &frame)
            :m_ORBextractor(frame.m_ORBextractor),
             mTimeStamp(frame.mTimeStamp),m_camera(frame.m_camera), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
             N(frame.N), mvKeys(frame.mvKeys),
             mvuRight(frame.mvuRight),
             mvDepth(frame.mvDepth),
             mDescriptors(frame.mDescriptors.clone()),
             mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
             mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
             mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
             mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
             mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
    {
        for(int i=0;i<FRAME_GRID_COLS;i++)
            for(int j=0; j<FRAME_GRID_ROWS; j++)
                mGrid[i][j]=frame.mGrid[i][j];

        if(!frame.mTcw.empty())
            SetPose(frame.mTcw);
    }




    Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor& extractor, camera::Pinhole& f_pinhole_r, cv::Mat K, cv::Mat &distCoef)
            : m_ORBextractor(extractor),
              mTimeStamp(timeStamp), m_camera(f_pinhole_r), mK(K.clone()), mDistCoef(distCoef.clone()), m_derotatedKeys()
    {
//         auto t1 = mK.at<float>(0,0);
//        auto t2 = mK.at<float>(0,2);
//        auto t3 = mK.at<float>(1,1);
//        auto t4 = mK.at<float>(1,2);
//        auto t5 = mK.at<float>(2,2);

        // Frame ID
        mnId=nNextId++;

        // Scale Level Info
        mnScaleLevels = m_ORBextractor.GetLevels();
        mfScaleFactor = m_ORBextractor.GetScaleFactor();
        mfLogScaleFactor = log(mfScaleFactor);
        mvScaleFactors = m_ORBextractor.GetScaleFactors();
        mvInvScaleFactors = m_ORBextractor.GetInverseScaleFactors();
        mvLevelSigma2 = m_ORBextractor.GetScaleSigmaSquares();
        mvInvLevelSigma2 = m_ORBextractor.GetInverseScaleSigmaSquares();

        // ORB extraction
        ExtractORB(0,imGray);

        N = mvKeys.size();

        if(mvKeys.empty())
            return;

        UndistortKeyPoints();

        // Set no stereo information
        mvuRight = std::vector<float>(N,-1);
        mvDepth = std::vector<float>(N,-1);

        mvpMapPoints = std::vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
        mvbOutlier = std::vector<bool>(N,false);

        // This is done only for the first Frame (or after a change in the calibration)
        if(mbInitialComputations)
        {
            ComputeImageBounds(imGray);

            mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
            mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

            fx = K.at<float>(0,0);
            fy = K.at<float>(1,1);
            cx = K.at<float>(0,2);
            cy = K.at<float>(1,2);
            invfx = 1.0f/fx;
            invfy = 1.0f/fy;

            mbInitialComputations=false;
        }

        AssignFeaturesToGrid();
    }

    void Frame::AssignFeaturesToGrid()
    {
        int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
        for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
            for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
                mGrid[i][j].reserve(nReserve);

        for(int i=0;i<N;i++)
        {
            const cv::KeyPoint &kp = mvKeys[i];

            int nGridPosX, nGridPosY;
            if(PosInGrid(kp,nGridPosX,nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    void Frame::ExtractORB(int flag, const cv::Mat &im)
    {
        m_ORBextractor(im,cv::Mat(),mvKeys,mDescriptors);
    }

    void Frame::SetPose(cv::Mat Tcw)
    {
        mTcw = Tcw.clone();
        UpdatePoseMatrices();
    }

    void Frame::UpdatePoseMatrices()
    {
        mRcw = mTcw.rowRange(0,3).colRange(0,3);
        mRwc = mRcw.t();
        mtcw = mTcw.rowRange(0,3).col(3);
        mOw = -mRcw.t()*mtcw;
    }

    bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
    {
        pMP->mbTrackInView = false;

        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const cv::Mat Pc = mRcw*P+mtcw;
        const auto &PcX = Pc.at<float>(0);
        const auto &PcY= Pc.at<float>(1);
        const auto &PcZ = Pc.at<float>(2);

        // Check positive depth
        if(PcZ<0.0f)
            return false;

        // Project in image and check it is not outside
        const float invz = 1.0f/PcZ;
        const float u=fx*PcX*invz+cx;
        const float v=fy*PcY*invz+cy;

        if(u<mnMinX || u>mnMaxX)
            return false;
        if(v<mnMinY || v>mnMaxY)
            return false;

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const cv::Mat PO = P-mOw;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            return false;

        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn)/dist;

        if(viewCos<viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = u;
//        pMP->mTrackProjXR = u - mbf*invz;
        pMP->mTrackProjY = v;
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }

    std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
    {
        std::vector<size_t> vIndices;
        vIndices.reserve(N);

        const int nMinCellX = std::max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
        if(nMinCellX>=FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = std::min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
        if(nMaxCellX<0)
            return vIndices;

        const int nMinCellY = std::max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
        if(nMinCellY>=FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = std::min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
        if(nMaxCellY<0)
            return vIndices;

        const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

        for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
        {
            for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
            {
                const std::vector<size_t> vCell = mGrid[ix][iy];
                if(vCell.empty())
                    continue;

                for(size_t j=0, jend=vCell.size(); j<jend; j++)
                {
                    const cv::KeyPoint &kpUn = mvKeys[vCell[j]];
                    if(bCheckLevels)
                    {
                        if(kpUn.octave<minLevel)
                            continue;
                        if(maxLevel>=0)
                            if(kpUn.octave>maxLevel)
                                continue;
                    }

                    const float distx = kpUn.pt.x-x;
                    const float disty = kpUn.pt.y-y;

                    if(fabs(distx)<r && fabs(disty)<r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    void Frame::derotateKeys(const cv::Mat& f_rotation_r )
    {
        m_derotatedKeys.clear();
//        m_derotatedKeys.resize(mvKeys.size());
        const cv::Mat l_invK = m_camera.toK().inv();
        const cv::Mat l_K = m_camera.toK();
        cv::Mat l_homoKey = cv::Mat::zeros(3,1, CV_32F);
        for(const auto& l_curKey : mvKeys)
        {
            l_homoKey.at<float>(0) = l_curKey.pt.x;
            l_homoKey.at<float>(1) = l_curKey.pt.y;
            l_homoKey.at<float>(2) = 1;
            cv::Mat l_normlizedKey = l_invK * l_homoKey;
//            auto t1 = l_normlizedKey.at<float>(0);
//            auto t2 = l_normlizedKey.at<float>(1);
//            auto t3 = l_normlizedKey.at<float>(2);
            cv::Mat l_deroatedNormKey = f_rotation_r*l_normlizedKey;
            cv::Mat l_deroatedKey = l_K*l_deroatedNormKey;
//            auto t4 = l_deroatedKey.at<float>(0);
//            auto t5 = l_deroatedKey.at<float>(1);
//            auto t6 = l_deroatedKey.at<float>(2);
            cv::Point2f l_imgPoint = {l_deroatedKey.at<float>(0,0)/l_deroatedKey.at<float>(0,2), l_deroatedKey.at<float>(0,1)/l_deroatedKey.at<float>(0,2)};
            m_derotatedKeys.push_back(l_imgPoint);
        }
    }

    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
    {
        posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
        posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

        //Keypoint's coordinates are undistorted, which could cause to go out of the image
        if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
            return false;

        return true;
    }

    void Frame::UndistortKeyPoints()
    {
        if(mDistCoef.at<float>(0)==0.0)
        {
            mvKeysUn=mvKeys;
            return;
        }
    }

    void Frame::ComputeImageBounds(const cv::Mat &imLeft)
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
} //namespace ORB_SLAM
