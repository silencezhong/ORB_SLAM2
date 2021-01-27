/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CAMERAMODELS_PINHOLE_H
#define CAMERAMODELS_PINHOLE_H

#include <cassert>
#include <vector>
#include <opencv2/core/core.hpp>
#include "GeometricCamera.h"
#include "TwoViewReconstruction.h"


namespace camera {
    class Pinhole : public GeometricCamera {

    public:
        Pinhole(const std::vector<float>& _vParameters,  const float f_height2Road_f) : GeometricCamera(_vParameters, f_height2Road_f), tvr(nullptr) {
            assert(mvParameters.size() == 4);
            mnId=nNextId++;
            mnType = CAMERA_TYPE::CAM_PINHOLE;
        }

        explicit Pinhole(Pinhole* pPinhole) : GeometricCamera(pPinhole->mvParameters, pPinhole->m_height2Road_f), tvr(nullptr) {
            assert(mvParameters.size() == 4);
            mnId=nNextId++;
            mnType = CAMERA_TYPE::CAM_PINHOLE;
        }

        Pinhole() = default;
        Pinhole& operator=(const Pinhole&) = default;

        ~Pinhole(){
            delete tvr;
        }

        cv::Point2f project(const cv::Point3f &p3D) final ;
        cv::Point2f project(const cv::Mat &m3D) final ;
        Eigen::Vector2d project(const Eigen::Vector3d & v3D) final;
        cv::Mat projectMat(const cv::Point3f& p3D) final;

        float uncertainty2(const Eigen::Matrix<double,2,1> &p2D) final;

        cv::Point3f unproject(const cv::Point2f &p2D) final;
        cv::Mat unprojectMat(const cv::Point2f &p2D) final;

        cv::Mat projectJac(const cv::Point3f &p3D) final;
        Eigen::Matrix<double,2,3> projectJac(const Eigen::Vector3d& v3D) final;

        cv::Mat unprojectJac(const cv::Point2f &p2D) final;

        bool ReconstructWithTwoViews(const std::vector<cv::KeyPoint>& vKeys1, const std::vector<cv::KeyPoint>& vKeys2, const std::vector<int> &vMatches12,
                                             cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated) final;

        cv::Mat toK() const;

        bool epipolarConstrain(GeometricCamera* pCamera2, const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, const cv::Mat& R12, const cv::Mat& t12, const float sigmaLevel, const float unc) final;

        bool matchAndtriangulate(const cv::KeyPoint& kp1, const cv::KeyPoint& kp2, GeometricCamera* pOther,
                                 cv::Mat& Tcw1, cv::Mat& Tcw2,
                                 const float sigmaLevel1, const float sigmaLevel2,
                                 cv::Mat& x3Dtriangulated) final { return false;}

        friend std::ostream& operator<<(std::ostream& os, const Pinhole& ph);
        friend std::istream& operator>>(std::istream& os, Pinhole& ph);
    private:
        cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

        //Parameters vector corresponds to
        //      [fx, fy, cx, cy]

        TwoViewReconstruction* tvr;
    };
}

//BOOST_CLASS_EXPORT_KEY(ORBSLAM2::Pinhole)

#endif //CAMERAMODELS_PINHOLE_H
