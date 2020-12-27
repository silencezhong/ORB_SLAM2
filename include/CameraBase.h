/**
*
* Copyright (C) Jie Zhong <jie.k.zhong@gmail.com>
*
* You should have received a copy of the GNU General Public License
* along with MultiCol-SLAM . If not, see <http://www.gnu.org/licenses/>.
*/

/*
CameraBase.h

@brief:
This class is a base class of camera model.

 Author: Jie Zhong
Date: 27.Dec.2020
*/

#ifndef CAMERA_BASE_H
#define CAMERA_BASE_H

// extern includes
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>


namespace camera_model
{
    class CCameraBase
    {
    public:
        // construtors
        CCameraBase() = default;

        ~CCameraBase() = default;

        virtual void CameraToImg(const double& x, const double& y, const double& z,
                        double& u, double& v) const = 0;

        virtual void ImgToCamera(double& x, double& y, double& z,
                        const double& u, const double& v) const = 0;

        double Get_u0() { return m_u0_d; }
        double Get_v0() { return m_v0_d; }

        double GetWidth() { return m_width_d; }
        double GetHeight() { return m_height_d; }

    protected:
        //Calibration matrix
        cv::Mat mK;

        // principal
        double m_u0_d;
        double m_v0_d;

        // image width and height
        double m_width_d;
        double m_height_d;
    };
}
#endif // CAMERA_BASE_H