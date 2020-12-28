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

        virtual void CameraToImg(const float& x, const float& y, const float& z,
                        float& u, float& v) const = 0;

        virtual void ImgToCamera(float& x, float& y, float& z,
                        const float& u, const float& v) const = 0;

        float Get_cx() { return m_cx_f; }
        float Get_cy() { return m_cy_f; }

        float GetWidth() { return m_width_f; }
        float GetHeight() { return m_height_f; }

    protected:
        //Calibration matrix
        cv::Mat m_intrinsicsMat;

        // focal length
        float m_fx_f;
        float m_fy_f;

        // principal
        float m_cx_f;
        float m_cy_f;

        // image width and height
        float m_width_f;
        float m_height_f;
    };
}
#endif // CAMERA_BASE_H