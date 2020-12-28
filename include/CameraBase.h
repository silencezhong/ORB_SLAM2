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

#include "Images.h"


namespace camera_model
{
    class CCameraBase
    {
    public:
        // construtors
        CCameraBase(const float f_fx_f, const float f_fy_f, const float f_cx_f, const float f_cy_f, const float f_width_f, const float f_height_f, const int f_rgb_i)
        : m_fx_f(f_fx_f), m_fy_f(f_fy_f), m_cx_f(f_cx_f), m_cy_f(f_cy_f), m_image(f_width_f, f_height_f, f_rgb_i)
        {
            cv::Mat K = cv::Mat::eye(3,3,CV_32F);
            K.at<float>(0,0) = m_fx_f;
            K.at<float>(1,1) = m_fy_f;
            K.at<float>(0,2) = m_cx_f;
            K.at<float>(1,2) = m_cy_f;
            K.copyTo(m_intrinsicsMat);
        }
        CCameraBase() = delete;
        ~CCameraBase() = default;

        virtual void CameraToImg(const float& x, const float& y, const float& z,
                        float& u, float& v) const = 0;

        virtual void ImgToCamera(float& x, float& y, float& z,
                        const float& u, const float& v) const = 0;

        float  getFx() const {return m_fx_f;}
        float  getFy() const {return m_fy_f;}

        float getCx() { return m_cx_f; }
        float getCy() { return m_cy_f; }

        const images::CImages& getImageClass() const { return m_image; }

        void setImage(const cv::Mat& f_img_r)
        {
            m_image.setImage(f_img_r);
        }

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
        images::CImages m_image;
    };
}
#endif // CAMERA_BASE_H