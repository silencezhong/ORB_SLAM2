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

#ifndef CAMERA_FISHEYE_H
#define CAMERA_FISHEYE_H

#include "CameraBase.h"

namespace camera_model
{
    class CCameraFisheye : public CCameraBase
    {
    public:
        CCameraFisheye(const float f_fx_f, const float f_fy_f, const float f_cx_f, const float f_cy_f,
                       const float f_width_f, const float f_height_f, const int f_rgb_i,
                       const float f_k1_f, const float f_k2_f, const float f_p1_f, const float f_p2_f )
                : CCameraBase(f_fx_f, f_fy_f, f_cx_f, f_cy_f, f_width_f, f_height_f, f_rgb_i),
                  m_k1_f(f_k1_f), m_k2_f(f_k2_f), m_p1_f(f_p1_f), m_p2_f(f_p2_f)
        {
            cv::Mat DistCoef(4,1,CV_32F);
            DistCoef.at<float>(0) = m_k1_f;
            DistCoef.at<float>(1) = m_k2_f;
            DistCoef.at<float>(2) = m_p1_f;
            DistCoef.at<float>(3) = m_p2_f;
            DistCoef.copyTo(m_DistCoe);
        }
        CCameraFisheye() = delete;
        ~CCameraFisheye() = default;

        void CameraToImg(const float& x, const float& y, const float& z,
                         float& u, float& v) const final;

        void ImgToCamera(float& x, float& y, float& z,
                         const float& u, const float& v) const final;

        cv::Mat getDistortCoe() const { return m_DistCoe; }

        void showImage()
        {
            cv::Mat l_undistortIm = m_image.getImage();
            cv::fisheye::undistortImage(m_image.getImage(), l_undistortIm, m_intrinsicsMat, m_DistCoe );
//            cv::imshow("Display window", l_undistortIm);
//            cv::waitKey(0); // Wait for a keystroke in the window
            cv::imwrite("/home/kevin/save.jpg", l_undistortIm);
        }

    private:
        float m_k1_f;
        float m_k2_f;
        float m_p1_f;
        float m_p2_f;
        cv::Mat m_DistCoe;
    };
}
#endif