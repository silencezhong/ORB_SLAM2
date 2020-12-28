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

#ifndef IMAGES_H
#define IMAGES_H

// extern includes
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace images
{
    class CImages{
    public:
        CImages(const float f_width_f, const float f_height_f, const int f_rgb_i)
        {
            assert(f_width_f > 0);
            assert(f_height_f > 0);
            m_width_u = static_cast<uint>(f_width_f);
            m_height_u = static_cast<uint >(f_height_f);
            m_rgb_b = static_cast<bool>(f_rgb_i);
        }
        CImages() = delete;
        ~CImages() = default;

        void setImage(const cv::Mat &f_im_r)
        {
            m_imGray = f_im_r;
            if(m_imGray.channels()==3)
            {
                if(m_rgb_b)
                    cvtColor(m_imGray,m_imGray,CV_RGB2GRAY);
                else
                    cvtColor(m_imGray,m_imGray,CV_BGR2GRAY);
            }
        }

        const cv::Mat& getImage() const
        {
            return m_imGray;
        }

        bool getRgbFlag() const
        {
            return m_rgb_b;
        }

    private:
        uint m_frameIdx_u;
        uint m_width_u;
        uint m_height_u;
        bool m_rgb_b;
        cv::Mat m_imGray;
    };
}
#endif