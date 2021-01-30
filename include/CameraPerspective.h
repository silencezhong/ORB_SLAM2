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

#ifndef CAMERA_PERSPECTIVE_H
#define CAMERA_PERSPECTIVE_H

#include "CameraBase.h"

namespace camera_model
{
class CCameraPerspective : public CCameraBase
{
public:
    CCameraPerspective(const float f_fx_f, const float f_fy_f, const float f_cx_f, const float f_cy_f, const float f_width_f, const float f_height_f, const int f_rgb_i)
    : CCameraBase(f_fx_f, f_fy_f, f_cx_f, f_cy_f, f_width_f, f_height_f, f_rgb_i)
    {
    }

    CCameraPerspective() = delete;
    ~CCameraPerspective() = default;

    void CameraToImg(const float& x, const float& y, const float& z,
                             float& u, float& v) const final ;

    void ImgToCamera(float& x, float& y, float& z,
                     const float& u, const float& v) const final;

private:

};
}
#endif