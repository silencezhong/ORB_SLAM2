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
    CCameraPerspective() = default;
    ~CCameraPerspective() = default;

    void CameraToImg(const double& x, const double& y, const double& z,
                             double& u, double& v) const final ;

    void ImgToCamera(double& x, double& y, double& z,
                     const double& u, const double& v) const final;

private:
};
}
#endif