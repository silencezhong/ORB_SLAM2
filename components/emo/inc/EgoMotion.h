//
// Created by kevin on 24.01.21.
//

#ifndef EMO_EGOMOTION_H
#define EMO_EGOMOTION_H

#include <vector>
#include "camera/inc/PinholeCamera.h"

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Initializer.h"
#include "ORBmatcher.h"
#include "ORBextractor.h"

namespace emo {

    namespace LINE
    {
        enum class LINE_RELATION
        {
            ABOVE_LINE = 1,
            AT_LINE,
            BELOW_LINE
        };
        constexpr float G_BELONG_TO_LINE_THRESHOLD_F = 0.2f;
    }

    namespace utility
    {
        template<typename T>
        class optional
        {
        public:
            optional() : m_data(), m_flag_b(false){}
            explicit optional(const T& f_data_r) : m_data(f_data_r), m_flag_b(true){}

            optional(const optional&) = default;
            optional& operator=(const optional&) = delete;
            ~optional() = default;

            bool isValid() const
            {
                return m_flag_b;
            }

            T getData() const
            {
                return m_data;
            }

        private:
            T m_data;
            bool m_flag_b;
        };

        cv::Matx31f imagePoint2homogeneous(const cv::Point2f& f_imgPoint_r)
        {
            cv::Matx31f l_homoKey;
            l_homoKey(0) = f_imgPoint_r.x;
            l_homoKey(1) = f_imgPoint_r.y;
            l_homoKey(2) = 1;
            return l_homoKey;
        }

        void normlize(cv::Matx31f& f_3dVec_r)
        {
            const float l_norm_f = cv::norm(f_3dVec_r);
            f_3dVec_r(0) /= l_norm_f; f_3dVec_r(1) /= l_norm_f; f_3dVec_r(2) /= l_norm_f;
        }

        void copyRotation2Transform(const cv::Matx33f& f_rotation_r, cv::Matx44f& f_transform_r)
        {
            f_transform_r(0, 0) = f_rotation_r(0,0); f_transform_r(0, 1) = f_rotation_r(0,1); f_transform_r(0, 2) = f_rotation_r(0,2);
            f_transform_r(1, 0) = f_rotation_r(1,0); f_transform_r(1, 1) = f_rotation_r(1,1); f_transform_r(1, 2) = f_rotation_r(1,2);
            f_transform_r(2, 0) = f_rotation_r(2,0); f_transform_r(2, 1) = f_rotation_r(2,1); f_transform_r(2, 2) = f_rotation_r(2,2);
        }
        void copyTranslation2Transform(const cv::Matx31f& f_translation_r, cv::Matx44f& f_transform_r)
        {
            f_transform_r(0, 3) = f_translation_r(0); f_transform_r(1, 3) = f_translation_r(1); f_transform_r(2, 3) = f_translation_r(2);
        }
    }

    struct CLine
    {
    public:
        CLine(const cv::Point2f& f_p1_r, const cv::Point2f& f_p2_r)
        {
            if(std::abs(f_p1_r.x - f_p2_r.x) > std::numeric_limits<float >::min())
            {
                m_a_f = -(f_p2_r.y - f_p1_r.y)/(f_p2_r.x - f_p1_r.x);
                m_b_f = 1;
                m_c_f = -m_a_f*f_p1_r.x - f_p1_r.y;
            }
            else
            {
                m_a_f = 1;
                m_b_f = 0;
                m_c_f = -f_p1_r.x;
            }
            m_distanceNorminator_f = 1.0f/std::sqrt(m_a_f*m_a_f + m_b_f*m_b_f);
        }

        LINE::LINE_RELATION checkRelation2Line(const cv::Point2f& f_point_r)
        {
            const float l_evaluatePoint_f = m_a_f*f_point_r.x + m_b_f*f_point_r.y + m_c_f;
            if(std::abs(l_evaluatePoint_f) < LINE::G_BELONG_TO_LINE_THRESHOLD_F)
            {
                return LINE::LINE_RELATION ::AT_LINE;
            }
            else
            {
                // notice: the coordinate system is different to conventional Descartes coordinate.
                if(l_evaluatePoint_f > 0)
                {
                    return LINE::LINE_RELATION ::BELOW_LINE;
                }
                else
                {
                    return LINE::LINE_RELATION ::ABOVE_LINE;
                }
            }
        }

        CLine() = delete;
        CLine(const CLine&) = default;
        CLine& operator=(const CLine&) = delete;
        ~CLine() = default;

        float calculatePointdistance2FlowLine(const cv::Point2f& f_point_r)
        {
            return m_distanceNorminator_f*std::abs(m_a_f*f_point_r.x + m_b_f*f_point_r.y + m_c_f);
        }

        // flow vector is on the line a*x + b*y + c = 0
        float m_a_f;
        float m_b_f;
        float m_c_f;
        float m_distanceNorminator_f;
    };

    struct FlowEntry
    {
        FlowEntry(const cv::Point2f& f_startPoint_r, const float f_dx_f, const float f_dy_f, const cv::Point3f& f_unscaled3DPos_r, const cv::Matx33f& f_invK_r)
                : m_startPoint(f_startPoint_r),
                  m_dx_f(f_dx_f),
                  m_dy_f(f_dy_f),
                  m_flowLine(m_startPoint, cv::Point2f(m_startPoint.x + m_dx_f, m_startPoint.y + m_dy_f)),
                  m_weight_f(1.0f),
                  m_unscaled3DPos(f_unscaled3DPos_r)
        {
            m_length_f = std::sqrt(m_dx_f*m_dx_f + m_dy_f*m_dy_f);

            m_3dRayStartFrame = f_invK_r*utility::imagePoint2homogeneous(m_startPoint);
            m_3dRayDerotatedEndFrame = f_invK_r*utility::imagePoint2homogeneous(cv::Point2f(m_startPoint.x + m_dx_f, m_startPoint.y + m_dy_f));
            m_weight_f = 1.0f;
        }

        FlowEntry() = delete;
        FlowEntry(const FlowEntry&) = default;
        FlowEntry& operator=(const FlowEntry&) = delete;
        ~FlowEntry() = default;

        float calculatePointdistance2FlowLine(const cv::Point2f& f_point_r)
        {
            return m_flowLine.calculatePointdistance2FlowLine(f_point_r);
        }

        cv::Point2f getEndPoint() const
        {
            return cv::Point2f(m_startPoint.x + m_dx_f, m_startPoint.y + m_dy_f);
        }

        cv::Point2f m_startPoint;
        float       m_dx_f;
        float       m_dy_f;
        float       m_length_f;

        CLine        m_flowLine;

        // 3d ray inv(K)*image_homogeneous
        cv::Matx31f m_3dRayStartFrame;
        cv::Matx31f m_3dRayDerotatedEndFrame;

        // used for iterative least square
        float m_weight_f;

        // unscaled 3d Postion, estimated by decomposing F/H and triangulation
        cv::Point3f m_unscaled3DPos;
    };

    constexpr float G_MIN_FLOW_LENGTH_THRESHOLD_F = 2.0f;
    constexpr int   G_MIN_INITIAL_SET_NUMBER      = 5;
    constexpr float G_M_ESTIMATOR_INLIER_THRESHOLD_F = 20.0f;

    class EgoMotion {
    public:
        EgoMotion(
                int nFeatures, float fScaleFactor, int nLevels, int fIniThFAST, int fMinThFAST,
                const camera::Pinhole& f_camera_r ) :
        m_camera_r(f_camera_r),
        m_orbExtractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST),
        m_orbMatcher(1.0f, true),// ToDo: check meaning of first parameter
        m_initializer(),
        m_map()
        {

        }

        EgoMotion(const EgoMotion &) = delete;
        EgoMotion&operator = (const EgoMotion&) = delete;
        ~EgoMotion() = default;

        // init map
        void createInitialMap(const cv::Mat &f_img1_r, const cv::Mat &f_img2_r);

    private:

        // estimate a transform from camera coordinate of f_img2_r to camera coordinate of f_img1_r
        utility::optional<cv::Matx44f> estimateMotionBetweenTwoFrames(const cv::Mat& f_img1_r, const cv::Mat& f_img2_r);

        // estimate FOE using deroated flow
        utility::optional<cv::Point2f> estimateFOE( std::vector<FlowEntry>& f_derotatedFlowVec_r);

        // estimate the scale of unscaled translation t, R and t are estimated by decomposing E.
        utility::optional<float> estimateScale(
                std::vector<FlowEntry>& f_derotatedFlowVec_r,
                const cv::Point2f& f_FOE_r,
                const cv::Matx31f& f_derotatedTranslation_r,
                const cv::Matx31f& f_roadNorm_r,
                const float f_imgWidth_f,
                const float f_imgHeight_f );


        // camera
        const camera::Pinhole& m_camera_r;

        // feature matching
        ORBextractor m_orbExtractor;
        ORBmatcher  m_orbMatcher;

        // Initalization (only for monocular)
        Initializer m_initializer;

        //Map
        Map m_map;

        //Local Map
        KeyFrame *mpReferenceKF;
        std::vector<KeyFrame*> mvpLocalKeyFrames;
        std::vector<MapPoint*> mvpLocalMapPoints;
    };
}

#endif //EMO_EGOMOTION_H
