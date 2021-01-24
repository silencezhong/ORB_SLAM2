
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cassert>

#include "Frame.h"
#include "ORBextractor.h"
#include "Initializer.h"
#include "ORBmatcher.h"
#include "camera/inc/PinholeCamera.h"

using namespace std;
using namespace emo;

namespace FOE_ESTIMATION
{
    constexpr float G_MIN_FLOW_LENGTH_THRESHOLD_F = 2.0f;
    constexpr int   G_MIN_INITIAL_SET_NUMBER      = 5;
    constexpr float G_M_ESTIMATOR_INLIER_THRESHOLD_F = 20.0f;
}

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
    cv::Vec3f to_vec3f(const cv::Mat1f& f_mat1f_r)
    {
        CV_Assert((f_mat1f_r.rows == 3) && (f_mat1f_r.cols == 1));
        return cv::Vec3f(f_mat1f_r.at<float>(0), f_mat1f_r.at<float>(1), f_mat1f_r.at<float>(2));
    }

    cv::Point3f toPoint(const cv::Matx31f& f_mat_r)
    {
        return cv::Point3f(f_mat_r(0), f_mat_r(1), f_mat_r(2));
    }

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

    cv::Mat toMat(const cv::Point2f& f_point_r)
    {
        cv::Mat l_return = cv::Mat(2,1, CV_32F);
        l_return.at<float>(0,0) = f_point_r.x;
        l_return.at<float>(1,0) = f_point_r.y;
        return l_return;
    }
}

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

void updateWeight(
        std::vector<FlowEntry>& f_flowVec_r,
        const cv::Point2f& f_foe_r,
        const cv::Mat& f_img_r)
{
    for(auto& l_flowEntry_r : f_flowVec_r)
    {
        if(l_flowEntry_r.m_length_f > FOE_ESTIMATION::G_MIN_FLOW_LENGTH_THRESHOLD_F)
        {
            const float l_residual_f = l_flowEntry_r.calculatePointdistance2FlowLine(f_foe_r);
            l_flowEntry_r.m_weight_f = (l_residual_f <= FOE_ESTIMATION::G_M_ESTIMATOR_INLIER_THRESHOLD_F) ? 1.0f : FOE_ESTIMATION::G_M_ESTIMATOR_INLIER_THRESHOLD_F/l_residual_f;
        }
    }
}

optional<float> estimateScale(
        std::vector<FlowEntry>& f_flowVec_r,
        const cv::Point2f& f_FOE_r,
        const cv::Matx31f& f_derotatedTranslation_r,
        const cv::Matx31f& f_roadNorm_r,
        const float f_imgWidth_f,
        const float f_imgHeight_f,
        const cv::Mat& f_img_r)
{
    // road homography : lambda*x_end = H*x_start, x_end = [col_end, row_end, 1], x_start similar, is in homogeneous coordinate
    // H = K*(R - t*n'/d)*invK, R: rotation, t: translation, n: road normal vector.
    // Since in the flow entry x_end is already derotated from end frame to start frame, x_deroated_end = K*invR*invK*x_end
    // lambda * x_derotated_end = K* (I - derotated_t*n'*s) *invK * x_start, let s = 1/d, derotated_t = inv(R)*t
    // <=> lambda * invK * x_derotated_end = (I - derotated_t*n*s) * (invK * x_start), following we simply use x_derotated_end to represent invK*x_derotated_end, x_start similar
    //
    // lambda * x_derotated_end = (I - s*M) * x_start, let M = derotated_t * n' is a 3x3 matrix
    // let x_deroated_end = [u v 1]', x_start = [x, y ,1], M = (a1 a2 a3; a4 a5 a6; a7 a8 a9);
    // Then get rid of lambda: u = (1-sa1)x - sa2y -sa3)/(-sa7x - sa8y +(1-sa9))
    //                         v = (-sa4x + (1-sa5)y - sa6)/(-sa7x - sa8y +(1-sa9))
    // <=>lu =  (-sa7x - sa8y +(1-sa9))*u - ((1-sa1)x - sa2y -sa3) = 0;
    //    lv =  (-sa7x - sa8y +(1-sa9))*v - (-sa4x + (1-sa5)y - sa6) = 0;
    // to optimize E = sum(wi*(l_ui² + l_vi²)) using weighted least square with regarding to s
    //
    // since K = [fx 0 cx; 0 fy cy; 0 0 1] and x_derotated_end_img is [..,.., 1], x_deroated_end = invK*x_derotated_end_img has format [y1 y2 1]', x_start similar
    //
    //syms u v x y real
    //syms a1 a2 a3 a4 a5 a6 a7 a8 a9 real
    //syms s real
    //W = eye(3) - s*[a1 a2 a3; a4 a5 a6; a7 a8 a9];
    //E = 0.5*((u*W(3,:)*[x y 1]' - W(1,:)*[x y 1]')^2 + (v*W(3,:)*[x y 1]' - W(2,:)*[x y 1]')^2);
    //derivative_s = collect(expand(diff(E,s)),s);
    //
    // m = (a1^2*x^2 + 2*a1*a2*x*y + 2*a1*a3*x - 2*a1*a7*u*x^2 - 2*a1*a8*u*x*y - 2*a1*a9*u*x + a2^2*y^2 + 2*a2*a3*y - 2*a2*a7*u*x*y - 2*a2*a8*u*y^2 - 2*a2*a9*u*y + a3^2 - 2*a3*a7*u*x - 2*a3*a8*u*y - 2*a3*a9*u + a4^2*x^2 + 2*a4*a5*x*y + 2*a4*a6*x - 2*a4*a7*v*x^2 - 2*a4*a8*v*x*y - 2*a4*a9*v*x + a5^2*y^2 + 2*a5*a6*y - 2*a5*a7*v*x*y - 2*a5*a8*v*y^2 - 2*a5*a9*v*y + a6^2 - 2*a6*a7*v*x - 2*a6*a8*v*y - 2*a6*a9*v + a7^2*u^2*x^2 + a7^2*v^2*x^2 + 2*a7*a8*u^2*x*y + 2*a7*a8*v^2*x*y + 2*a7*a9*u^2*x + 2*a7*a9*v^2*x + a8^2*u^2*y^2 + a8^2*v^2*y^2 + 2*a8*a9*u^2*y + 2*a8*a9*v^2*y + a9^2*u^2 + a9^2*v^2);
    // n = a3*u + a6*v - a3*x - a6*y - a9*u^2 - a9*v^2 - a1*x^2 - a5*y^2 + a7*u*x^2 - a7*u^2*x - a7*v^2*x - a8*u^2*y + a8*v*y^2 - a8*v^2*y + a1*u*x + a9*u*x + a2*u*y + a4*v*x + a5*v*y + a9*v*y - a2*x*y - a4*x*y + a8*u*x*y + a7*v*x*y;
    // s = -sum(wi*ni)/sum(wi*mi)

    cv::Matx33f l_translationMultiRoadNorm = f_derotatedTranslation_r*f_roadNorm_r.t(); // M
    const float a1 = l_translationMultiRoadNorm(0,0); const float a2 = l_translationMultiRoadNorm(0,1); const float a3 = l_translationMultiRoadNorm(0,2);
    const float a4 = l_translationMultiRoadNorm(1,0); const float a5 = l_translationMultiRoadNorm(1,1); const float a6 = l_translationMultiRoadNorm(1,2);
    const float a7 = l_translationMultiRoadNorm(2,0); const float a8 = l_translationMultiRoadNorm(2,1); const float a9 = l_translationMultiRoadNorm(2,2);
    std::vector<float> l_numeratorVec(f_flowVec_r.size(), 0.0f); //Vi
    std::vector<float> l_denominatorVec(f_flowVec_r.size(), 0.0f); //Ui
    std::vector<bool>  l_roadFlagVec(f_flowVec_r.size(), false);

    // determine the road ROI
    CLine l_leftRoadLine(cv::Point2f(0, f_imgHeight_f), f_FOE_r);
    CLine l_rightRoadLine(cv::Point2f(f_imgWidth_f, f_imgHeight_f), f_FOE_r);

    // prepare the calculation
    for(int l_idx_i = 0; l_idx_i < f_flowVec_r.size(); ++l_idx_i)
    {
        FlowEntry& l_flowEntry = f_flowVec_r[l_idx_i];
        const bool l_isBelowLeftLine_b  = LINE::LINE_RELATION::BELOW_LINE == l_leftRoadLine.checkRelation2Line(l_flowEntry.m_startPoint);
        const bool l_isBelowRightLine_b = LINE::LINE_RELATION::BELOW_LINE == l_rightRoadLine.checkRelation2Line(l_flowEntry.m_startPoint);
        if(l_isBelowLeftLine_b && l_isBelowRightLine_b )
        {
            l_roadFlagVec[l_idx_i] = true;
            const float x = l_flowEntry.m_3dRayStartFrame(0);        const float y = l_flowEntry.m_3dRayStartFrame(1);
            const float u = l_flowEntry.m_3dRayDerotatedEndFrame(0); const float v = l_flowEntry.m_3dRayDerotatedEndFrame(1);
            l_denominatorVec[l_idx_i] = (pow(a1,2)*pow(x,2) + 2*a1*a2*x*y + 2*a1*a3*x - 2*a1*a7*u*pow(x,2) - 2*a1*a8*u*x*y - 2*a1*a9*u*x + pow(a2,2)*pow(y,2) + 2*a2*a3*y - 2*a2*a7*u*x*y - 2*a2*a8*u*pow(y,2) - 2*a2*a9*u*y + pow(a3,2) - 2*a3*a7*u*x - 2*a3*a8*u*y - 2*a3*a9*u + pow(a4,2)*pow(x,2) + 2*a4*a5*x*y + 2*a4*a6*x - 2*a4*a7*v*pow(x,2) - 2*a4*a8*v*x*y - 2*a4*a9*v*x + pow(a5,2)*pow(y,2) + 2*a5*a6*y - 2*a5*a7*v*x*y - 2*a5*a8*v*pow(y,2) - 2*a5*a9*v*y + pow(a6,2) - 2*a6*a7*v*x - 2*a6*a8*v*y - 2*a6*a9*v + pow(a7,2)*pow(u,2)*pow(x,2) + pow(a7,2)*pow(v,2)*pow(x,2) + 2*a7*a8*pow(u,2)*x*y + 2*a7*a8*pow(v,2)*x*y + 2*a7*a9*pow(u,2)*x + 2*a7*a9*pow(v,2)*x + pow(a8,2)*pow(u,2)*pow(y,2) + pow(a8,2)*pow(v,2)*pow(y,2) + 2*a8*a9*pow(u,2)*y + 2*a8*a9*pow(v,2)*y + pow(a9,2)*pow(u,2) + pow(a9,2)*pow(v,2));
            l_numeratorVec[l_idx_i] = a3*u + a6*v - a3*x - a6*y - a9*pow(u,2) - a9*pow(v,2) - a1*pow(x,2) - a5*pow(y,2) + a7*u*pow(x,2) - a7*pow(u,2)*x - a7*pow(v,2)*x - a8*pow(u,2)*y + a8*v*pow(y,2) - a8*pow(v,2)*y + a1*u*x + a9*u*x + a2*u*y + a4*v*x + a5*v*y + a9*v*y - a2*x*y - a4*x*y + a8*u*x*y + a7*v*x*y;
            l_flowEntry.m_weight_f = 1.0f;
        }
    }

    // lambda, calculate residual
    auto l_calcResidual = [&l_translationMultiRoadNorm](const cv::Matx31f& f_start_r, const cv::Matx31f& f_end_r, const float l_s_f)->float
    {
        const cv::Matx33f A = cv::Matx33f::eye() - l_s_f*l_translationMultiRoadNorm;
        auto l_predEnd = A*f_start_r;
        l_predEnd *= 1.0f/l_predEnd(2);
        return static_cast<float>(cv::norm(f_end_r - l_predEnd));
    };
    float  l_s_f = 0.0f;
    for(int l_iter_i = 0; l_iter_i < 4; ++l_iter_i)
    {
        float l_denominator_f = 0.0f;
        float l_numerator_f   = 0.0f;
        for(int l_idx_i = 0; l_idx_i < f_flowVec_r.size(); ++l_idx_i)
        {
            FlowEntry& l_flowEntry = f_flowVec_r[l_idx_i];
            if(l_roadFlagVec[l_idx_i])
            {
                l_denominator_f += l_flowEntry.m_weight_f*l_denominatorVec[l_idx_i];
                l_numerator_f   += l_flowEntry.m_weight_f*l_numeratorVec[l_idx_i];
            }
        }
        l_s_f = -l_numerator_f/l_denominator_f;
        //update weight
        for(int l_idx_i = 0; l_idx_i < f_flowVec_r.size(); ++l_idx_i)
        {
            FlowEntry& l_flowEntry = f_flowVec_r[l_idx_i];
            if(l_roadFlagVec[l_idx_i])
            {
                /////////////////////////////// debug vis code
//                cv::Mat img;
//                cv::Mat mask = cv::Mat::zeros(f_img_r.size(), f_img_r.type());
//                cv::line(mask, l_flowEntry.m_startPoint, l_flowEntry.getEndPoint(), cv::Scalar(0, 0,256),4);
//                add(f_img_r, mask, img);
//                imshow("test2", img);
//                cvWaitKey(0);
                /////////////////////////////// debug vis code
                const float l_residual = l_calcResidual(l_flowEntry.m_3dRayStartFrame, l_flowEntry.m_3dRayDerotatedEndFrame, l_s_f);
                if(l_residual < 0.25f)
                {
                    l_flowEntry.m_weight_f = 1.0f;
                }
                else
                {
                    l_flowEntry.m_weight_f = 0.25f / l_residual;
                }
            }
        }
    }

    if(std::abs(l_s_f) > std::numeric_limits<float>::min())
    {
        return optional<float>(1.0f/l_s_f);
    }
    else
    {
        return optional<float>();
    }
}

optional<cv::Point2f> estimateFOE(
        std::vector<FlowEntry>& f_flowVec_r,
        const cv::Mat& f_img_r)
{
    /////////////////////////////// debug vis code
    cv::Mat mask = cv::Mat::zeros(f_img_r.size(), f_img_r.type());
    ///////////////////////////////

    // calculate an initial estimate of FOE by calculating average of small flow
    int l_smallFlowCount_i = 0;
    float l_sumX_f = 0.0f;
    float l_sumY_f = 0.0f;
    for(const auto& l_flowEntry : f_flowVec_r)
    {
        if(l_flowEntry.m_length_f <= FOE_ESTIMATION::G_MIN_FLOW_LENGTH_THRESHOLD_F)
        {
            l_smallFlowCount_i++;
            l_sumX_f += l_flowEntry.m_startPoint.x;
            l_sumY_f += l_flowEntry.m_startPoint.y;
        }
    }

    if(l_smallFlowCount_i > FOE_ESTIMATION::G_MIN_INITIAL_SET_NUMBER)
    {
        const float l_initFoeX_f = l_sumX_f/ static_cast<float >(l_smallFlowCount_i);
        const float l_initFoeY_f = l_sumY_f/ static_cast<float >(l_smallFlowCount_i);
        const cv::Point2f l_initialFOE = cv::Point2f(l_initFoeX_f, l_initFoeY_f);

        /////////////////////////////// debug vis code
        cv::circle(mask, cv::Point2f(l_initFoeX_f, l_initFoeY_f), 2, cv::Scalar(0, 0,256),2);
        cv::Mat img;
        add(f_img_r, mask, img);
        imshow("test2", img);
        cvWaitKey(0);
        /////////////////////////////// debug vis code
        //calculate initial weight
        updateWeight(f_flowVec_r, l_initialFOE, f_img_r);
    }

    cv::Point2f l_FOE;
    for(int l_iterIdx_i = 0; l_iterIdx_i < 5; ++l_iterIdx_i)
    {
        float l_sumWiAiSqr_f = 0.0f; // sum(weighti*ai*ai)
        float l_sumWiBiSqr_f = 0.0f; // sum(weighti*bi*bi)
        float l_sumWiAiBi_f  = 0.0f; // sum(weighti*ai*bi)
        float l_sumWiAiCi_f  = 0.0f;
        float l_sumWiBiCi_f  = 0.0f;
        for(const auto& l_flowEntry_r : f_flowVec_r)
        {
            if(l_flowEntry_r.m_length_f > FOE_ESTIMATION::G_MIN_FLOW_LENGTH_THRESHOLD_F)
            {
                l_sumWiAiSqr_f += l_flowEntry_r.m_weight_f * l_flowEntry_r.m_flowLine.m_a_f * l_flowEntry_r.m_flowLine.m_a_f;
                l_sumWiBiSqr_f += l_flowEntry_r.m_weight_f * l_flowEntry_r.m_flowLine.m_b_f * l_flowEntry_r.m_flowLine.m_b_f;
                l_sumWiAiBi_f += l_flowEntry_r.m_weight_f * l_flowEntry_r.m_flowLine.m_a_f * l_flowEntry_r.m_flowLine.m_b_f;
                l_sumWiAiCi_f += l_flowEntry_r.m_weight_f * l_flowEntry_r.m_flowLine.m_a_f * l_flowEntry_r.m_flowLine.m_c_f;
                l_sumWiBiCi_f += l_flowEntry_r.m_weight_f * l_flowEntry_r.m_flowLine.m_b_f * l_flowEntry_r.m_flowLine.m_c_f;
            }
        }
        cv::Mat l_A = cv::Mat(2, 2, CV_32F);
        l_A.at<float>(0,0) = l_sumWiAiSqr_f; l_A.at<float>(1,1) = l_sumWiBiSqr_f;
        l_A.at<float>(0,1) = l_sumWiAiBi_f; l_A.at<float>(1,0) = l_sumWiAiBi_f;
        cv::Mat l_b = cv::Mat(2 , 1, CV_32F);
        l_b.at<float >(0,0) = -l_sumWiAiCi_f;
        l_b.at<float >(1,0) = -l_sumWiBiCi_f;
        const cv::Mat l_FOEMat = l_A.inv()*l_b;
        l_FOE = cv::Point2f(l_FOEMat.at<float>(0,0), l_FOEMat.at<float>(1,0));
        updateWeight(f_flowVec_r, l_FOE, f_img_r);

//        /////////////////////////////// debug vis code
//        cv::circle(mask, l_FOE, 2, cv::Scalar(0, 0,256),4);
//        cv::Mat img;
//        add(f_img_r, mask, img);
//        imshow("test2", img);
//        cvWaitKey(0);
//        /////////////////////////////// debug vis code
    }
    return optional<cv::Point2f>(l_FOE);
}

int main( int argc, char** argv )
{
     // *** Parse and load input images
    char *l_img1Name_p = argv[1];
    char *l_img2Name_p = argv[2];

    cv::Mat l_fisrsImg = cv::imread(l_img1Name_p, CV_LOAD_IMAGE_UNCHANGED);   // Read the file
    cv::Mat l_2ndImg = cv::imread(l_img2Name_p, CV_LOAD_IMAGE_UNCHANGED);   // Read the file
    cv::Mat l_visImag = l_fisrsImg;
    if(l_fisrsImg.channels()==3) {
        cvtColor(l_fisrsImg, l_fisrsImg, CV_RGB2GRAY);
        cvtColor(l_2ndImg, l_2ndImg, CV_RGB2GRAY);
    }

    // init camera
    const cv::FileStorage l_Settings( argv[3], cv::FileStorage::READ);
    assert(l_Settings.isOpened());
    float fx, fy, cx, cy;

    cv::FileNode node = l_Settings["PerspectiveCamera.fx"];
    assert(!node.empty() && node.isReal());
    fx = node.real();
    node = l_Settings["PerspectiveCamera.fy"];
    assert(!node.empty() && node.isReal());
    fy = node.real();
    node = l_Settings["PerspectiveCamera.cx"];
    assert(!node.empty() && node.isReal());
    cx = node.real();
    node = l_Settings["PerspectiveCamera.cy"];
    assert(!node.empty() && node.isReal());
    cy = node.real();
    vector<float> vCamCalib{fx,fy,cx,cy};
    auto l_pinholeCamera = camera::Pinhole(vCamCalib);

    // Distortion parameters
    cv::Mat l_DistCoef(4,1,CV_32F);
    node = l_Settings["PerspectiveCamera.k1"];
    assert(!node.empty() && node.isReal());
    l_DistCoef.at<float>(0) = node.real();
    node = l_Settings["PerspectiveCamera.k2"];
    assert(!node.empty() && node.isReal());
    l_DistCoef.at<float>(1) = node.real();
    node = l_Settings["PerspectiveCamera.p1"];
    assert(!node.empty() && node.isReal());
    l_DistCoef.at<float>(2) = node.real();
    node = l_Settings["PerspectiveCamera.p2"];
    assert(!node.empty() && node.isReal());
    l_DistCoef.at<float>(3) = node.real();

    // load image size
    node = l_Settings["PerspectiveImage.width"];
    assert(!node.empty() && node.isReal());
    const float l_imgWidth = node.real();
    node = l_Settings["PerspectiveImage.height"];
    assert(!node.empty() && node.isReal());
    const float l_imgHeight = node.real();

    // load ORB setting
    int nFeatures = l_Settings["ORBextractor.nFeatures"];
    float fScaleFactor = l_Settings["ORBextractor.scaleFactor"];
    int nLevels = l_Settings["ORBextractor.nLevels"];
    int fIniThFAST = l_Settings["ORBextractor.iniThFAST"];
    int fMinThFAST = l_Settings["ORBextractor.minThFAST"];
    auto l_orbExtractor = ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //////////////////debug code
    cv::FileStorage file("test.ext", cv::FileStorage::WRITE);
//    file << "matName" << l_fisrsImg;
//    auto l1 =l_pinholeCamera.toK();
//    file << "l1" << l1;
//    file.release();
//    cv::Mat l_start = cv::Mat(2,1, CV_32F);
//    l_start.at<float>(0,0) = l_flowEntry.m_startPoint.x;
//    l_start.at<float>(1,0) =  l_flowEntry.m_startPoint.y;
//    file.release();
    //////////////////debug code
    // init frames
    Frame m_firstFrame = Frame(l_fisrsImg, 1, l_orbExtractor,l_pinholeCamera);
    std::vector<cv::Point2f> mvbPrevMatched;
    mvbPrevMatched.resize(m_firstFrame.mvKeys.size());
    for(size_t i=0; i<m_firstFrame.mvKeys.size(); i++)
        mvbPrevMatched[i]=m_firstFrame.mvKeys[i].pt;

    Frame m_2ndFrame = Frame(l_2ndImg, 1, l_orbExtractor, l_pinholeCamera);
    ORBmatcher  l_orbMatcher  = ORBmatcher(1.0, true);

    // match the ORB features
    assert(m_firstFrame.mvKeys.size() > 100);
    assert(m_2ndFrame.mvKeys.size() > 100);
    std::vector<int> mvIniMatches;
    fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
    int nmatches = l_orbMatcher.SearchForInitialization(m_firstFrame, m_2ndFrame, mvbPrevMatched,mvIniMatches,100);
    assert(nmatches > 70);

    // show flow
    std::vector<pair<int, int>> l_Matches12Vec;
    for(size_t i=0, iend=mvIniMatches.size();i<iend; i++)
    {
        if(mvIniMatches[i]>=0)
        {
            l_Matches12Vec.emplace_back(make_pair(i,mvIniMatches[i]));
        }
    }

    /////////////////////////////// debug vis code
    cv::Mat mask = cv::Mat::zeros(l_visImag.size(), l_visImag.type());
    for(auto& l_match_r : l_Matches12Vec)
    {
        cv::KeyPoint l_point1 = m_firstFrame.mvKeys[l_match_r.first];
        cv::KeyPoint l_point2 = m_2ndFrame.mvKeys[l_match_r.second];

        cv::line(mask, l_point1.pt, l_point2.pt, cv::Scalar(0, 0,256),2);
    }
    cv::Mat img;
    add(l_visImag, mask, img);
    imshow("test", img);
    cvWaitKey(0);
    /////////////////////////////// debug vis code

    // calculate the motion between two images by decomposing F(Fundamental matrix) or H(Homography matrix)
    cv::Mat Rcw; // Current Camera Rotation
    cv::Mat tcw; // Current Camera Translation
    vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
    std::vector<cv::Point3f> mvIniP3D;
    Initializer l_Initializer = Initializer(m_firstFrame,1.0,200);
    l_Initializer.m_visImg = l_visImag.clone();
    if(l_Initializer.Initialize(m_2ndFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
    {
//        cv::FileStorage file("test.ext", cv::FileStorage::WRITE);
//        file << "Rcw" << Rcw;
        // deroate keys in frame2 and visualize the derotated flow
        const auto& l_inlierFlagVec_r = l_Initializer.getInlierFlag();
        const cv::Matx33f l_rotation = cv::Matx33f((float*)Rcw.ptr());
        const cv::Matx33f l_derotation = l_rotation.t();
        const cv::Matx31f l_translationEnd2Start = cv::Matx31f((float*)tcw.ptr());
        const cv::Matx33f l_K                    = cv::Matx33f((float*)l_pinholeCamera.toK().clone().ptr());
        const cv::Matx33f l_invK                 = l_K.inv();

        assert(l_inlierFlagVec_r.size() == l_Matches12Vec.size());
        m_2ndFrame.derotateKeys(l_derotation);
        cv::Mat maskDerotaedFlow = cv::Mat::zeros(l_visImag.size(), l_visImag.type());
        std::vector<FlowEntry> l_flowVec;



        for(int l_idx_i = 0; l_idx_i < l_Matches12Vec.size(); ++l_idx_i)
        {
            if(!l_inlierFlagVec_r[l_idx_i]) continue;
            cv::KeyPoint l_point1 = m_firstFrame.mvKeys[l_Matches12Vec[l_idx_i].first];
            cv::Point2f  l_point2 = m_2ndFrame.m_derotatedKeys[l_Matches12Vec[l_idx_i].second];
            l_flowVec.emplace_back(FlowEntry(l_point1.pt, l_point2.x - l_point1.pt.x, l_point2.y - l_point1.pt.y, mvIniP3D[l_Matches12Vec[l_idx_i].first], l_invK));
            /////////////////////////////// debug vis code
            cv::line(maskDerotaedFlow, l_point1.pt, l_point2, cv::Scalar(0, 0,256),2);
            /////////////////////////////// debug vis code
        }

        /////////////////////////////// debug vis code
        cv::Mat img2;
        add(l_visImag, maskDerotaedFlow, img2);
        /////////////////////////////// debug vis code
        const auto l_result = estimateFOE(l_flowVec, img2.clone());
        assert(l_result.isValid());
        const cv::Point2f l_FOE = l_result.getData();

        /////////////////////////////// debug vis code
        cv::line(maskDerotaedFlow, l_FOE, cv::Point2f(0,l_imgHeight), cv::Scalar(0, 0,256),2);
        cv::line(maskDerotaedFlow, l_FOE, cv::Point2f(l_imgWidth,l_imgHeight), cv::Scalar(0, 0,256),2);
        cv::line(maskDerotaedFlow, cv::Point2f(0,l_FOE.y), cv::Point2f(l_imgWidth,l_FOE.y), cv::Scalar(0, 0,256),2);
        add(l_visImag, maskDerotaedFlow, img2);
        imshow("test2", img2);
        cvWaitKey(0);
        /////////////////////////////// debug vis code

        // use the road homography to estimate the scale
        // x_end = (R + t*n'/d)*x_start
        // in the flow entry x_end is already derotated from end frame to start frame:
        // inv(R)*x_end = (I + (inv(R)*t)*n'/d)*x_start => x_derotated_end = (I + derotated_t*n'*s) * x_start, let s = 1/d
        cv::Matx31f l_vanishingLine(0, 1, -l_FOE.y);
        cv::Matx31f l_roadNorm = l_K.t()*l_vanishingLine;
        utility::normlize(l_roadNorm);
        cv::Matx31f l_derotatedTranslation = l_derotation*l_translationEnd2Start;
        ///////////////////////////////debug export code
        file << "l_roadNorm" << cv::Mat(l_roadNorm);
        file<<"l_deroatedTranslation"<<cv::Mat(l_derotatedTranslation);
        file.release();
        ///////////////////////////////debug export code

        const auto l_d_f = estimateScale(l_flowVec, l_FOE, l_derotatedTranslation, l_roadNorm, l_imgWidth, l_imgHeight, l_visImag);
        if(l_d_f.isValid())
        {
            // recover homography matrix
            // test estimated homography
            cv::Matx33f H = cv::Matx33f::eye() - l_derotatedTranslation*l_roadNorm.t()*(1.0f/l_d_f.getData());
            H = l_K*H*l_invK;
            H = H*(1.0f/H(2,2));

            /////////////////////////////// debug vis code
            cv::Mat img;
            cv::Mat mask = cv::Mat::zeros(l_visImag.size(), l_visImag.type());
            for(int l_i = 0; l_i < 12; ++l_i)
            {
                for(int l_j = 0; l_j < 10; ++l_j)
                {
                    cv::Point2f l_start = cv::Point2f(100.0f+l_i*100.0f, 200.0f+l_j*15.0f);
                    cv::Matx31f l_end = H*cv::Matx31f(100.0f+l_i*100.0f, 200.0f+l_j*15.0f, 1.0f);
                    cv::Point2f l_endPoint = cv::Point2f(l_end(0)/l_end(2), l_end(1)/l_end(2));
                    cv::line(mask, l_start, l_endPoint, cv::Scalar(0, 0,256),2);
                }
            }
            add(l_visImag, mask, img);
            imshow("test3", img);
            cvWaitKey(0);
            /////////////////////////////// debug vis code
        }
    }
}


    


