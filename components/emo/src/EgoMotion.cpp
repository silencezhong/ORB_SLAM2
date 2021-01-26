//
// Created by kevin on 24.01.21.
//

#include "EgoMotion.h"

namespace emo
{

void EgoMotion::createInitialMap(const cv::Mat &f_img1_r, const cv::Mat &f_img2_r)
{
    // create key frames
    Frame l_frame1 = Frame(f_img1_r, 1, m_orbExtractor, m_camera_r);
    Frame l_frame2 = Frame(f_img1_r, 2, m_orbExtractor, m_camera_r);
    auto l_transform = estimateMotionBetweenTwoFrames(f_img1_r, f_img2_r);

    l_frame1.SetPose(cv::Mat::eye(4,4,CV_32F));
    assert(l_transform.isValid());
    l_frame2.SetPose(cv::Mat(l_transform.getData()));

    KeyFrame* l_keyFrame1_p = new KeyFrame(l_frame1, &m_map);
    KeyFrame* l_keyFrame2_p = new KeyFrame(l_frame2, &m_map);

    // add key frames to map
    m_map.AddKeyFrame(l_keyFrame1_p);
    m_map.AddKeyFrame(l_keyFrame2_p);

    // Create MapPoints and asscoiate to keyframes

}

utility::optional<cv::Matx44f> EgoMotion::estimateMotionBetweenTwoFrames(const cv::Mat &f_img1_r, const cv::Mat &f_img2_r)
{
    // extract orb features
    Frame m_firstFrame = Frame(f_img1_r, 1, m_orbExtractor, m_camera_r);
    Frame m_2ndFrame = Frame(f_img2_r, 1, m_orbExtractor, m_camera_r);

    std::vector<cv::Point2f> mvbPrevMatched;
    mvbPrevMatched.resize(m_firstFrame.mvKeys.size());
    for(size_t i=0; i<m_firstFrame.mvKeys.size(); i++)
        mvbPrevMatched[i]=m_firstFrame.mvKeys[i].pt;

    assert(m_firstFrame.mvKeys.size() > 100);
    assert(m_2ndFrame.mvKeys.size() > 100);

    // match orb features
    std::vector<int> mvIniMatches;
    fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
    int nmatches = m_orbMatcher.SearchForInitialization(m_firstFrame, m_2ndFrame, mvbPrevMatched,mvIniMatches,100);
    assert(nmatches > 70);
    std::vector<pair<int, int>> l_Matches12Vec;
    for(size_t i=0, iend=mvIniMatches.size();i<iend; i++)
    {
        if(mvIniMatches[i]>=0)
        {
            l_Matches12Vec.emplace_back(make_pair(i,mvIniMatches[i]));
        }
    }

    // estimate rotation and unscaled translation between two frames
    cv::Mat Rcw; // Current Camera Rotation
    cv::Mat tcw; // Current Camera Translation
    vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
    std::vector<cv::Point3f> mvIniP3D;
    if(m_initializer.Initialize(m_firstFrame,1.0,200, m_2ndFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
    {
        const auto& l_inlierFlagVec_r = m_initializer.getInlierFlag();
        const cv::Matx33f l_rotation = cv::Matx33f((float*)Rcw.ptr());
        const cv::Matx33f l_derotation = l_rotation.t();
        const cv::Matx31f l_translationEnd2Start = cv::Matx31f((float*)tcw.ptr());
        const cv::Matx33f l_K                    = cv::Matx33f((float*)m_camera_r.toK().clone().ptr());
        const cv::Matx33f l_invK                 = l_K.inv();

        // calculate derotated flow
        assert(l_inlierFlagVec_r.size() == l_Matches12Vec.size());
        m_2ndFrame.derotateKeys(l_derotation);
        std::vector<FlowEntry> l_flowVec;
        for(int l_idx_i = 0; l_idx_i < l_Matches12Vec.size(); ++l_idx_i)
        {
            if(l_inlierFlagVec_r[l_idx_i])
            {
                cv::KeyPoint l_point1 = m_firstFrame.mvKeys[l_Matches12Vec[l_idx_i].first];
                cv::Point2f  l_point2 = m_2ndFrame.m_derotatedKeys[l_Matches12Vec[l_idx_i].second];
                l_flowVec.emplace_back(FlowEntry(l_point1.pt, l_point2.x - l_point1.pt.x, l_point2.y - l_point1.pt.y, mvIniP3D[l_Matches12Vec[l_idx_i].first], l_invK));
            }
        }

        // estimate FOE
        const auto l_result = estimateFOE(l_flowVec);
        assert(l_result.isValid());
        const cv::Point2f l_FOE = l_result.getData();

        ////////////////////estimate scale
        // use the road homography to estimate the scale
        // x_end = (R + t*n'/d)*x_start
        // in the flow entry x_end is already derotated from end frame to start frame:
        // inv(R)*x_end = (I + (inv(R)*t)*n'/d)*x_start => x_derotated_end = (I + derotated_t*n'*s) * x_start, let s = 1/d
        cv::Matx31f l_vanishingLine(0, 1, -l_FOE.y);
        cv::Matx31f l_roadNorm = l_K.t()*l_vanishingLine;
        utility::normlize(l_roadNorm);
        cv::Matx31f l_derotatedTranslation = l_derotation*l_translationEnd2Start;

        const int l_imgWidth = f_img1_r.cols;
        const int l_imgHeight = f_img1_r.rows;
        const auto l_d_f = estimateScale(l_flowVec, l_FOE, l_derotatedTranslation, l_roadNorm, l_imgWidth, l_imgHeight);
        if(l_d_f.isValid())
        {
            // recover homography matrix
            cv::Matx33f H = cv::Matx33f::eye() - l_derotatedTranslation*l_roadNorm.t()*(1.0f/l_d_f.getData());
            H = l_K*H*l_invK;
            H = H*(1.0f/H(2,2));
            const float l_scaleTo3DWrold_f = m_camera_r.getCameraHeight2Road()/std::abs(l_d_f.getData());
            const cv::Matx31f l_scaledTranslation = l_scaleTo3DWrold_f * l_translationEnd2Start;
            cv::Matx44f l_transform = cv::Matx44f::eye();
            utility::copyRotation2Transform(l_rotation, l_transform);
            utility::copyTranslation2Transform(l_scaledTranslation, l_transform);
            return utility::optional<cv::Matx44f>(l_transform);
        }
    }
    return utility::optional<cv::Matx44f>();
}

utility::optional<cv::Point2f> EgoMotion::estimateFOE( std::vector<FlowEntry>& f_derotatedFlowVec_r)
{
    auto l_updateWeight = [=](
            std::vector<FlowEntry>& f_derotatedFlowVec_r,
            const cv::Point2f& f_foe_r)
    {
        for(auto& l_flowEntry_r : f_derotatedFlowVec_r)
        {
            if(l_flowEntry_r.m_length_f > G_MIN_FLOW_LENGTH_THRESHOLD_F)
            {
                const float l_residual_f = l_flowEntry_r.calculatePointdistance2FlowLine(f_foe_r);
                l_flowEntry_r.m_weight_f = (l_residual_f <= G_M_ESTIMATOR_INLIER_THRESHOLD_F) ? 1.0f : G_M_ESTIMATOR_INLIER_THRESHOLD_F/l_residual_f;
            }
        }
    };

    // calculate an initial estimate of FOE by calculating average of small flow
    int l_smallFlowCount_i = 0;
    float l_sumX_f = 0.0f;
    float l_sumY_f = 0.0f;
    for(const auto& l_flowEntry : f_derotatedFlowVec_r)
    {
        if(l_flowEntry.m_length_f <= G_MIN_FLOW_LENGTH_THRESHOLD_F)
        {
            l_smallFlowCount_i++;
            l_sumX_f += l_flowEntry.m_startPoint.x;
            l_sumY_f += l_flowEntry.m_startPoint.y;
        }
    }

    if(l_smallFlowCount_i > G_MIN_INITIAL_SET_NUMBER)
    {
        const float l_initFoeX_f = l_sumX_f/ static_cast<float >(l_smallFlowCount_i);
        const float l_initFoeY_f = l_sumY_f/ static_cast<float >(l_smallFlowCount_i);
        const cv::Point2f l_initialFOE = cv::Point2f(l_initFoeX_f, l_initFoeY_f);
        //calculate initial weight
        l_updateWeight(f_derotatedFlowVec_r, l_initialFOE);
    }

    cv::Point2f l_FOE;
    for(int l_iterIdx_i = 0; l_iterIdx_i < 5; ++l_iterIdx_i)
    {
        float l_sumWiAiSqr_f = 0.0f; // sum(weighti*ai*ai)
        float l_sumWiBiSqr_f = 0.0f; // sum(weighti*bi*bi)
        float l_sumWiAiBi_f  = 0.0f; // sum(weighti*ai*bi)
        float l_sumWiAiCi_f  = 0.0f;
        float l_sumWiBiCi_f  = 0.0f;
        for(const auto& l_flowEntry_r : f_derotatedFlowVec_r)
        {
            if(l_flowEntry_r.m_length_f > G_MIN_FLOW_LENGTH_THRESHOLD_F)
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
        l_updateWeight(f_derotatedFlowVec_r, l_FOE);
    }

    return utility::optional<cv::Point2f>(l_FOE);
}

utility::optional<float> EgoMotion::estimateScale(
        std::vector<FlowEntry>& f_derotatedFlowVec_r,
        const cv::Point2f& f_FOE_r,
        const cv::Matx31f& f_derotatedTranslation_r,
        const cv::Matx31f& f_roadNorm_r,
        const float f_imgWidth_f,
        const float f_imgHeight_f )
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
    //following is MATLAB symbolic calculation code:
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
    std::vector<float> l_numeratorVec(f_derotatedFlowVec_r.size(), 0.0f); //Vi
    std::vector<float> l_denominatorVec(f_derotatedFlowVec_r.size(), 0.0f); //Ui
    std::vector<bool>  l_roadFlagVec(f_derotatedFlowVec_r.size(), false);

    // determine the road ROI
    CLine l_leftRoadLine(cv::Point2f(0, f_imgHeight_f), f_FOE_r);
    CLine l_rightRoadLine(cv::Point2f(f_imgWidth_f, f_imgHeight_f), f_FOE_r);

    // prepare the calculation
    for(int l_idx_i = 0; l_idx_i < f_derotatedFlowVec_r.size(); ++l_idx_i)
    {
        FlowEntry& l_flowEntry = f_derotatedFlowVec_r[l_idx_i];
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
        for(int l_idx_i = 0; l_idx_i < f_derotatedFlowVec_r.size(); ++l_idx_i)
        {
            FlowEntry& l_flowEntry = f_derotatedFlowVec_r[l_idx_i];
            if(l_roadFlagVec[l_idx_i])
            {
                l_denominator_f += l_flowEntry.m_weight_f*l_denominatorVec[l_idx_i];
                l_numerator_f   += l_flowEntry.m_weight_f*l_numeratorVec[l_idx_i];
            }
        }
        l_s_f = -l_numerator_f/l_denominator_f;
        //update weight
        for(int l_idx_i = 0; l_idx_i < f_derotatedFlowVec_r.size(); ++l_idx_i)
        {
            FlowEntry& l_flowEntry = f_derotatedFlowVec_r[l_idx_i];
            if(l_roadFlagVec[l_idx_i])
            {
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
        return utility::optional<float>(1.0f/l_s_f);
    }
    else
    {
        return utility::optional<float>();
    }
}

}// namespace emo
