
#include "gtest/gtest.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camera/inc/PinholeCamera.h"

#include "EgoMotion.h"

class EgoMotionTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        // load images
        m_fisrsImg = cv::imread("/home/kevin/SLAM/workbench/ORB_SLAM2/components/emo/test/translation1.png", CV_LOAD_IMAGE_UNCHANGED);
        m_2ndImg   = cv::imread("/home/kevin/SLAM/workbench/ORB_SLAM2/components/emo/test/translation2.png",  CV_LOAD_IMAGE_UNCHANGED);
        if(m_fisrsImg.channels()==3) {
            cvtColor(m_fisrsImg, m_fisrsImg, CV_RGB2GRAY);
            cvtColor(m_2ndImg, m_2ndImg, CV_RGB2GRAY);
        }

        // init camera
        const cv::FileStorage l_Settings( "/home/kevin/SLAM/workbench/ORB_SLAM2/Examples/Monocular/CameraConfig.yaml", cv::FileStorage::READ);
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
        node = l_Settings["PerspectiveCamera.height2road"];
        assert(!node.empty() && node.isReal());
        const float l_cameraHeight_f = node.real();
        vector<float> vCamCalib{fx,fy,cx,cy};
        m_pinholeCamera = camera::Pinhole(vCamCalib, l_cameraHeight_f);

        // load orb extractor parameters
        nFeatures = l_Settings["ORBextractor.nFeatures"];
        fScaleFactor = l_Settings["ORBextractor.scaleFactor"];
        nLevels = l_Settings["ORBextractor.nLevels"];
        fIniThFAST = l_Settings["ORBextractor.iniThFAST"];
        fMinThFAST = l_Settings["ORBextractor.minThFAST"];
    }

    // virtual void TearDown() {}
    cv::Mat m_fisrsImg;
    cv::Mat m_2ndImg;
    camera::Pinhole m_pinholeCamera;
    int nFeatures;
    float fScaleFactor;
    int nLevels;
    int fIniThFAST;
    int fMinThFAST;
};
//
TEST_F(EgoMotionTest, DISABLED_estimateMotionBetweenTwoFrames)
{
    emo::EgoMotion l_egomotion(nFeatures, fScaleFactor, nLevels,fIniThFAST,  fMinThFAST,m_pinholeCamera );
    auto l_returnClass = l_egomotion.estimateMotionBetweenTwoFrames(m_fisrsImg, m_2ndImg);
    ASSERT_TRUE(l_returnClass.isValid());
    auto l_transform = l_returnClass.getData();
    cv::Vec3f l_translation(l_transform(0,3), l_transform(1,3), l_transform(2,3));
    const float l_translationGroundTruth_f = 0.55f;
    EXPECT_TRUE(std::abs(cv::norm(l_translation)- l_translationGroundTruth_f) < 0.05f);
}

TEST_F(EgoMotionTest, createInitialMap)
{
    emo::EgoMotion l_egomotion(nFeatures, fScaleFactor, nLevels,fIniThFAST,  fMinThFAST,m_pinholeCamera );
    l_egomotion.createInitialMap(m_fisrsImg, m_2ndImg);
}


int main (int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    //::testing::GTEST_FLAG(filter) = "IsPrimeTest.*:FactorialTest.*";
    return RUN_ALL_TESTS();

    return 0;
}
