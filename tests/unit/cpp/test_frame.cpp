// test_frame.cpp
// 任务 T0.2 单元测试：Frame 结构（Classification、classifications、user_data map）

#include <gtest/gtest.h>

#include <any>
#include <string>

#include "core/frame.h"

namespace visionpipe {
namespace {

class FrameTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ==================== Classification 结构测试 ====================

TEST_F(FrameTest, ClassificationDefaultConstruction) {
    Classification c;
    EXPECT_EQ(c.detection_index, -1);
    EXPECT_EQ(c.class_id, 0);
    EXPECT_EQ(c.confidence, 0.0f);
}

TEST_F(FrameTest, ClassificationCustomValues) {
    Classification c;
    c.detection_index = 7;
    c.class_id = 42;
    c.confidence = 0.875f;

    EXPECT_EQ(c.detection_index, 7);
    EXPECT_EQ(c.class_id, 42);
    EXPECT_EQ(c.confidence, 0.875f);
}

// ==================== Frame.classifications 测试 ====================

TEST_F(FrameTest, ClassificationsDefaultEmpty) {
    Frame frame;
    EXPECT_EQ(frame.classifications.size(), 0u);
    EXPECT_TRUE(frame.classifications.empty());
}

TEST_F(FrameTest, ClassificationsAddAndRead) {
    Frame frame;

    Classification c1;
    c1.detection_index = 0;
    c1.class_id = 3;
    c1.confidence = 0.9f;
    frame.classifications.push_back(c1);

    Classification c2;
    c2.detection_index = 1;
    c2.class_id = 5;
    c2.confidence = 0.75f;
    frame.classifications.push_back(c2);

    Classification c3;  // 整图分类，detection_index 保持 -1
    c3.class_id = 10;
    c3.confidence = 0.5f;
    frame.classifications.push_back(c3);

    ASSERT_EQ(frame.classifications.size(), 3u);

    EXPECT_EQ(frame.classifications[0].detection_index, 0);
    EXPECT_EQ(frame.classifications[0].class_id, 3);
    EXPECT_EQ(frame.classifications[0].confidence, 0.9f);

    EXPECT_EQ(frame.classifications[1].detection_index, 1);
    EXPECT_EQ(frame.classifications[1].class_id, 5);
    EXPECT_EQ(frame.classifications[1].confidence, 0.75f);

    EXPECT_EQ(frame.classifications[2].detection_index, -1);
    EXPECT_EQ(frame.classifications[2].class_id, 10);
    EXPECT_EQ(frame.classifications[2].confidence, 0.5f);
}

TEST_F(FrameTest, ClassificationsClearedByFrameClear) {
    Frame frame;

    Classification c;
    c.detection_index = 2;
    c.class_id = 8;
    c.confidence = 0.6f;
    frame.classifications.push_back(c);
    frame.classifications.push_back(c);

    ASSERT_EQ(frame.classifications.size(), 2u);

    frame.clear();

    EXPECT_EQ(frame.classifications.size(), 0u);
    EXPECT_TRUE(frame.classifications.empty());
}

// ==================== Frame.user_data 测试 (map<string, any>) ====================

TEST_F(FrameTest, UserDataDefaultEmpty) {
    Frame frame;
    EXPECT_EQ(frame.user_data.size(), 0u);
    EXPECT_TRUE(frame.user_data.empty());
}

TEST_F(FrameTest, UserDataInsertAndRetrieveString) {
    Frame frame;
    frame.user_data["label"] = std::string("hello");

    ASSERT_EQ(frame.user_data.count("label"), 1u);
    std::string value;
    ASSERT_NO_THROW(value = std::any_cast<std::string>(frame.user_data["label"]));
    EXPECT_EQ(value, "hello");
}

TEST_F(FrameTest, UserDataInsertAndRetrieveInt) {
    Frame frame;
    frame.user_data["count"] = 123;

    ASSERT_EQ(frame.user_data.count("count"), 1u);
    int value = 0;
    ASSERT_NO_THROW(value = std::any_cast<int>(frame.user_data["count"]));
    EXPECT_EQ(value, 123);
}

TEST_F(FrameTest, UserDataInsertAndRetrieveDouble) {
    Frame frame;
    frame.user_data["score"] = 3.14159;

    ASSERT_EQ(frame.user_data.count("score"), 1u);
    double value = 0.0;
    ASSERT_NO_THROW(value = std::any_cast<double>(frame.user_data["score"]));
    EXPECT_EQ(value, 3.14159);
}

TEST_F(FrameTest, UserDataMultipleKeysCoexist) {
    Frame frame;
    frame.user_data["name"] = std::string("frame_a");
    frame.user_data["index"] = 42;
    frame.user_data["weight"] = 0.5;

    EXPECT_EQ(frame.user_data.size(), 3u);
    EXPECT_EQ(frame.user_data.count("name"), 1u);
    EXPECT_EQ(frame.user_data.count("index"), 1u);
    EXPECT_EQ(frame.user_data.count("weight"), 1u);

    EXPECT_EQ(std::any_cast<std::string>(frame.user_data["name"]), "frame_a");
    EXPECT_EQ(std::any_cast<int>(frame.user_data["index"]), 42);
    EXPECT_EQ(std::any_cast<double>(frame.user_data["weight"]), 0.5);
}

TEST_F(FrameTest, UserDataClearedByFrameClear) {
    Frame frame;
    frame.user_data["a"] = 1;
    frame.user_data["b"] = std::string("x");
    frame.user_data["c"] = 2.5;

    ASSERT_EQ(frame.user_data.size(), 3u);

    frame.clear();

    EXPECT_EQ(frame.user_data.size(), 0u);
    EXPECT_TRUE(frame.user_data.empty());
    EXPECT_EQ(frame.user_data.count("a"), 0u);
    EXPECT_EQ(frame.user_data.count("b"), 0u);
    EXPECT_EQ(frame.user_data.count("c"), 0u);
}

TEST_F(FrameTest, UserDataOverwriteExistingKey) {
    Frame frame;
    frame.user_data["key"] = 100;

    ASSERT_EQ(frame.user_data.size(), 1u);
    EXPECT_EQ(std::any_cast<int>(frame.user_data["key"]), 100);

    // 同 key 覆盖（同类型）
    frame.user_data["key"] = 200;
    EXPECT_EQ(frame.user_data.size(), 1u);
    EXPECT_EQ(std::any_cast<int>(frame.user_data["key"]), 200);

    // 同 key 覆盖（不同类型）
    frame.user_data["key"] = std::string("replaced");
    EXPECT_EQ(frame.user_data.size(), 1u);
    EXPECT_EQ(std::any_cast<std::string>(frame.user_data["key"]), "replaced");
}

}  // namespace
}  // namespace visionpipe
