// test_bounded_queue.cpp
// 任务 T0.2 单元测试：BoundedQueue 模板类

#include <gtest/gtest.h>

#include <chrono>
#include <thread>
#include <vector>

#include "core/bounded_queue.h"

namespace visionpipe {
namespace {

class BoundedQueueTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ==================== 基本功能测试 ====================

TEST_F(BoundedQueueTest, ConstructorValidCapacity) {
    BoundedQueue<int> queue(10);
    EXPECT_EQ(queue.capacity(), 10);
    EXPECT_EQ(queue.policy(), OverflowPolicy::DROP_OLDEST);
    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);
}

TEST_F(BoundedQueueTest, ConstructorZeroCapacityThrows) {
    EXPECT_THROW(BoundedQueue<int> queue(0), ConfigError);
}

TEST_F(BoundedQueueTest, PushPopBasic) {
    BoundedQueue<int> queue(10);
    queue.push(42);
    EXPECT_EQ(queue.size(), 1);

    auto result = queue.pop();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
    EXPECT_TRUE(queue.empty());
}

TEST_F(BoundedQueueTest, PushPopMultiple) {
    BoundedQueue<int> queue(10);
    for (int i = 0; i < 5; ++i) {
        queue.push(i);
    }
    EXPECT_EQ(queue.size(), 5);

    for (int i = 0; i < 5; ++i) {
        auto result = queue.pop();
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(*result, i);
    }
    EXPECT_TRUE(queue.empty());
}

TEST_F(BoundedQueueTest, PopEmptyReturnsNullopt) {
    BoundedQueue<int> queue(10);
    auto result = queue.pop();
    EXPECT_FALSE(result.has_value());
}

TEST_F(BoundedQueueTest, PopBlockingBasic) {
    BoundedQueue<int> queue(10);
    queue.push(100);

    auto item = queue.pop_blocking();
    EXPECT_EQ(item, 100);
}

// ==================== DROP_OLDEST 溢出策略测试 ====================

TEST_F(BoundedQueueTest, DropOldestOnOverflow) {
    BoundedQueue<int> queue(10, OverflowPolicy::DROP_OLDEST);

    // 入队 15 个元素，容量只有 10
    for (int i = 0; i < 15; ++i) {
        queue.push(i);
    }

    auto stats = queue.stats();
    EXPECT_EQ(stats.dropped_count, 5);  // 15 - 10 = 5 被丢弃
    EXPECT_EQ(stats.current_size, 10);

    // 最老的 0-4 被丢弃，队首应该是 5
    auto first = queue.pop();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(*first, 5);

    // 验证剩余元素是 5-14
    for (int i = 6; i <= 14; ++i) {
        auto result = queue.pop();
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(*result, i);
    }
}

TEST_F(BoundedQueueTest, DropOldestStats) {
    BoundedQueue<int> queue(3, OverflowPolicy::DROP_OLDEST);

    queue.push(1);
    queue.push(2);
    queue.push(3);

    auto stats = queue.stats();
    EXPECT_EQ(stats.total_pushed, 3);
    EXPECT_EQ(stats.dropped_count, 0);

    queue.push(4);  // 应该丢弃 1
    stats = queue.stats();
    EXPECT_EQ(stats.total_pushed, 4);
    EXPECT_EQ(stats.dropped_count, 1);
    EXPECT_EQ(stats.current_size, 3);
}

// ==================== DROP_NEWEST 溢出策略测试 ====================

TEST_F(BoundedQueueTest, DropNewestOnOverflow) {
    BoundedQueue<int> queue(10, OverflowPolicy::DROP_NEWEST);

    // 先填满队列
    for (int i = 0; i < 10; ++i) {
        queue.push(i);
    }

    // 再入队 5 个元素，应该都被丢弃
    for (int i = 10; i < 15; ++i) {
        queue.push(i);
    }

    auto stats = queue.stats();
    EXPECT_EQ(stats.dropped_count, 5);
    EXPECT_EQ(stats.current_size, 10);

    // 队首应该是 0（最老的元素保留）
    auto first = queue.pop();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(*first, 0);
}

// ==================== BLOCK 溢出策略测试 ====================

TEST_F(BoundedQueueTest, BlockOnFull) {
    BoundedQueue<int> queue(2, OverflowPolicy::BLOCK);

    queue.push(1);
    queue.push(2);
    EXPECT_EQ(queue.size(), 2);

    // 异步 pop 以解除阻塞
    std::thread popper([&queue]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        queue.pop();
    });

    auto start = std::chrono::steady_clock::now();
    queue.push(3);  // 应该阻塞直到 pop
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_GE(elapsed, std::chrono::milliseconds(50));
    popper.join();

    // 验证队列状态
    EXPECT_EQ(queue.size(), 2);
    auto item1 = queue.pop();
    auto item2 = queue.pop();
    ASSERT_TRUE(item1.has_value());
    ASSERT_TRUE(item2.has_value());
    EXPECT_EQ(*item1, 2);
    EXPECT_EQ(*item2, 3);
}

TEST_F(BoundedQueueTest, BlockingPopWaitsForData) {
    BoundedQueue<int> queue(10);

    int result = 0;
    std::thread pusher([&queue, &result]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        queue.push(42);
    });

    auto start = std::chrono::steady_clock::now();
    result = queue.pop_blocking();  // 应该阻塞直到有数据
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_GE(elapsed, std::chrono::milliseconds(50));
    EXPECT_EQ(result, 42);
    pusher.join();
}

TEST_F(BoundedQueueTest, PopForTimeout) {
    BoundedQueue<int> queue(10);

    auto start = std::chrono::steady_clock::now();
    auto result = queue.pop_for(std::chrono::milliseconds(100));
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(result.has_value());
    EXPECT_GE(elapsed, std::chrono::milliseconds(80));
}

TEST_F(BoundedQueueTest, PopForSuccess) {
    BoundedQueue<int> queue(10);

    std::thread pusher([&queue]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        queue.push(99);
    });

    auto result = queue.pop_for(std::chrono::milliseconds(200));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 99);
    pusher.join();
}

// ==================== Stop 后 drain 行为测试 ====================

TEST_F(BoundedQueueTest, PopDrainsItemsAfterStop) {
    BoundedQueue<int> queue(10);
    queue.push(1);
    queue.push(2);
    queue.push(3);

    queue.stop();

    // stop 后队列中仍有数据，pop() 应正常返回
    auto r1 = queue.pop();
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(*r1, 1);

    auto r2 = queue.pop();
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(*r2, 2);

    auto r3 = queue.pop();
    ASSERT_TRUE(r3.has_value());
    EXPECT_EQ(*r3, 3);

    // 队列已空，返回 nullopt
    EXPECT_FALSE(queue.pop().has_value());
}

// ==================== Stop/Reset 测试 ====================

TEST_F(BoundedQueueTest, StopWakesBlockedThreads) {
    BoundedQueue<int> queue(2, OverflowPolicy::BLOCK);
    queue.push(1);
    queue.push(2);  // 队列满

    std::atomic<bool> push_completed{false};
    std::thread pusher([&queue, &push_completed]() {
        queue.push(3);  // 会阻塞
        push_completed = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    queue.stop();  // 应该唤醒阻塞的 push
    pusher.join();

    EXPECT_TRUE(push_completed);
}

TEST_F(BoundedQueueTest, StopWakesBlockingPop) {
    BoundedQueue<int> queue(10);  // 空队列

    std::atomic<bool> pop_threw{false};
    std::thread popper([&queue, &pop_threw]() {
        try {
            queue.pop_blocking();
        } catch (const VisionPipeError&) {
            pop_threw = true;
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    queue.stop();
    popper.join();

    EXPECT_TRUE(pop_threw);
}

TEST_F(BoundedQueueTest, ResetClearsQueue) {
    BoundedQueue<int> queue(10);
    queue.push(1);
    queue.push(2);
    queue.push(3);

    EXPECT_EQ(queue.size(), 3);

    queue.stop();
    queue.reset();

    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);

    // 可以继续使用
    queue.push(100);
    EXPECT_EQ(queue.size(), 1);
}

// ==================== 多线程测试 ====================

TEST_F(BoundedQueueTest, MultiProducerMultiConsumer) {
    constexpr size_t NUM_PRODUCERS = 4;
    constexpr size_t NUM_CONSUMERS = 4;
    constexpr size_t ITEMS_PER_PRODUCER = 1000;

    BoundedQueue<int> queue(100, OverflowPolicy::DROP_OLDEST);
    std::atomic<int> total_consumed{0};
    std::atomic<bool> done{false};

    // 生产者线程
    std::vector<std::thread> producers;
    for (size_t p = 0; p < NUM_PRODUCERS; ++p) {
        producers.emplace_back([&queue, p]() {
            for (size_t i = 0; i < ITEMS_PER_PRODUCER; ++i) {
                queue.push(static_cast<int>(p * ITEMS_PER_PRODUCER + i));
            }
        });
    }

    // 消费者线程
    std::vector<std::thread> consumers;
    for (size_t c = 0; c < NUM_CONSUMERS; ++c) {
        consumers.emplace_back([&queue, &total_consumed, &done]() {
            while (!done) {
                auto item = queue.pop();
                if (item.has_value()) {
                    ++total_consumed;
                } else {
                    std::this_thread::yield();
                }
            }
            // 排空队列
            while (queue.pop().has_value()) {
                ++total_consumed;
            }
        });
    }

    // 等待生产者完成
    for (auto& t : producers) {
        t.join();
    }

    // 等待队列排空
    while (!queue.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    done = true;
    for (auto& t : consumers) {
        t.join();
    }

    auto stats = queue.stats();
    // 由于 DROP_OLDEST 策略，可能有些元素被丢弃
    EXPECT_EQ(total_consumed + stats.dropped_count, NUM_PRODUCERS * ITEMS_PER_PRODUCER);
}

// ==================== 统计信息测试 ====================

TEST_F(BoundedQueueTest, StatsTracking) {
    BoundedQueue<int> queue(5, OverflowPolicy::DROP_OLDEST);

    // 入队 10 个元素，容量 5，会丢弃 5 个
    for (int i = 0; i < 10; ++i) {
        queue.push(i);
    }

    // 出队 3 个元素
    for (int i = 0; i < 3; ++i) {
        queue.pop();
    }

    auto stats = queue.stats();
    EXPECT_EQ(stats.capacity, 5);
    EXPECT_EQ(stats.current_size, 2);  // 5 - 3 = 2
    EXPECT_EQ(stats.total_pushed, 10);
    EXPECT_EQ(stats.total_popped, 3);
    EXPECT_EQ(stats.dropped_count, 5);
}

// ==================== 移动语义测试 ====================

TEST_F(BoundedQueueTest, MoveOnlyType) {
    struct MoveOnly {
        int value;
        MoveOnly(int v) : value(v) {}
        MoveOnly(MoveOnly&& other) noexcept : value(other.value) { other.value = 0; }
        MoveOnly& operator=(MoveOnly&& other) noexcept {
            value = other.value;
            other.value = 0;
            return *this;
        }
        MoveOnly(const MoveOnly&) = delete;
        MoveOnly& operator=(const MoveOnly&) = delete;
    };

    BoundedQueue<MoveOnly> queue(10);
    queue.push(MoveOnly(42));

    auto result = queue.pop();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->value, 42);
}

TEST_F(BoundedQueueTest, QueueMove) {
    BoundedQueue<int> queue1(10, OverflowPolicy::DROP_OLDEST);
    queue1.push(1);
    queue1.push(2);

    BoundedQueue<int> queue2(std::move(queue1));
    EXPECT_EQ(queue2.size(), 2);
    EXPECT_EQ(queue2.capacity(), 10);

    auto item = queue2.pop();
    ASSERT_TRUE(item.has_value());
    EXPECT_EQ(*item, 1);
}

}  // namespace
}  // namespace visionpipe
