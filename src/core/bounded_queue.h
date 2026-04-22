#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include "core/error.h"

namespace visionpipe {

/// @brief 队列溢出策略
enum class OverflowPolicy {
    DROP_OLDEST,  ///< 队列满时丢弃最老元素
    DROP_NEWEST,  ///< 队列满时丢弃新元素
    BLOCK         ///< 队列满时阻塞直到有空间
};

/// @brief 队列统计信息
struct QueueStats {
    size_t capacity;           ///< 队列容量
    size_t current_size;       ///< 当前元素数量
    uint64_t total_pushed;     ///< 累计入队次数
    uint64_t total_popped;     ///< 累计出队次数
    uint64_t dropped_count;    ///< 累计丢弃次数
};

/// @brief 线程安全的有界队列
/// @tparam T 元素类型，必须可移动
template <typename T>
class BoundedQueue {
public:
    /// @brief 构造函数
    /// @param capacity 队列容量，必须 > 0
    /// @param policy 溢出策略
    BoundedQueue(size_t capacity, OverflowPolicy policy = OverflowPolicy::DROP_OLDEST)
        : capacity_(capacity)
        , policy_(policy)
        , stopped_(false) {
        if (capacity == 0) {
            throw ConfigError("BoundedQueue capacity must be > 0");
        }
    }

    /// @brief 析构函数，停止所有阻塞操作
    ~BoundedQueue() { stop(); }

    // 禁止拷贝
    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    // 允许移动
    BoundedQueue(BoundedQueue&& other) noexcept
        : capacity_(other.capacity_)
        , policy_(other.policy_)
        , stopped_(other.stopped_.load())
        , total_pushed_(other.total_pushed_.load())
        , total_popped_(other.total_popped_.load())
        , dropped_count_(other.dropped_count_.load()) {
        std::lock_guard<std::mutex> lock(other.mutex_);
        queue_ = std::move(other.queue_);
    }

    BoundedQueue& operator=(BoundedQueue&& other) noexcept {
        if (this != &other) {
            std::scoped_lock lock(mutex_, other.mutex_);
            capacity_ = other.capacity_;
            policy_ = other.policy_;
            stopped_ = other.stopped_.load();
            total_pushed_ = other.total_pushed_.load();
            total_popped_ = other.total_popped_.load();
            dropped_count_ = other.dropped_count_.load();
            queue_ = std::move(other.queue_);
        }
        return *this;
    }

    /// @brief 入队
    /// @param item 要入队的元素
    /// @note DROP_OLDEST/DROP_NEWEST 模式下永不阻塞，BLOCK 模式下队列满时阻塞
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (stopped_) {
            return;
        }

        if (queue_.size() >= capacity_) {
            switch (policy_) {
                case OverflowPolicy::DROP_OLDEST: {
                    // 丢弃最老元素
                    queue_.pop();
                    ++dropped_count_;
                    break;
                }
                case OverflowPolicy::DROP_NEWEST: {
                    // 丢弃新元素（不入队）
                    ++dropped_count_;
                    return;
                }
                case OverflowPolicy::BLOCK: {
                    // 阻塞直到有空间或停止
                    not_full_.wait(lock, [this] { return queue_.size() < capacity_ || stopped_; });
                    if (stopped_) {
                        return;
                    }
                    break;
                }
            }
        }

        queue_.push(std::move(item));
        ++total_pushed_;
        not_empty_.notify_one();
    }

    /// @brief 非阻塞出队
    /// @return 队列为空时返回 nullopt，否则返回队首元素
    std::optional<T> pop() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (stopped_ || queue_.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();
        ++total_popped_;
        not_full_.notify_one();
        return item;
    }

    /// @brief 阻塞出队
    /// @return 队列中的元素
    /// @note 队列为空时阻塞直到有数据或停止
    T pop_blocking() {
        std::unique_lock<std::mutex> lock(mutex_);

        not_empty_.wait(lock, [this] { return !queue_.empty() || stopped_; });

        if (stopped_ && queue_.empty()) {
            throw VisionPipeError("BoundedQueue stopped while waiting for item");
        }

        T item = std::move(queue_.front());
        queue_.pop();
        ++total_popped_;
        not_full_.notify_one();
        return item;
    }

    /// @brief 带超时的阻塞出队
    /// @param timeout 超时时间
    /// @return 超时或停止时返回 nullopt，否则返回元素
    template <typename Rep, typename Period>
    std::optional<T> pop_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!not_empty_.wait_for(lock, timeout, [this] { return !queue_.empty() || stopped_; })) {
            return std::nullopt;  // 超时
        }

        if (stopped_ && queue_.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();
        ++total_popped_;
        not_full_.notify_one();
        return item;
    }

    /// @brief 获取队列统计信息
    QueueStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return QueueStats{
            .capacity = capacity_,
            .current_size = queue_.size(),
            .total_pushed = total_pushed_,
            .total_popped = total_popped_,
            .dropped_count = dropped_count_
        };
    }

    /// @brief 队列是否为空
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    /// @brief 获取当前元素数量
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    /// @brief 停止队列，唤醒所有阻塞的等待者
    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    /// @brief 重置队列（清空元素并重置停止状态）
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            queue_.pop();
        }
        stopped_ = false;
    }

    /// @brief 获取容量
    size_t capacity() const { return capacity_; }

    /// @brief 获取溢出策略
    OverflowPolicy policy() const { return policy_; }

private:
    size_t capacity_;
    OverflowPolicy policy_;
    std::atomic<bool> stopped_;

    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;

    // 统计计数器（使用 atomic 保证 stats() 的线程安全读取）
    mutable std::atomic<uint64_t> total_pushed_{0};
    mutable std::atomic<uint64_t> total_popped_{0};
    mutable std::atomic<uint64_t> dropped_count_{0};
};

}  // namespace visionpipe
