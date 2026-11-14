// SPDX-License-Identifier: Apache-2.0
// FIFO and buffer-ownership checks only; no CANN event or DMA timing model.
#pragma once
#include "selected_bitmap_host_stub.h"
#include <deque>
#include <exception>

#define __global__
#define KERNEL_TASK_TYPE_DEFAULT(...)

namespace AscendC {
using TPosition = QuePosition;
constexpr int PIPE_ALL = 1;
inline uint32_t GetBlockNum() { return 32; }
template<class T> using DataCopyPadExtParams = PaddingParams;
inline size_t copyQueuePeak = 0;
inline size_t copyQueueDepth = 0;

template<QuePosition, QuePosition, uint32_t Depth> class TQueBind {
public:
    void Init(size_t count, size_t bytes) {
        if (count != Depth) throw std::logic_error("Queue depth mismatch");
        bytes_ = bytes;
        for (size_t i = 0; i < count; ++i) {
            buffers_.push_back(std::make_shared<std::vector<uint64_t>>((bytes + 7) / 8));
            states_.push_back(State::Free);
        }
    }
    template<class T> LocalTensor<T> AllocTensor() {
        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (states_[i] == State::Free) {
                states_[i] = State::Allocated;
                return TensorAt<T>(i);
            }
        }
        throw std::logic_error("Reused an occupied copy buffer");
    }
    template<class T> void EnQue(LocalTensor<T> tensor) {
        const size_t i = Index(tensor);
        if (states_[i] != State::Allocated) throw std::logic_error("EnQue without ownership");
        states_[i] = State::Queued;
        fifo_.push_back(i);
        copyQueueDepth = fifo_.size();
        copyQueuePeak = std::max(copyQueuePeak, fifo_.size());
    }
    template<class T> LocalTensor<T> DeQue() {
        if (fifo_.empty()) throw std::logic_error("Dequeued an empty copy queue");
        const size_t i = fifo_.front();
        fifo_.pop_front();
        copyQueueDepth = fifo_.size();
        if (states_[i] != State::Queued) throw std::logic_error("Invalid queued buffer");
        states_[i] = State::Dequeued;
        return TensorAt<T>(i);
    }
    template<class T> void FreeTensor(LocalTensor<T> tensor) {
        const size_t i = Index(tensor);
        if (states_[i] != State::Dequeued) throw std::logic_error("Freed an in-flight copy buffer");
        states_[i] = State::Free;
    }
    ~TQueBind() noexcept(false) {
        copyQueueDepth = 0;
        if (std::uncaught_exceptions() == 0 &&
            std::any_of(states_.begin(), states_.end(), [](State s) { return s != State::Free; })) {
            throw std::logic_error("Copy queue was not drained");
        }
    }

private:
    enum class State { Free, Allocated, Queued, Dequeued };
    template<class T> LocalTensor<T> TensorAt(size_t i) {
        return {reinterpret_cast<uint8_t*>(buffers_[i]->data()), bytes_, buffers_[i]};
    }
    template<class T> size_t Index(LocalTensor<T> tensor) const {
        const auto* address = tensor.Access(0, false);
        for (size_t i = 0; i < buffers_.size(); ++i) {
            if (address == reinterpret_cast<uint8_t*>(buffers_[i]->data())) return i;
        }
        throw std::logic_error("Foreign tensor passed to copy queue");
    }
    size_t bytes_ = 0;
    std::vector<std::shared_ptr<std::vector<uint64_t>>> buffers_;
    std::vector<State> states_;
    std::deque<size_t> fifo_;
};
} // namespace AscendC
