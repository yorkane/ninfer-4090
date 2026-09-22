#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

// Media preprocessing is memory-bandwidth bound, so spreading a single resize
// over every core of a large host adds thread churn without throughput. Cap the
// per-operation fan-out and let callers pick a serial path when an outer loop
// already provides the parallelism.
inline constexpr unsigned kMediaParallelWorkers = 16;

// Process [0, count) across up to max_workers threads.
// Falls back to a serial loop for small jobs so tiny images do not pay
// thread-startup cost. Callers must keep each index's work independent.
template <typename Body>
void parallel_for_rows(std::size_t count, std::size_t min_rows_per_worker, Body&& body,
                       unsigned max_workers = kMediaParallelWorkers) {
    if (count == 0) { return; }
    if (max_workers <= 1) {
        for (std::size_t i = 0; i < count; ++i) { body(i); }
        return;
    }
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t by_rows =
        std::max<std::size_t>(1, count / std::max<std::size_t>(1, min_rows_per_worker));
    const unsigned workers =
        static_cast<unsigned>(std::min<std::size_t>(std::min(hw, max_workers), by_rows));
    if (workers <= 1) {
        for (std::size_t i = 0; i < count; ++i) { body(i); }
        return;
    }
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (unsigned w = 0; w < workers; ++w) {
        pool.emplace_back([&] {
            for (;;) {
                const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= count) { return; }
                body(i);
            }
        });
    }
    for (auto& t : pool) { t.join(); }
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
