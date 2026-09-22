#pragma once

#include "ninfer/types.h"

namespace ninfer::targets::qwen3_6 {

struct StartupFeatures {
    bool vision                    = false;
    std::uint32_t vision_max_tokens = 8192;
    // Processor media-item ceiling and aggregate sampled-video pixel budget.
    // Carried here so the frontend can mirror the serve-layer ingress gate.
    std::size_t max_media_items    = 16;
    std::uint64_t max_decoded_video_pixels = 128ULL * 1024ULL * 1024ULL;
    std::uint64_t video_max_pixels = 0; // 0 => follow the artifact's video config
    SpeculativeBackend speculative = SpeculativeBackend::None;
    ProposalHead proposal_head     = ProposalHead::Full;

    bool operator==(const StartupFeatures&) const = default;

    [[nodiscard]] bool speculative_enabled() const noexcept {
        return speculative != SpeculativeBackend::None;
    }

    [[nodiscard]] bool mtp() const noexcept { return speculative == SpeculativeBackend::Mtp; }

    [[nodiscard]] bool dflash() const noexcept { return speculative == SpeculativeBackend::DFlash; }

    [[nodiscard]] bool optimized_proposal() const noexcept {
        return speculative_enabled() && proposal_head == ProposalHead::Optimized;
    }
};

[[nodiscard]] inline StartupFeatures startup_features(const EngineOptions& options) noexcept {
    return StartupFeatures{
        .vision                    = options.enable_vision,
        .vision_max_tokens         = options.vision_max_tokens > 0 ? options.vision_max_tokens : 8192,
        .max_media_items           = options.max_media_items > 0 ? options.max_media_items : 16,
        .max_decoded_video_pixels  = options.max_decoded_video_pixels > 0
                                         ? options.max_decoded_video_pixels
                                         : (128ULL * 1024ULL * 1024ULL),
        .video_max_pixels          = options.video_max_pixels,
        .speculative               = options.speculative.backend,
        .proposal_head             = options.speculative.proposal_head,
    };
}

} // namespace ninfer::targets::qwen3_6
