#include "mcts/enumeration_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mcts_enum {

long double EnumerationGeometry::center(
    int k, const std::vector<std::int64_t>& z) const {
    long double c = 0.0L;
    for (int i = k + 1; i < gso_.dimension(); ++i) {
        c -= gso_.mu(i, k) * static_cast<long double>(z[i]);
    }
    return c;
}

GeometryInfo EnumerationGeometry::legal_interval(
    int k,
    long double rho,
    long double radius_sq_scaled,
    const std::vector<std::int64_t>& z) const {
    GeometryInfo info;
    info.k = k;
    info.rho = rho;
    info.radius_sq = radius_sq_scaled;
    if (k < 0) return info;

    const long double rel = std::max<long double>(0.0L, config_.numeric_guard_rel);
    const long double abs = std::max<long double>(0.0L, config_.numeric_guard_abs);
    const long double safe_radius = radius_sq_scaled * (1.0L + rel) + abs;
    const long double safe_rho = std::max(0.0L, rho * (1.0L - rel) - abs);
    if (!(safe_rho < safe_radius)) {
        info.pruned = true;
        return info;
    }

    const long double g_safe = std::max(
        gso_.g(k) * (1.0L - rel) - abs,
        std::numeric_limits<long double>::min());
    info.center = center(k, z);
    info.delta = std::sqrt(std::max(0.0L, (safe_radius - safe_rho) / g_safe));
    info.delta = std::nextafter(info.delta * (1.0L + rel),
                                std::numeric_limits<long double>::infinity());

    const long double lower = info.center - info.delta;
    const long double upper = info.center + info.delta;
    if (lower < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
        upper > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error("legal coefficient interval exceeds int64 range");
    }

    const long double floor_lower = std::floor(lower);
    const long double ceil_upper = std::ceil(upper);
    if (floor_lower >= static_cast<long double>(std::numeric_limits<std::int64_t>::max()) ||
        ceil_upper <= static_cast<long double>(std::numeric_limits<std::int64_t>::min())) {
        info.pruned = true;
        return info;
    }

    info.lo = static_cast<std::int64_t>(floor_lower) + 1;
    info.hi = static_cast<std::int64_t>(ceil_upper) - 1;
    if (info.hi < info.lo) info.pruned = true;

    const unsigned long long count = info.pruned
        ? 0ULL
        : static_cast<unsigned long long>(info.hi - info.lo) + 1ULL;
    if (count > config_.max_legal_actions) {
        throw std::runtime_error(
            "legal action interval exceeds max_legal_actions; increase the explicit safety cap");
    }
    return info;
}

std::vector<CandidateInfo> EnumerationGeometry::candidates(
    const GeometryInfo& info) const {
    if (info.pruned || info.hi < info.lo) return {};
    std::vector<CandidateInfo> out;
    out.reserve(static_cast<std::size_t>(info.hi - info.lo + 1));
    for (std::int64_t action = info.lo;; ++action) {
        const long double offset = static_cast<long double>(action) - info.center;
        CandidateInfo candidate;
        candidate.action = action;
        candidate.normalized_offset = info.delta > 0.0L
            ? static_cast<float>(offset / info.delta) : 0.0f;
        candidate.normalized_abs_offset = std::fabs(candidate.normalized_offset);
        out.push_back(candidate);
        if (action == info.hi) break;
    }
    std::stable_sort(out.begin(), out.end(), [](const CandidateInfo& a, const CandidateInfo& b) {
        if (a.normalized_abs_offset != b.normalized_abs_offset) {
            return a.normalized_abs_offset < b.normalized_abs_offset;
        }
        return a.action < b.action;
    });
    const float denom = out.size() > 1 ? static_cast<float>(out.size() - 1) : 1.0f;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i].se_rank = static_cast<float>(i) / denom;
    }
    return out;
}

long double EnumerationGeometry::next_rho(
    int k, long double rho, long double center_value,
    std::int64_t action) const {
    const long double y = static_cast<long double>(action) - center_value;
    return rho + gso_.g(k) * y * y;
}

}  // namespace mcts_enum
