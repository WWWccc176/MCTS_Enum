#include "mcts/enumeration_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mcts_enum {

long double EnumerationGeometry::center_from_prefix(
    int k,
    const std::vector<std::int64_t>& prefix_high_to_low) const {
    long double c = 0.0L;
    const int n = gso_.dimension();
    for (std::size_t p = 0; p < prefix_high_to_low.size(); ++p) {
        const int i = n - 1 - static_cast<int>(p);
        if (i <= k) break;
        c -= gso_.mu(i, k) * static_cast<long double>(prefix_high_to_low[p]);
    }
    return c;
}

GeometryInfo EnumerationGeometry::legal_interval(
    int k,
    long double rho,
    long double radius_sq_scaled,
    const std::vector<std::int64_t>& prefix_high_to_low) const {
    return legal_interval_from_center(
        k, rho, radius_sq_scaled, center_from_prefix(k, prefix_high_to_low));
}

GeometryInfo EnumerationGeometry::legal_interval_from_center(
    int k,
    long double rho,
    long double radius_sq_scaled,
    long double center) const {
    GeometryInfo info;
    info.k = k;
    info.rho = rho;
    info.radius_sq = radius_sq_scaled;
    info.center = center;
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
    info.delta = std::sqrt(std::max(0.0L, (safe_radius - safe_rho) / g_safe));
    info.delta = std::nextafter(
        info.delta * (1.0L + rel),
        std::numeric_limits<long double>::infinity());

    const long double lower = center - info.delta;
    const long double upper = center + info.delta;
    if (lower < static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
        upper > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error("legal coefficient interval exceeds int64 range");
    }

    info.lo = static_cast<std::int64_t>(std::floor(lower)) + 1;
    info.hi = static_cast<std::int64_t>(std::ceil(upper)) - 1;
    if (info.hi < info.lo) info.pruned = true;
    return info;
}

long double EnumerationGeometry::next_rho(
    int k, long double rho, long double center, std::int64_t action) const {
    const long double y = static_cast<long double>(action) - center;
    return rho + gso_.g(k) * y * y;
}

}  // namespace mcts_enum
