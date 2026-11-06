#pragma once
#include <algorithm>
#include <cstdint>
#include <utility>

/**
 * @file dp_kernel.hpp
 * @brief Building blocks shared by every CPU knapsack DP variant.
 *
 * The classic, tiled-wavefront and DAG solvers use the same inner kernel, so
 * that a timing difference between them can only come from the
 * traversal order, the memory layout or the synchronisation scheme, never from
 * one variant having a better-written inner loop than another.
 */

namespace kp
{

/**
 * @brief Static block partition of the half-open range [0, n) among nt workers.
 *
 * Worker tid receives [begin, end); the blocks are disjoint, cover the whole
 * range and differ in size by at most one element. Arithmetic is done in 64 bit
 * so that n * nt never overflows (the previous `tid * capacity / nt` overflowed
 * a 32-bit int for capacity > ~6.7e7 at 32 threads).
 */
inline std::pair<std::int64_t, std::int64_t> static_block(std::int64_t n, int tid, int nt)
{
	const std::int64_t base = n / nt;
	const std::int64_t rem = n % nt;
	const std::int64_t begin = static_cast<std::int64_t>(tid) * base + std::min<std::int64_t>(tid, rem);
	const std::int64_t len = base + (tid < rem ? 1 : 0);
	return {begin, begin + len};
}

/**
 * @brief One row segment of the 0/1 knapsack recurrence.
 *
 * Computes cur[w] = max(prev[w], prev[w - wi] + vi) for w in [begin, end),
 * with cur[w] = prev[w] where the item does not fit (w < wi). The loop is split
 * at w = wi so that the hot part has no data-dependent branch and the compiler
 * can vectorise it.
 *
 * @param prev row i-1, indexed by absolute capacity
 * @param cur  row i, indexed by absolute capacity
 */
inline void dp_row_segment(const int *__restrict prev, int *__restrict cur, std::int64_t begin, std::int64_t end, int wi,
						   int vi)
{
	const std::int64_t split = std::clamp<std::int64_t>(wi, begin, end);
	for (std::int64_t w = begin; w < split; ++w)
		cur[w] = prev[w];
	for (std::int64_t w = split; w < end; ++w)
		cur[w] = std::max(prev[w], prev[w - wi] + vi);
}

} // namespace kp
