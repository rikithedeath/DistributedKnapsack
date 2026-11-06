#include "dp_kernel.hpp"
#include "knapsack.hpp"
#include "phases.hpp"
#include <cstdint>
#include <memory>
#include <omp.h>
#include <vector>

/**
 * @brief Classic row-parallel DP with two rolling rows and a decision bit table.
 *
 * Same recurrence, same frontier and same thread partitioning as knapsackdp(),
 * but the value table is replaced by two rows of C+1 ints that are swapped at
 * every item, and backtracking uses one bit per cell ("item i was taken at
 * capacity w"). Memory drops from 4 B to 1/8 B per cell (32x), so this is the
 * strongest classic baseline a tiled solver has to beat.
 *
 * Each thread owns a range of whole 64-bit words of the bit table, so no two
 * threads ever write the same word. Ties keep the item out, exactly like the
 * backtracking of knapsackdp(), so both return the same item set.
 */
KnapsackSolution knapsackdpbits(const std::vector<int> &weights, const std::vector<int> &values, int capacity)
{
	PhaseTimes::instance().clear();
	const int n = static_cast<int>(weights.size());
	const std::int64_t cols = static_cast<std::int64_t>(capacity) + 1;
	const std::int64_t words = (cols + 63) / 64;

	// Default-initialised (untouched) buffers: every page is first-touched by the
	// thread that owns its slice. The two rows take 2 x 4 B x (C+1); the bit
	// table n x (C+1) / 8 B.
	std::unique_ptr<int[]> rowA, rowB;
	std::unique_ptr<std::uint64_t[]> taken;
	{
		ScopedPhase phase("alloc");
		rowA.reset(new int[static_cast<std::size_t>(cols)]);
		rowB.reset(new int[static_cast<std::size_t>(cols)]);
		taken.reset(new std::uint64_t[static_cast<std::size_t>(n) * static_cast<std::size_t>(words)]);
	}

	{
		// Parallel first touch with the compute partition (see knapsackdp()).
		ScopedPhase phase("first_touch");
#pragma omp parallel
		{
			const auto [wbegin, wend] = kp::static_block(words, omp_get_thread_num(), omp_get_num_threads());
			const std::int64_t begin = std::min(wbegin * 64, cols);
			const std::int64_t end = std::min(wend * 64, cols);
			for (std::int64_t w = begin; w < end; ++w)
			{
				rowA[w] = 0;
				rowB[w] = 0;
			}
			for (int i = 0; i < n; ++i)
				for (std::int64_t k = wbegin; k < wend; ++k)
					taken[static_cast<std::size_t>(i) * static_cast<std::size_t>(words) + k] = 0;
		}
	}

	const int *finalRow = nullptr;
	{
		ScopedPhase phase("compute");
#pragma omp parallel
		{
			const int tid = omp_get_thread_num();
			const int nt = omp_get_num_threads();
			const auto [wbegin, wend] = kp::static_block(words, tid, nt);
			int *prev = rowA.get();
			int *cur = rowB.get();

			for (int i = 0; i < n; ++i)
			{
				const int wi = weights[i];
				const int vi = values[i];
				std::uint64_t *bits = taken.get() + static_cast<std::size_t>(i) * static_cast<std::size_t>(words);
				const std::int64_t begin = std::min(wbegin * 64, cols);
				const std::int64_t end = std::min(wend * 64, cols);
				// Values with the shared vectorisable kernel, then the decision bits:
				// the item is taken at w iff it strictly improved the value.
				kp::dp_row_segment(prev, cur, begin, end, wi, vi);
				for (std::int64_t k = wbegin; k < wend; ++k)
				{
					const std::int64_t w0 = k * 64;
					const std::int64_t w1 = std::min(w0 + 64, cols);
					std::uint64_t mask = 0;
					for (std::int64_t w = w0; w < w1; ++w)
						mask |= static_cast<std::uint64_t>(cur[w] > prev[w]) << (w - w0);
					bits[k] = mask;
				}
				std::swap(prev, cur); // every thread swaps, so all agree on the roles
#pragma omp barrier
			}
#pragma omp single
			finalRow = prev;
		}
	}

	ScopedPhase phase("backtrack");
	KnapsackSolution solution{};
	solution.totalValue = finalRow[capacity];
	solution.totalWeight = 0;
	std::int64_t w = capacity;
	for (int i = n - 1; i >= 0; --i)
	{
		const std::uint64_t word = taken[static_cast<std::size_t>(i) * static_cast<std::size_t>(words) + (w / 64)];
		if ((word >> (w % 64)) & 1u)
		{
			solution.items.push_back(i);
			solution.totalWeight += weights[i];
			w -= weights[i];
		}
	}
	return solution;
}
