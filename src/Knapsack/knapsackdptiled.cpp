#include "dp_kernel.hpp"
#include "knapsack.hpp"
#include "phases.hpp"
#include "time.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <omp.h>

namespace
{
using DpTable = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}

/**
 * @brief Tiled DP executed as an explicit anti-diagonal wavefront, without a graph.
 *
 * The (n+1) x (C+1) table is the same contiguous row-major table used by
 * knapsackdp(); it is cut into tiles of item_block rows by cap_block columns and
 * the tiles are processed by anti-diagonals d = b + q, all tiles of a diagonal in
 * parallel. This schedule is valid for any weights:
 *   - row i of tile (b, q) reads row i-1 at columns w and w - w_i <= w, i.e. in
 *     tile (b, q') or, for the first row of the block, (b-1, q') with q' <= q;
 *   - (b, q') with q' < q lies on diagonal b + q' < d, (b, q) itself is being
 *     computed row by row, and (b-1, q') lies on diagonal b - 1 + q' < d.
 * With w_i >= 1 every tile also depends on its left neighbour, so the DAG solver
 * finds exactly these levels (level(b, q) = b + q): this variant has the same
 * parallelism as knapsackdpdag() but none of its graph construction, per-tile
 * allocations or boundary-row copies, which isolates the cost of the graph.
 *
 * With nt = 1 it is the serial "tiled traversal" of the table: comparing it to
 * knapsackdp() at one thread measures the pure locality effect of tiling.
 */
KnapsackSolution knapsackdptiled(const std::vector<int> &weights, const std::vector<int> &values, int capacity,
								 int item_block, int cap_block)
{
	PhaseTimes::instance().clear();
	const int n = static_cast<int>(weights.size());
	const std::int64_t cols = static_cast<std::int64_t>(capacity) + 1;
	if (item_block <= 0)
		item_block = 10;
	if (cap_block <= 0)
		cap_block = std::max<int>(1, static_cast<int>(cols / 10));
	const int nb = (n + item_block - 1) / item_block;
	const std::int64_t nq = (cols + cap_block - 1) / cap_block;

	DpTable dp;
	{
		ScopedPhase phase("alloc");
		dp.resize(n + 1, cols);
	}

	{
		// Parallel first touch (also zeroes row 0), as in knapsackdp(). Tiles of
		// one diagonal go to different threads, so the placement can only be
		// approximate here: it follows the column partition.
		ScopedPhase phase("first_touch");
#pragma omp parallel
		{
			const auto [begin, end] = kp::static_block(cols, omp_get_thread_num(), omp_get_num_threads());
			for (int i = 0; i <= n; ++i)
			{
				int *row = dp.row(i).data();
				for (std::int64_t w = begin; w < end; ++w)
					row[w] = 0;
			}
		}
	}

	{
		ScopedPhase phase("compute");
		START_BLOCK("KnapsackDPTiled::Compute");

		const std::int64_t diagonals = nb == 0 ? 0 : nb + nq - 1;
		for (std::int64_t d = 0; d < diagonals; ++d)
		{
			const int b_lo = static_cast<int>(std::max<std::int64_t>(0, d - (nq - 1)));
			const int b_hi = static_cast<int>(std::min<std::int64_t>(nb - 1, d));
#pragma omp parallel for schedule(static)
			for (int b = b_lo; b <= b_hi; ++b)
			{
				const std::int64_t q = d - b;
				const std::int64_t lo = q * cap_block;
				const std::int64_t hi = std::min<std::int64_t>(lo + cap_block, cols);
				const int i_start = b * item_block;
				const int i_end = std::min(i_start + item_block, n);
				for (int i = i_start; i < i_end; ++i)
					kp::dp_row_segment(dp.row(i).data(), dp.row(i + 1).data(), lo, hi, weights[i], values[i]);
			}
		}
		END_BLOCK("KnapsackDPTiled::Compute");
	}

	ScopedPhase phase("backtrack");
	std::vector<int> includedItems;
	int totalValue = dp(n, capacity);
	int totalWeight = 0;
	int w = capacity;
	for (int i = n; i > 0 && totalValue > 0; --i)
	{
		if (totalValue != dp(i - 1, w))
		{
			includedItems.push_back(i - 1);
			totalValue -= values[i - 1];
			w -= weights[i - 1];
			totalWeight += weights[i - 1];
		}
	}
	return {includedItems, dp(n, capacity), totalWeight};
}
