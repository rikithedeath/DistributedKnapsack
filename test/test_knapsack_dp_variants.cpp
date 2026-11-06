// Cross-checks every CPU DP variant against an independent reference, for
// several thread counts, including the corner cases that break naive row
// partitioning (capacity smaller than the number of threads).
#include "Knapsack/dp_kernel.hpp"
#include "Knapsack/knapsack.hpp"
#include "Knapsack/knapsackdpdag_impl.hpp"
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <omp.h>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace
{

/// Textbook in-place 1-D DP: shares no code with the solvers under test.
int reference_value(const std::vector<int> &w, const std::vector<int> &v, int capacity)
{
	std::vector<int> row(static_cast<std::size_t>(capacity) + 1, 0);
	for (std::size_t i = 0; i < w.size(); ++i)
		for (int c = capacity; c >= w[i]; --c)
			row[c] = std::max(row[c], row[c - w[i]] + v[i]);
	return row[capacity];
}

/// The item set must be distinct, fit, and add up to the reported totals.
void expect_valid(const KnapsackSolution &s, const std::vector<int> &w, const std::vector<int> &v, int capacity,
				  const std::string &what)
{
	std::set<int> seen;
	long long tw = 0, tv = 0;
	for (int idx : s.items)
	{
		ASSERT_GE(idx, 0) << what;
		ASSERT_LT(idx, static_cast<int>(w.size())) << what;
		ASSERT_TRUE(seen.insert(idx).second) << what << ": duplicated item " << idx;
		tw += w[idx];
		tv += v[idx];
	}
	EXPECT_LE(tw, capacity) << what;
	EXPECT_EQ(tw, s.totalWeight) << what;
	EXPECT_EQ(tv, s.totalValue) << what;
}

struct Instance
{
	std::vector<int> weights, values;
	int capacity;
};

Instance random_instance(std::mt19937 &rng, int max_n, int max_capacity, int max_weight)
{
	Instance in;
	const int n = static_cast<int>(rng() % static_cast<unsigned>(max_n + 1));
	in.capacity = static_cast<int>(rng() % static_cast<unsigned>(max_capacity + 1));
	in.weights.resize(n);
	in.values.resize(n);
	for (int i = 0; i < n; ++i)
	{
		in.weights[i] = 1 + static_cast<int>(rng() % static_cast<unsigned>(max_weight));
		in.values[i] = 1 + static_cast<int>(rng() % 50u);
	}
	return in;
}

class ThreadCount : public ::testing::TestWithParam<int>
{
  protected:
	void SetUp() override
	{
		saved_ = omp_get_max_threads();
		omp_set_num_threads(GetParam());
	}
	void TearDown() override
	{
		omp_set_num_threads(saved_);
	}

  private:
	int saved_ = 1;
};

} // namespace

TEST(StaticBlock, PartitionIsDisjointAndComplete)
{
	for (std::int64_t n : {0, 1, 2, 3, 7, 64, 1000, 1001})
	{
		for (int nt : {1, 2, 3, 4, 7, 32, 80})
		{
			std::int64_t expected_begin = 0;
			for (int t = 0; t < nt; ++t)
			{
				const auto [b, e] = kp::static_block(n, t, nt);
				EXPECT_EQ(b, expected_begin) << "n=" << n << " nt=" << nt << " t=" << t;
				EXPECT_LE(b, e);
				EXPECT_LE(e - b, n / nt + 1);
				expected_begin = e;
			}
			EXPECT_EQ(expected_begin, n) << "n=" << n << " nt=" << nt;
		}
	}
}

TEST(StaticBlock, NoOverflowForLargeCapacities)
{
	// tid * capacity overflowed a 32-bit int for capacity > ~6.7e7 at 32 threads.
	const std::int64_t n = 2'000'000'001LL;
	const auto [b, e] = kp::static_block(n, 79, 80);
	EXPECT_EQ(e, n);
	EXPECT_GT(b, 0);
}

TEST_P(ThreadCount, AllVariantsMatchReferenceOnRandomInstances)
{
	std::mt19937 rng(20261106u + static_cast<unsigned>(GetParam()));
	for (int trial = 0; trial < 60; ++trial)
	{
		// Small capacities on purpose: C + 1 < threads exercises empty slices.
		const int max_capacity = (trial % 3 == 0) ? 4 : 400;
		const auto in = random_instance(rng, 40, max_capacity, 60);
		const int ref = reference_value(in.weights, in.values, in.capacity);
		const std::string tag = "trial=" + std::to_string(trial) + " n=" + std::to_string(in.weights.size()) +
								" C=" + std::to_string(in.capacity) + " threads=" + std::to_string(GetParam());

		const auto dp = knapsackdp(in.weights, in.values, in.capacity);
		EXPECT_EQ(dp.totalValue, ref) << "knapsackdp " << tag;
		expect_valid(dp, in.weights, in.values, in.capacity, "knapsackdp " + tag);

		const auto bits = knapsackdpbits(in.weights, in.values, in.capacity);
		EXPECT_EQ(bits.totalValue, ref) << "knapsackdpbits " << tag;
		expect_valid(bits, in.weights, in.values, in.capacity, "knapsackdpbits " + tag);
		// Same tie-breaking as the classic backtracking: same item set.
		EXPECT_EQ(bits.items, dp.items) << "knapsackdpbits " << tag;

		for (const auto &[ib, cb] : std::vector<std::pair<int, int>>{{1, 1}, {3, 7}, {10, 64}, {25, 1000}})
		{
			const std::string blocks = " item_block=" + std::to_string(ib) + " cap_block=" + std::to_string(cb);
			const auto tiled = knapsackdptiled(in.weights, in.values, in.capacity, ib, cb);
			EXPECT_EQ(tiled.totalValue, ref) << "knapsackdptiled " << tag << blocks;
			expect_valid(tiled, in.weights, in.values, in.capacity, "knapsackdptiled " + tag + blocks);

			const auto dag = knapsackdpdag(in.weights, in.values, in.capacity, ib, cb);
			EXPECT_EQ(dag.totalValue, ref) << "knapsackdpdag " << tag << blocks;
			expect_valid(dag, in.weights, in.values, in.capacity, "knapsackdpdag " + tag + blocks);
		}
	}
}

TEST_P(ThreadCount, TopoSchedulerMatchesReferenceUnderContention)
{
	// Many small tiles and wide frontiers: lots of concurrent pushes, the
	// situation in which the old single "filled" counter published unwritten slots.
	std::mt19937 rng(99u + static_cast<unsigned>(GetParam()));
	for (int trial = 0; trial < 20; ++trial)
	{
		const auto in = random_instance(rng, 200, 3000, 5);
		if (in.weights.empty())
			continue;
		const int ref = reference_value(in.weights, in.values, in.capacity);
		auto g = build_graph(in.weights, in.capacity, 2, 8);
		const int topo = solve_dag_topo(g, in.weights, in.values, in.capacity, 2, 8);
		EXPECT_EQ(topo, ref) << "trial=" << trial << " threads=" << GetParam();
	}
}

INSTANTIATE_TEST_SUITE_P(Threads, ThreadCount, ::testing::Values(1, 2, 3, 7));

TEST(KnapsackDPDAG, GraphHasNoParallelEdges)
{
	const std::vector<int> weights{3, 1, 4, 1, 5, 9, 2, 6, 5, 3};
	const std::vector<int> values{2, 7, 1, 8, 2, 8, 1, 8, 2, 8};
	for (int ib : {1, 3, 10})
	{
		for (int cb : {1, 2, 5})
		{
			const auto g = build_graph(weights, 30, ib, cb);
			std::set<std::pair<std::size_t, std::size_t>> unique;
			for (auto e : boost::make_iterator_range(boost::edges(g)))
				EXPECT_TRUE(unique.emplace(boost::source(e, g), boost::target(e, g)).second)
					<< "parallel edge, item_block=" << ib << " cap_block=" << cb;
		}
	}
}
