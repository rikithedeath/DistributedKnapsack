#pragma once
#include <boost/graph/adjacency_list.hpp>
#include <optional>
#include <vector>
struct KnapsackSolution
{
	std::vector<int> items; // Indices of items included in the knapsack
	int totalValue;			// Total value of the included items
	int totalWeight;		// Total weight of the included items
};

/** Classic DP: full row-major table, rows computed one after the other, each row split among the threads. */
KnapsackSolution knapsackdp(const std::vector<int> &weights, const std::vector<int> &values, int capacity);

/** Classic DP with two rolling rows and a 1-bit-per-cell decision table (32x less memory). */
KnapsackSolution knapsackdpbits(const std::vector<int> &weights, const std::vector<int> &values, int capacity);

/** Tiled DP on the same full table, scheduled as an explicit anti-diagonal wavefront (no graph). */
KnapsackSolution knapsackdptiled(const std::vector<int> &weights, const std::vector<int> &values, int capacity,
								 int item_block = 10, int cap_block = 0);

std::optional<KnapsackSolution> knapsackcopasequential(const std::vector<int> &weights, const std::vector<int> &values,
													   int capacity);

std::optional<KnapsackSolution> knapsackcopa(const std::vector<int> &weights, const std::vector<int> &values,
											 int capacity);

/** Tiled DP whose tile dependencies are an explicit Boost.Graph DAG, executed by levels. */
KnapsackSolution knapsackdpdag(const std::vector<int> &weights, const std::vector<int> &values, int capacity,
							   int item_block = 10, int cap_block = 0);
