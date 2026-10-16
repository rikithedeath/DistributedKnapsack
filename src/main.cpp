#include "Knapsack/knapsack.hpp"
#include "Knapsack/knapsackdpdag_impl.hpp"
#include "time.hpp"
#include <argparse/argparse.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include <omp.h>

#ifdef KP_GPU
#include "knapsackdp.hpp"
#endif

namespace ba = boost::accumulators;

namespace
{

/// Simple JSON string escaping (hostnames and file paths never need more).
std::string json_escape(const std::string &s)
{
	std::string out;
	out.reserve(s.size());
	for (char c : s)
	{
		switch (c)
		{
		case '\\':
			out += "\\\\";
			break;
		case '"':
			out += "\\\"";
			break;
		case '\n':
			out += "\\n";
			break;
		case '\r':
			out += "\\r";
			break;
		case '\t':
			out += "\\t";
			break;
		default:
			if (static_cast<unsigned char>(c) < 0x20)
				out += fmt::format("\\u{:04x}", static_cast<unsigned>(static_cast<unsigned char>(c)));
			else
				out += c;
		}
	}
	return out;
}

int env_int(const char *name, int fallback)
{
	const char *value = std::getenv(name);
	if (value == nullptr)
		return fallback;
	try
	{
		return std::stoi(value);
	}
	catch (const std::exception &)
	{
		return fallback;
	}
}

std::string trim(const std::string &s)
{
	const auto begin = s.find_first_not_of(" \t\r\n");
	if (begin == std::string::npos)
		return "";
	const auto end = s.find_last_not_of(" \t\r\n");
	return s.substr(begin, end - begin + 1);
}

/// Append a pre-serialized JSON entry to the JSON array stored in `path`,
/// mirroring testengine.py::_append_result (existing arrays are re-read,
/// appended to and rewritten; malformed content resets the file).
void append_json_entry(const std::string &path, const std::string &entry)
{
	std::string body = entry;
	std::ifstream in(path);
	if (in.is_open())
	{
		std::stringstream ss;
		ss << in.rdbuf();
		in.close();
		std::string existing = trim(ss.str());
		if (existing.size() >= 2 && existing.front() == '[' && existing.back() == ']')
		{
			std::string inner = trim(existing.substr(1, existing.size() - 2));
			if (!inner.empty())
				body = inner + ",\n  " + entry;
		}
	}

	std::ofstream out(path, std::ios::trunc);
	if (!out.is_open())
	{
		spdlog::error("Could not open '{}' for writing", path);
		return;
	}
	out << "[\n  " << body << "\n]\n";
}

} // namespace

int main(int argc, char **argv)
{
	argparse::ArgumentParser program("kpbench");
	program.add_description("Standalone knapsack benchmark (pure C++, perf-friendly).");
	program.add_argument("--test")
		.default_value(std::string("all"))
		.help("Test to run: knapsackdp, knapsackdpdag, knapsackdp_gpu");
	program.add_argument("--seed").scan<'i', int>().default_value(42).help("Random seed");
	program.add_argument("--numItems").scan<'i', int>().default_value(30).help("Number of items");
	program.add_argument("--numThreads").scan<'i', int>().default_value(1).help("Number of threads");
	program.add_argument("--capacity").scan<'i', int>().default_value(1000).help("Knapsack capacity");
	program.add_argument("--minWeight").scan<'i', int>().default_value(1).help("Minimum weight");
	program.add_argument("--maxWeight").scan<'i', int>().default_value(50).help("Maximum weight");
	program.add_argument("--minValue").scan<'i', int>().default_value(1).help("Minimum value");
	program.add_argument("--maxValue").scan<'i', int>().default_value(50).help("Maximum value");
	program.add_argument("--save")
		.default_value(std::string("results.json"))
		.help("Output JSON file for results (empty string disables saving)");
	program.add_argument("--itemBlock").scan<'i', int>().default_value(10)
		.help("Item block size for knapsackdpdag (default: 10)");
	program.add_argument("--capBlock").scan<'i', int>().default_value(0)
		.help("Capacity block size for knapsackdpdag (default: 0)");
	program.add_argument("--dump-instance").flag().help("Print the generated instance and exit");

	try
	{
		program.parse_args(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cerr << "Error parsing command line: " << e.what() << "\n";
		std::cerr << program << "\n";
		return 1;
	}

	const std::string test = program.get<std::string>("--test");
	const std::array<std::string, 3> allowed{"knapsackdp", "knapsackdpdag", "knapsackdp_gpu"};
	if (std::find(allowed.begin(), allowed.end(), test) == allowed.end())
	{
		fmt::print(stderr, "Error: unknown test '{}'. Allowed values: knapsackdp, knapsackdpdag, knapsackdp_gpu\n",
				   test);
		return 1;
	}

	const int numThreads = program.get<int>("--numThreads");
	const int numItems = program.get<int>("--numItems");
	const int capacity = program.get<int>("--capacity");
	const int minWeight = program.get<int>("--minWeight");
	const int maxWeight = program.get<int>("--maxWeight");
	const int minValue = program.get<int>("--minValue");
	const int maxValue = program.get<int>("--maxValue");
	const int itemBlock = program.get<int>("--itemBlock");
	const int capBlock = program.get<int>("--capBlock");
	const int seed = program.get<int>("--seed");
	const std::string saveFile = program.get<std::string>("--save");
	const bool dump_instance = program.get<bool>("--dump-instance") != 0;

	// Control the OpenMP thread count before any OpenMP usage happens.
	setenv("OMP_NUM_THREADS", std::to_string(numThreads).c_str(), 1);
	omp_set_num_threads(numThreads);

	// Generate the instance: weights first, then values (same order as the
	// Python driver, but with std::mt19937 instead of Python's random module).
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
	std::vector<int> weights(static_cast<std::size_t>(numItems));
	std::vector<int> values(static_cast<std::size_t>(numItems));
	{
		std::uniform_int_distribution<int> weightDist(minWeight, maxWeight);
		for (auto &w : weights)
			w = weightDist(rng);
		std::uniform_int_distribution<int> valueDist(minValue, maxValue);
		for (auto &v : values)
			v = valueDist(rng);
	}

	if (dump_instance)
	{
		fmt::print("weights = {}\n", fmt::join(weights, ", "));
		fmt::print("values = {}\n", fmt::join(values, ", "));
		fmt::print("capacity = {}\n", capacity);
		return 0;
	}

	const auto start = std::chrono::steady_clock::now();
	KnapsackSolution solution;
	if (test == "knapsackdp")
	{
		solution = knapsackdp(weights, values, capacity);
	}
	else if (test == "knapsackdpdag")
	{
		solution = knapsackdpdag(weights, values, capacity, itemBlock, capBlock);
	}
	else // knapsackdp_gpu
	{
#ifdef KP_GPU
		const auto res = kp::gpu::knapsackdp(weights, values, capacity);
		solution = KnapsackSolution{res.items, res.totalValue, res.totalWeight};
#else
		spdlog::error("Requested to use gpu solver but no solver compiled");
		return 1;
#endif
	}
	const auto end = std::chrono::steady_clock::now();
	const double duration = std::chrono::duration<double>(end - start).count();

	fmt::print("{}: {:.4f}s | Items: {} | Profit: {} | Weight: {}\n", test, duration, numItems, solution.totalValue,
			   solution.totalWeight);

	const auto sections = TimeSectionRegister::instance().get_all_sections();
	if (!sections.empty())
	{
		fmt::print("  Time sections:\n");
		std::vector<std::pair<const std::string *, const TimeSection *>> ordered;
		ordered.reserve(sections.size());
		for (const auto &[name, section] : sections)
			ordered.emplace_back(&name, &section);
		std::sort(ordered.begin(), ordered.end(),
				  [](const auto &a, const auto &b) { return *a.first < *b.first; });
		for (const auto &[name, section] : ordered)
			fmt::print("    {}: count={} mean={:.4f}ms min={:.4f}ms max={:.4f}ms\n", *name, section->count,
					   section->mean, section->min, section->max);
	}

	std::optional<DAGStats> dagStats;
	if (test == "knapsackdpdag")
	{
		dagStats = get_dag_stats();
		fmt::print("    DAG stats: tiles={} edges={} levels={}\n", dagStats->tiles, dagStats->edges, dagStats->levels);
		if (dagStats->levels > 0)
			fmt::print("    wavefront widths: min={:.0f} median={:.1f} mean={:.2f} max={:.0f}\n",
					   ba::min(dagStats->frontier), ba::median(dagStats->frontier), ba::mean(dagStats->frontier),
					   ba::max(dagStats->frontier));
	}

	// Save a schema-compatible entry to the JSON results file.
	char hostname[256] = {0};
	gethostname(hostname, sizeof(hostname) - 1);

	const int processors =
		test == "knapsackdp_gpu" ? env_int("KP_GPU_THREADS", numThreads) : env_int("OMP_NUM_THREADS", numThreads);

	std::string timeSections;
	if (sections.empty())
	{
		timeSections = "{}";
	}
	else
	{
		timeSections = "{";
		bool first = true;
		for (const auto &[name, section] : sections)
		{
			if (!first)
				timeSections += ", ";
			first = false;
			fmt::format_to(std::back_inserter(timeSections),
						   "\"{}\": {{\"count\": {}, \"mean\": {}, \"min\": {}, \"max\": {}, \"variance\": {}}}", name,
						   section.count, section.mean, section.min, section.max, section.variance);
		}
		timeSections += "}";
	}

	std::string entry = fmt::format(
		"{{\"hostname\": \"{}\", \"testname\": \"{}\", \"testtype\": \"shared memory\", \"time\": {}, "
		"\"processors\": {}, \"solution_weight\": {}, \"solution_profit\": {}, \"capacity\": {}, \"num_items\": {}, "
		"\"min_weight\": {}, \"max_weight\": {}, \"seed\": {}, \"time_sections\": {}",
		json_escape(hostname), test, duration, processors, solution.totalWeight, solution.totalValue, capacity, numItems,
		minWeight, maxWeight, seed, timeSections);

	if (test == "knapsackdpdag")
		fmt::format_to(std::back_inserter(entry), ", \"item_block\": {}, \"cap_block\": {}", itemBlock, capBlock);

	if (test == "knapsackdpdag" && dagStats && dagStats->levels > 0)
		fmt::format_to(
			std::back_inserter(entry),
			", \"frontier_distribution\": {{\"min\": {}, \"median\": {}, \"mean\": {}, \"max\": {}}}, "
			"\"dag\": {{\"tiles\": {}, \"edges\": {}, \"levels\": {}}}",
			ba::min(dagStats->frontier), ba::median(dagStats->frontier), ba::mean(dagStats->frontier),
			ba::max(dagStats->frontier), dagStats->tiles, dagStats->edges, dagStats->levels);

	entry += "}";

	if (!saveFile.empty())
	{
		append_json_entry(saveFile, entry);
		fmt::print("Result saved to {}\n", saveFile);
	}

	return 0;
}
