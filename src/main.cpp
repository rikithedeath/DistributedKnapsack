#include "Knapsack/knapsack.hpp"
#include "Knapsack/knapsackdpdag_impl.hpp"
#include "build_info.hpp"
#include "phases.hpp"
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
#include <memory>
#include <optional>
#include <numeric>
#include <random>
#include <set>
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

std::string env_str(const char *name)
{
	const char *value = std::getenv(name);
	return value == nullptr ? std::string() : std::string(value);
}

/// Optimal value computed by the textbook in-place 1-D DP (serial, value only).
/// It shares no code with the solvers under test, so it is an independent
/// reference for --verify.
int reference_value(const std::vector<int> &weights, const std::vector<int> &values, int capacity)
{
	std::vector<int> row(static_cast<std::size_t>(capacity) + 1, 0);
	for (std::size_t i = 0; i < weights.size(); ++i)
		for (int w = capacity; w >= weights[i]; --w)
			row[w] = std::max(row[w], row[w - weights[i]] + values[i]);
	return row[capacity];
}

/// A returned solution is valid if its items are distinct, in range, fit the
/// capacity and add up to the reported weight and value.
bool solution_is_valid(const KnapsackSolution &s, const std::vector<int> &weights, const std::vector<int> &values,
					   int capacity)
{
	std::set<int> seen;
	long long w = 0, v = 0;
	for (int idx : s.items)
	{
		if (idx < 0 || idx >= static_cast<int>(weights.size()) || !seen.insert(idx).second)
			return false;
		w += weights[idx];
		v += values[idx];
	}
	return w <= capacity && w == s.totalWeight && v == s.totalValue;
}

/// STREAM-like triad a[i] = b[i] + s * c[i] on three arrays of n doubles,
/// first-touched in parallel with the same static schedule. Returns the best
/// time of `reps` repetitions; bytes moved per repetition = 3 * 8 * n
/// (STREAM convention, write-allocate traffic not counted).
double triad_best_time(std::int64_t n, int reps)
{
	std::unique_ptr<double[]> a(new double[static_cast<std::size_t>(n)]);
	std::unique_ptr<double[]> b(new double[static_cast<std::size_t>(n)]);
	std::unique_ptr<double[]> c(new double[static_cast<std::size_t>(n)]);
#pragma omp parallel for schedule(static)
	for (std::int64_t i = 0; i < n; ++i)
	{
		a[i] = 0.0;
		b[i] = 1.0;
		c[i] = 2.0;
	}
	double best = 1e300;
	for (int r = 0; r < reps; ++r)
	{
		const auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
		for (std::int64_t i = 0; i < n; ++i)
			a[i] = b[i] + 3.0 * c[i];
		const auto t1 = std::chrono::steady_clock::now();
		best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
	}
	// Keep the result observable so the loops cannot be optimised away.
	if (a[n / 2] != 7.0)
		spdlog::warn("triad check failed");
	return best;
}

} // namespace

int main(int argc, char **argv)
{
	argparse::ArgumentParser program("kpbench");
	program.add_description("Standalone knapsack benchmark (pure C++, perf-friendly).");
	program.add_argument("--test")
		.default_value(std::string("all"))
		.help("Test to run: knapsackdp, knapsackdpbits, knapsackdptiled, knapsackdpdag, knapsackdp_gpu, bandwidth "
			  "(bandwidth: STREAM-like triad on 3 arrays of --capacity doubles)");
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
	program.add_argument("--verify")
		.flag()
		.help("Also compute the optimum with an independent serial 1-D DP (not timed) and fail on mismatch");

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
	const std::array<std::string, 6> allowed{"knapsackdp",	  "knapsackdpbits", "knapsackdptiled",
											 "knapsackdpdag", "knapsackdp_gpu", "bandwidth"};
	if (std::find(allowed.begin(), allowed.end(), test) == allowed.end())
	{
		fmt::print(stderr, "Error: unknown test '{}'. Allowed values: {}\n", test, fmt::join(allowed, ", "));
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
	const bool verify = program.get<bool>("--verify") != 0;

	// Control the OpenMP thread count before any OpenMP usage happens.
	setenv("OMP_NUM_THREADS", std::to_string(numThreads).c_str(), 1);
	omp_set_num_threads(numThreads);
	// What OpenMP will really use (it can differ from the request, e.g. with
	// OMP_DYNAMIC or a thread limit): this, not the request, is recorded.
	const int threadsEffective = omp_get_max_threads();
	if (threadsEffective != numThreads)
		spdlog::warn("requested {} threads, OpenMP will use {}", numThreads, threadsEffective);

	char hostname[256] = {0};
	gethostname(hostname, sizeof(hostname) - 1);
	const std::string buildJson = fmt::format(
		"\"build\": {{\"type\": \"{}\", \"compiler\": \"{}\", \"flags\": \"{}\", \"commit\": \"{}\", \"timers\": {}}}, "
		"\"omp\": {{\"threads_effective\": {}, \"proc_bind\": \"{}\", \"places\": \"{}\"}}",
		json_escape(KP_BUILD_TYPE), json_escape(KP_CXX_COMPILER), json_escape(KP_CXX_FLAGS), json_escape(KP_GIT_COMMIT),
		KP_TIMERS_ENABLED ? "true" : "false", threadsEffective, json_escape(env_str("OMP_PROC_BIND")),
		json_escape(env_str("OMP_PLACES")));
	fmt::print("build: {} | {} | flags \"{}\" | commit {} | timers {} | threads {} | proc_bind '{}' | places '{}'\n",
			   KP_BUILD_TYPE, KP_CXX_COMPILER, KP_CXX_FLAGS, KP_GIT_COMMIT, KP_TIMERS_ENABLED ? "on" : "off",
			   threadsEffective, env_str("OMP_PROC_BIND"), env_str("OMP_PLACES"));

	if (test == "bandwidth")
	{
		const std::int64_t n = std::max(1, capacity);
		constexpr int reps = 10;
		const double t = triad_best_time(n, reps);
		const double gbs = 3.0 * 8.0 * static_cast<double>(n) / t / 1e9;
		fmt::print("bandwidth: {:.6f}s | Elements: {} | GB/s: {:.2f}\n", t, n, gbs);
		if (!saveFile.empty())
		{
			append_json_entry(saveFile,
							  fmt::format("{{\"hostname\": \"{}\", \"testname\": \"bandwidth\", \"time\": {}, "
										  "\"processors\": {}, \"elements\": {}, \"bandwidth_gbs\": {}, {}}}",
										  json_escape(hostname), t, threadsEffective, n, gbs, buildJson));
			fmt::print("Result saved to {}\n", saveFile);
		}
		return 0;
	}

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
	else if (test == "knapsackdpbits")
	{
		solution = knapsackdpbits(weights, values, capacity);
	}
	else if (test == "knapsackdptiled")
	{
		solution = knapsackdptiled(weights, values, capacity, itemBlock, capBlock);
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

	// Coarse phases recorded by the solver (always on, see phases.hpp).
	const auto phases = PhaseTimes::instance().get();
	std::string phasesJson = "{";
	for (std::size_t k = 0; k < phases.size(); ++k)
		phasesJson += fmt::format("{}\"{}\": {}", k == 0 ? "" : ", ", phases[k].first, phases[k].second);
	phasesJson += "}";
	if (!phases.empty())
	{
		fmt::print("  phases:");
		for (const auto &[name, sec] : phases)
			fmt::print(" {}={:.4f}s", name, sec);
		fmt::print("\n");
	}

	const bool valid = solution_is_valid(solution, weights, values, capacity);
	std::string verifiedJson = "null";
	bool verifiedOk = true;
	if (verify)
	{
		const int ref = reference_value(weights, values, capacity);
		verifiedOk = ref == solution.totalValue;
		verifiedJson = verifiedOk ? "true" : "false";
		fmt::print("  verify: reference={} solver={} -> {}\n", ref, solution.totalValue, verifiedOk ? "OK" : "MISMATCH");
	}
	if (!valid)
		fmt::print(stderr, "ERROR: the returned item set is not a valid solution\n");

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
	const int processors = test == "knapsackdp_gpu" ? env_int("KP_GPU_THREADS", numThreads) : threadsEffective;

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
		"\"min_weight\": {}, \"max_weight\": {}, \"min_value\": {}, \"max_value\": {}, \"sum_weights\": {}, "
		"\"seed\": {}, "
		"\"time_sections\": {}, \"phases\": {}, \"valid\": {}, \"verified\": {}, {}",
		json_escape(hostname), test, duration, processors, solution.totalWeight, solution.totalValue, capacity, numItems,
		minWeight, maxWeight, minValue, maxValue, std::accumulate(weights.begin(), weights.end(), 0LL), seed, timeSections, phasesJson, valid ? "true" : "false", verifiedJson,
		buildJson);

	if (test == "knapsackdpdag" || test == "knapsackdptiled")
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

	return (valid && verifiedOk) ? 0 : 2;
}
