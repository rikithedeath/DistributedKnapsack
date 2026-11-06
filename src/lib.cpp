#include "Knapsack/knapsack.hpp"
#include "Knapsack/knapsackdpdag_impl.hpp"
#include "build_info.hpp"
#include "phases.hpp"
#include "time.hpp"
#include <iostream>
#include <omp.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>

#ifdef KP_GPU
#include "knapsackdp.hpp"
constexpr bool use_gpu = true;
#else
constexpr bool use_gpu = false;
#endif

#ifdef KP_MPI
#include "Knapsackmpi/knapsackmpi.hpp"
#include <boost/mpi.hpp>
#endif

#ifdef KP_MPI
constexpr bool use_mpi = true;
#else
constexpr bool use_mpi = false;
#endif
namespace py = pybind11;

void hello()
{
	std::cout << "Hello, World!" << std::endl;
}

struct KnapsackArguments
{
	std::vector<int> weights;
	std::vector<int> values;
	int capacity;

	KnapsackArguments(const py::list &weights, const py::list &values, int capacity)
	{
		int n = py::len(weights);
		this->weights = std::vector<int>(n);
		this->values = std::vector<int>(n);
		this->capacity = capacity;
		for (int i = 0; i < n; ++i)
		{
			this->weights[i] = py::cast<int>(weights[i]);
			this->values[i] = py::cast<int>(values[i]);
		}
	}
};

KnapsackSolution knapsackdpsolver(KnapsackArguments args)
{
	return knapsackdp(args.weights, args.values, args.capacity);
}

KnapsackSolution knapsackcopasolver(KnapsackArguments args)
{
	auto res = knapsackcopa(args.weights, args.values, args.capacity);
	if (res.has_value())
		return res.value();
	return KnapsackSolution();
}

KnapsackSolution knapsackcopasequentialsolver(KnapsackArguments args)
{
	auto res = knapsackcopasequential(args.weights, args.values, args.capacity);
	if (res.has_value())
		return res.value();
	return KnapsackSolution();
}

KnapsackSolution knapsackdpgpusolver(KnapsackArguments args)
{
#ifdef KP_GPU
	const auto res = kp::gpu::knapsackdp(args.weights, args.values, args.capacity);
	return KnapsackSolution{res.items, res.totalValue, res.totalWeight};

#else

	spdlog::error("Requested to use gpu solver but no solver compiled");
	return {};
#endif
}

KnapsackSolution knapsackdpmpisolver(KnapsackArguments args)
{
#ifdef KP_MPI
	static boost::mpi::environment env(boost::mpi::threading::single);
	static boost::mpi::communicator comm;
	auto res = kp::mpi::knapsackdpmpi(comm, args.weights, args.values, args.capacity);
	if (res.has_value())
		return res.value();
	return KnapsackSolution();
#else

	spdlog::error("Requested to use mpi solver but no solver compiled");
	return {};
#endif
}

void disable_logging()
{
	spdlog::set_level(spdlog::level::err);
}

PYBIND11_MODULE(libdistributed_knapsack, m)
{
	m.doc() = "Distributed Knapsack solvers";
	m.def("hello", &hello);
	py::class_<KnapsackSolution>(m, "KnapsackSolution")
		.def(py::init<>())
		.def_readwrite("items", &KnapsackSolution::items)
		.def_readwrite("totalValue", &KnapsackSolution::totalValue)
		.def_readwrite("totalWeight", &KnapsackSolution::totalWeight);
	py::class_<KnapsackArguments>(m, "KnapsackArguments")
		.def(py::init<py::list, py::list, int>())
		.def_readwrite("weights", &KnapsackArguments::weights)
		.def_readwrite("values", &KnapsackArguments::values)
		.def_readwrite("capacity", &KnapsackArguments::capacity);
	m.def(
		"knapsackdp",
		[](KnapsackArguments args, int cap_block) { return knapsackdp(args.weights, args.values, args.capacity); },
		py::arg("args"), py::arg("cap_block") = 1);
	m.def(
		"knapsackdpbits", [](KnapsackArguments args) { return knapsackdpbits(args.weights, args.values, args.capacity); },
		py::arg("args"));
	m.def(
		"knapsackdptiled",
		[](KnapsackArguments args, int item_block, int cap_block) {
			return knapsackdptiled(args.weights, args.values, args.capacity, item_block, cap_block);
		},
		py::arg("args"), py::arg("item_block") = 10, py::arg("cap_block") = 0);
	m.def("knapsackcopa", &knapsackcopasolver, py::arg("args"));
	m.def("knapsackcopasequential", &knapsackcopasequentialsolver, py::arg("args"));
	m.def("knapsackdpgpu", &knapsackdpgpusolver, py::arg("args"));
	m.def("knapsackdpmpi", &knapsackdpmpisolver, py::arg("args"));
	m.def(
		"knapsackdpdag",
		[](KnapsackArguments args, int item_block, int cap_block) {
			return knapsackdpdag(args.weights, args.values, args.capacity, item_block, cap_block);
		},
		py::arg("args"), py::arg("item_block") = 10, py::arg("cap_block") = 0);
	m.def("disable_logging", &disable_logging);
	py::class_<TimeSection>(m, "TimeSection")
		.def_readonly("min", &TimeSection::min)
		.def_readonly("max", &TimeSection::max)
		.def_readonly("mean", &TimeSection::mean)
		.def_readonly("variance", &TimeSection::variance)
		.def_readonly("count", &TimeSection::count);
	m.def(
		"get_section", [](const std::string &name) { return TimeSectionRegister::instance().get_section(name); },
		py::arg("name"));
	m.def("get_all_sections", []() { return TimeSectionRegister::instance().get_all_sections(); });
	py::class_<DAGStats>(m, "DAGStats")
		.def(py::init<>())
		.def_readonly("tiles", &DAGStats::tiles)
		.def_readonly("edges", &DAGStats::edges)
		.def_readonly("levels", &DAGStats::levels)
		.def_property_readonly("frontierMean", [](const DAGStats &s) { return ba::mean(s.frontier); })
		.def_property_readonly("frontierMedian", [](const DAGStats &s) { return ba::median(s.frontier); })
		.def_property_readonly("frontierMin", [](const DAGStats &s) { return ba::min(s.frontier); })
		.def_property_readonly("frontierMax", [](const DAGStats &s) { return ba::max(s.frontier); });
	m.def("get_dag_stats", &get_dag_stats);
	// Threads OpenMP will actually use: the Python driver checks it against the
	// request, because OMP_NUM_THREADS is read once, when libgomp is loaded.
	m.def("omp_max_threads", []() { return omp_get_max_threads(); });
	m.def("set_num_threads", [](int n) { omp_set_num_threads(n); }, py::arg("n"));
	m.def("get_phases", []() { return PhaseTimes::instance().get(); });
	m.def("build_info", []() {
		py::dict d;
		d["type"] = KP_BUILD_TYPE;
		d["compiler"] = KP_CXX_COMPILER;
		d["flags"] = KP_CXX_FLAGS;
		d["commit"] = KP_GIT_COMMIT;
		d["timers"] = static_cast<bool>(KP_TIMERS_ENABLED);
		return d;
	});
}
