#pragma once
#include <chrono>
#include <string>
#include <utility>
#include <vector>

/**
 * @file phases.hpp
 * @brief Always-on wall-clock timing of the coarse phases of a solver.
 *
 * Unlike the fine-grained TimeSectionRegister (compiled in only with
 * ENABLE_TIMERS), these timers are cheap enough to stay enabled in benchmark
 * builds: each solver records a handful of phases (allocation, graph build,
 * compute, backtracking) once per call. They let an experiment separate the
 * time of the DP kernel from fixed costs such as allocating and zeroing a
 * multi-GB table or building the tile graph.
 *
 * Only the thread that called the solver records phases; never call them from
 * inside a parallel region.
 */
class PhaseTimes
{
  public:
	static PhaseTimes &instance()
	{
		static PhaseTimes reg;
		return reg;
	}

	void clear()
	{
		phases_.clear();
	}

	void add(const std::string &name, double seconds)
	{
		phases_.emplace_back(name, seconds);
	}

	const std::vector<std::pair<std::string, double>> &get() const
	{
		return phases_;
	}

  private:
	std::vector<std::pair<std::string, double>> phases_;
};

/** @brief RAII helper: records the lifetime of the object as phase `name`. */
class ScopedPhase
{
  public:
	explicit ScopedPhase(std::string name) : name_(std::move(name)), start_(std::chrono::steady_clock::now())
	{
	}

	~ScopedPhase()
	{
		const auto end = std::chrono::steady_clock::now();
		PhaseTimes::instance().add(name_, std::chrono::duration<double>(end - start_).count());
	}

	ScopedPhase(const ScopedPhase &) = delete;
	ScopedPhase &operator=(const ScopedPhase &) = delete;

  private:
	std::string name_;
	std::chrono::steady_clock::time_point start_;
};
