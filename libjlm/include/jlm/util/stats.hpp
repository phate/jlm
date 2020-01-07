/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_STATS_HPP
#define JLM_UTIL_STATS_HPP

#include <jive/rvsdg/graph.h>
#include <jive/rvsdg/region.h>

#include <jlm/util/file.hpp>

#include <chrono>

namespace jlm {

class rvsdg_destruction_stat{
public:
	rvsdg_destruction_stat(
		size_t nnodes,
		size_t ntacs,
		size_t time,
		const jlm::filepath & filename)
	: time_(time)
	, ntacs_(ntacs)
	, nnodes_(nnodes)
	, filename_(filename)
	{}

	size_t
	time() const noexcept
	{
		return time_;
	}

	size_t
	ntacs() const noexcept
	{
		return ntacs_;
	}

	size_t
	nnodes() const noexcept
	{
		return nnodes_;
	}

	const jlm::filepath &
	filename() const noexcept
	{
		return filename_;
	}

private:
	size_t time_;
	size_t ntacs_;
	size_t nnodes_;
	jlm::filepath filename_;
};

class stat {
public:
	virtual
	~stat();

	virtual std::string
	to_str() const = 0;
};

class stats_descriptor final {
public:
	stats_descriptor()
	: stats_descriptor(std::string("/tmp/jlm-stats.log"))
	{}

	stats_descriptor(const jlm::filepath & path)
	: print_cfr_time(false)
	, print_annotation_time(false)
	, print_aggregation_time(false)
	, print_rvsdg_construction(false)
	, print_rvsdg_destruction(false)
	, print_rvsdg_optimization(false)
	, file_(path)
	{
		file_.open("a");
	}

	const jlm::file &
	file() const noexcept
	{
		return file_;
	}

	void
	set_file(const jlm::filepath & path) noexcept
	{
		file_ = std::move(jlm::file(path));
		file_.open("a");
	}

	void
	print_stat(const rvsdg_destruction_stat & stat) const noexcept
	{
		fprintf(file().fd(), "RVSDGDESTRUCTION %s %zu %zu %zu\n",
			stat.filename().to_str().c_str(), stat.nnodes(), stat.ntacs(), stat.time());
	}

	void
	print_stat(const stat & s) const noexcept
	{
		fprintf(file().fd(), "%s\n", s.to_str().c_str());
	}

	bool print_cfr_time;
	bool print_annotation_time;
	bool print_aggregation_time;
	bool print_rvsdg_construction;
	bool print_rvsdg_destruction;
	bool print_rvsdg_optimization;

private:
	jlm::file file_;
};

}

class statscollector final {
public:
	inline
	statscollector()
	: time_(0)
	, nnodes_after_(0)
	, ninputs_after_(0)
	, nnodes_before_(0)
	, ninputs_before_(0)
	{}

	statscollector(const statscollector &) = delete;

	statscollector(statscollector &&) = delete;

	statscollector &
	operator=(const statscollector &) = delete;

	statscollector &
	operator=(statscollector &&) = delete;

	inline void
	run(const std::function<void(jive::graph&)> & f, jive::graph & rvsdg)
	{
		nnodes_before_ = jive::nnodes(rvsdg.root());
		ninputs_before_ = jive::ninputs(rvsdg.root());

		auto start = std::chrono::high_resolution_clock::now();
		f(rvsdg);
		auto end = std::chrono::high_resolution_clock::now();

		time_ = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
		nnodes_after_ = jive::nnodes(rvsdg.root());
		ninputs_after_ = jive::ninputs(rvsdg.root());
	}

	inline size_t
	time() const noexcept
	{
		return time_;
	}

	inline size_t
	nnodes_before() const noexcept
	{
		return nnodes_before_;
	}

	inline size_t
	nnodes_after() const noexcept
	{
		return nnodes_after_;
	}

	inline size_t
	ninputs_before() const noexcept
	{
		return ninputs_before_;
	}

	inline size_t
	ninputs_after() const noexcept
	{
		return ninputs_after_;
	}

private:
	size_t time_;
	size_t nnodes_after_;
	size_t ninputs_after_;
	size_t nnodes_before_;
	size_t ninputs_before_;
};

#endif
