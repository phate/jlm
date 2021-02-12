/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_UTIL_STATS_HPP
#define JLM_UTIL_STATS_HPP

#include <jlm/util/file.hpp>

namespace jlm {

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
	, print_cne_stat(false)
	, print_dne_stat(false)
	, print_iln_stat(false)
	, print_inv_stat(false)
	, print_ivt_stat(false)
	, print_pull_stat(false)
	, print_push_stat(false)
	, print_reduction_stat(false)
	, print_unroll_stat(false)
	, print_annotation_time(false)
	, print_aggregation_time(false)
	, print_rvsdg_construction(false)
	, print_rvsdg_destruction(false)
	, print_rvsdg_optimization(false)
	, print_jlm_rvsdg_conversion(false)
	, file_(path)
	{
		file_.open("a");
	}


	const jlm::filepath &
	filepath() const noexcept
	{
		return file_.path();
	}

	void
	set_file(const jlm::filepath & path) noexcept
	{
		file_ = jlm::file(path);
		file_.open("a");
	}

	void
	print_stat(const stat & s) const noexcept
	{
		fprintf(file_.fd(), "%s\n", s.to_str().c_str());
	}

	bool print_cfr_time;
	bool print_cne_stat;
	bool print_dne_stat;
	bool print_iln_stat;
	bool print_inv_stat;
	bool print_ivt_stat;
	bool print_pull_stat;
	bool print_push_stat;
	bool print_reduction_stat;
	bool print_unroll_stat;
	bool print_annotation_time;
	bool print_aggregation_time;
	bool print_rvsdg_construction;
	bool print_rvsdg_destruction;
	bool print_rvsdg_optimization;
	bool print_jlm_rvsdg_conversion;

private:
	jlm::file file_;
};

}

#endif
