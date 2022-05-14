/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/opt/inlining.hpp>
#include <jlm/opt/optimization.hpp>
#include <jlm/opt/pull.hpp>

#include <jlm/util/Statistics.hpp>
#include <jlm/util/strfmt.hpp>
#include <jlm/util/time.hpp>

namespace jlm {

/* optimization class */

optimization::~optimization()
{}

/* optimization_stat class */

class optimization_stat final : public Statistics {
public:
	virtual
	~optimization_stat()
	{}

	optimization_stat(const jlm::filepath & filename)
	: Statistics(StatisticsDescriptor::StatisticsId::RvsdgOptimization)
  , filename_(filename)
	, nnodes_before_(0)
	, nnodes_after_(0)
	{}

	void
	start(const jive::graph & graph) noexcept
	{
		nnodes_before_ = jive::nnodes(graph.root());
		timer_.start();
	}

	void
	end(const jive::graph & graph) noexcept
	{
		timer_.stop();
		nnodes_after_ = jive::nnodes(graph.root());
	}

	virtual std::string
	ToString() const override
	{
		return strfmt("RVSDGOPTIMIZATION ", filename_.to_str(), " ",
			nnodes_before_, " ", nnodes_after_, " ", timer_.ns());
	}

private:
	jlm::timer timer_;
	jlm::filepath filename_;
	size_t nnodes_before_, nnodes_after_;
};

void
optimize(
  RvsdgModule & rm,
  const StatisticsDescriptor & sd,
  const std::vector<optimization*> & opts)
{
	optimization_stat stat(rm.SourceFileName());

	stat.start(rm.Rvsdg());
	for (const auto & opt : opts)
		opt->run(rm, sd);
	stat.end(rm.Rvsdg());

  sd.PrintStatistics(stat);
}

}
