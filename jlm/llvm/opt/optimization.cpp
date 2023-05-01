/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/inlining.hpp>
#include <jlm/llvm/opt/optimization.hpp>

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
	: Statistics(Statistics::Id::RvsdgOptimization)
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

  static std::unique_ptr<optimization_stat>
  Create(const jlm::filepath & sourceFile)
  {
    return std::make_unique<optimization_stat>(sourceFile);
  }

private:
	jlm::timer timer_;
	jlm::filepath filename_;
	size_t nnodes_before_, nnodes_after_;
};

void
optimize(
  RvsdgModule & rm,
  StatisticsCollector & statisticsCollector,
  const std::vector<optimization*> & opts)
{
	auto statistics = optimization_stat::Create(rm.SourceFileName());

	statistics->start(rm.Rvsdg());
	for (const auto & opt : opts)
		opt->run(rm, statisticsCollector);
	statistics->end(rm.Rvsdg());

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));
}

}
