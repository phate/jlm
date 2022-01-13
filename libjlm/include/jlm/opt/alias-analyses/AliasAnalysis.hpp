/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_ALIASANALYSIS_HPP
#define JLM_OPT_ALIAS_ANALYSES_ALIASANALYSIS_HPP

#include <memory>

namespace jlm {

class RvsdgModule;
class StatisticsDescriptor;

namespace aa {

class PointsToGraph;

/**
* \brief Alias Analysis Interface
*/
class AliasAnalysis {
public:
	virtual
	~AliasAnalysis() = default;

	/**
	  * \brief Analyze RVSDG module
	  *
	  * \param module RVSDG module the analysis is performed on.
	  * \param sd Statistics descriptor for printing analysis statistics.
	  *
	  * \return A PointsTo graph.
	  */
	virtual std::unique_ptr<PointsToGraph>
	Analyze(
    const RvsdgModule & module,
    const StatisticsDescriptor & sd) = 0;
};

}}

#endif
