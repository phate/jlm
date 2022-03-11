/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_STEENSGAARD_HPP
#define JLM_OPT_ALIAS_ANALYSES_STEENSGAARD_HPP

#include <jlm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/util/disjointset.hpp>

#include <string>

namespace jive {
	class argument;
	class gamma_node;
	class graph;
	class node;
	class output;
	class region;
	class simple_node;
	class structural_node;
	class theta_node;
}

namespace jlm {

namespace delta { class node; }
namespace lambda { class node; }
namespace phi { class node; }

class LoadNode;
class StoreNode;

namespace aa {

class Location;
class PointsToGraph;

/** \brief LocationSet class
*/
class LocationSet final {

	using DisjointLocationSet = typename jlm::disjointset<Location*>;

public:
	using const_iterator = std::unordered_map<
	  const jive::output*
	, Location*
	>::const_iterator;

	~LocationSet();

	LocationSet();

	LocationSet(const LocationSet &) = delete;

	LocationSet(LocationSet &&) = delete;

	LocationSet &
	operator=(const LocationSet &) = delete;

	LocationSet &
	operator=(LocationSet &&) = delete;

	DisjointLocationSet::set_iterator
	begin() const
	{
		return djset_.begin();
	}

	DisjointLocationSet::set_iterator
	end() const
	{
		return djset_.end();
	}

  Location &
  InsertAllocaLocation(const jive::node & node);

  Location &
  InsertMallocLocation(const jive::node & node);

  Location &
  InsertLambdaLocation(const lambda::node & lambda);

  Location &
  InsertDeltaLocation(const delta::node & delta);

	Location &
	InsertImportLocation(const jive::argument * argument);

	Location &
	InsertDummyLocation();

	bool
	contains(const jive::output * output) const noexcept;

	Location &
	FindOrInsertRegisterLocation(
    const jive::output * output,
    bool unknown,
    bool pointsToExternalMemory);

	const DisjointLocationSet::set &
	set(Location & l) const
	{
		return *djset_.find(&l);
	}

  size_t
  NumDisjointSets() const noexcept
  {
    return djset_.nsets();
  }

  size_t
  NumLocations() const noexcept
  {
    return djset_.nvalues();
  }

	Location &
	GetRootLocation(Location & l) const;

	Location &
	Find(const jive::output * output);

	Location &
	Merge(Location & l1, Location & l2);

	std::string
	to_dot() const;

	void
	clear();

private:
	Location &
	InsertRegisterLocation(
    const jive::output * output,
    bool unknown,
    bool pointsToExternalMemory);

	Location *
	lookup(const jive::output * output);

	DisjointLocationSet djset_;
	std::vector<std::unique_ptr<Location>> locations_;
	std::unordered_map<const jive::output*, Location*> map_;
};

/** \brief Steensgaard alias analysis
 *
 * This class implements a Steensgaard alias analysis. The analysis is inter-procedural, field-insensitive,
 * context-insensitive, flow-insensitive, and uses a static heap model. It is an implementation corresponding to the
 * algorithm presented in Bjarne Steensgaard - Points-to Analysis in Almost Linear Time.
 */
class Steensgaard final : public AliasAnalysis {
public:
	~Steensgaard() override;

	Steensgaard() = default;

	Steensgaard(const Steensgaard &) = delete;

	Steensgaard(Steensgaard &&) = delete;

	Steensgaard &
	operator=(const Steensgaard &) = delete;

	Steensgaard &
	operator=(Steensgaard &&) = delete;

	std::unique_ptr<PointsToGraph>
	Analyze(
    const RvsdgModule & module,
    const StatisticsDescriptor & sd) override;

private:
	void
	ResetState();

	void
	Analyze(const jive::graph & graph);

	void
	Analyze(jive::region & region);

	void
	Analyze(const lambda::node & node);

	void
	Analyze(const delta::node & node);

	void
	Analyze(const phi::node & node);

	void
	Analyze(const jive::gamma_node & node);

	void
	Analyze(const jive::theta_node & node);

	void
	Analyze(const jive::simple_node & node);

	void
	Analyze(const jive::structural_node & node);

	void
	AnalyzeAlloca(const jive::simple_node & node);

	void
	AnalyzeMalloc(const jive::simple_node & node);

	void
	AnalyzeLoad(const LoadNode & loadNode);

	void
	AnalyzeStore(const StoreNode & storeNode);

	void
	AnalyzeCall(const CallNode & callNode);

	void
	AnalyzeGep(const jive::simple_node & node);

	void
	AnalyzeBitcast(const jive::simple_node & node);

	void
	AnalyzeBits2ptr(const jive::simple_node & node);

	void
	AnalyzeConstantPointerNull(const jive::simple_node & node);

	void
	AnalyzeUndef(const jive::simple_node & node);

	void
	AnalyzeMemcpy(const jive::simple_node & node);

	void
	AnalyzeConstantArray(const jive::simple_node & node);

	void
	AnalyzeConstantStruct(const jive::simple_node & node);

	void
	AnalyzeConstantAggregateZero(const jive::simple_node & node);

	void
	AnalyzeExtractValue(const jive::simple_node & node);

	static std::unique_ptr<PointsToGraph>
	ConstructPointsToGraph(const LocationSet & lset);

	/** \brief Perform a recursive union of Location \p x and \p y.
	*/
	void
	join(Location & x, Location & y);

	LocationSet locationSet_;
};

}}

#endif
