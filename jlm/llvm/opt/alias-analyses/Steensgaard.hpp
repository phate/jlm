/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_OPT_ALIAS_ANALYSES_STEENSGAARD_HPP
#define JLM_LLVM_OPT_ALIAS_ANALYSES_STEENSGAARD_HPP

#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/util/disjointset.hpp>

#include <string>

namespace jlm::llvm
{

namespace delta { class node; }
namespace lambda { class node; }
namespace phi { class node; }

class LoadNode;
class StoreNode;

namespace aa {

class Location;
class PointsToGraph;
class RegisterLocation;

enum class PointsToFlags {
  PointsToNone           = 1 << 0,
  PointsToUnknownMemory  = 1 << 1,
  PointsToExternalMemory = 1 << 2,
  PointsToEscapedMemory  = 1 << 3,
};

static inline PointsToFlags
operator|(PointsToFlags lhs, PointsToFlags rhs)
{
  typedef typename std::underlying_type<PointsToFlags>::type underlyingType;
  return static_cast<PointsToFlags>(static_cast<underlyingType>(lhs) | static_cast<underlyingType>(rhs));
}

static inline PointsToFlags
operator&(PointsToFlags lhs, PointsToFlags rhs)
{
  typedef typename std::underlying_type<PointsToFlags>::type underlyingType;
  return static_cast<PointsToFlags>(static_cast<underlyingType>(lhs) & static_cast<underlyingType>(rhs));
}

/** \brief LocationSet class
*/
class LocationSet final
{
public:
	using DisjointLocationSet = typename jlm::util::disjointset<Location*>;

	using const_iterator = std::unordered_map<
	  const jlm::rvsdg::output*
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
		return DisjointLocationSet_.begin();
	}

	DisjointLocationSet::set_iterator
	end() const
	{
		return DisjointLocationSet_.end();
	}

  Location &
  InsertAllocaLocation(const jlm::rvsdg::node & node);

  Location &
  InsertMallocLocation(const jlm::rvsdg::node & node);

  Location &
  InsertLambdaLocation(const lambda::node & lambda);

  Location &
  InsertDeltaLocation(const delta::node & delta);

	Location &
	InsertImportLocation(const jlm::rvsdg::argument & argument);

	Location &
	InsertDummyLocation();

	bool
	Contains(const jlm::rvsdg::output & output) const noexcept;

	Location &
	FindOrInsertRegisterLocation(
    const jlm::rvsdg::output & output,
    PointsToFlags pointsToFlags);

	const DisjointLocationSet::set &
	GetSet(Location & location) const
	{
		return *DisjointLocationSet_.find(&location);
	}

  size_t
  NumDisjointSets() const noexcept
  {
    return DisjointLocationSet_.nsets();
  }

  size_t
  NumLocations() const noexcept
  {
    return DisjointLocationSet_.nvalues();
  }

	Location &
	GetRootLocation(Location & location) const;

	Location &
	Find(const jlm::rvsdg::output & output);

  RegisterLocation *
  LookupRegisterLocation(const jlm::rvsdg::output & output);

	Location &
	Merge(Location & location1, Location & location2);

	std::string
	ToDot() const;

	void
	Clear();

private:
	RegisterLocation &
	InsertRegisterLocation(
    const jlm::rvsdg::output & output,
    PointsToFlags pointsToFlags);

	DisjointLocationSet DisjointLocationSet_;
	std::vector<std::unique_ptr<Location>> Locations_;
	std::unordered_map<const jlm::rvsdg::output*, RegisterLocation*> LocationMap_;
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
    jlm::util::StatisticsCollector & statisticsCollector) override;

private:
	void
	ResetState();

	void
	Analyze(const jlm::rvsdg::graph & graph);

	void
	Analyze(jlm::rvsdg::region & region);

	void
	Analyze(const lambda::node & node);

	void
	Analyze(const delta::node & node);

	void
	Analyze(const phi::node & node);

	void
	Analyze(const jlm::rvsdg::gamma_node & node);

	void
	Analyze(const jlm::rvsdg::theta_node & node);

	void
	Analyze(const jlm::rvsdg::simple_node & node);

	void
	Analyze(const jlm::rvsdg::structural_node & node);

	void
	AnalyzeAlloca(const jlm::rvsdg::simple_node & node);

	void
	AnalyzeMalloc(const jlm::rvsdg::simple_node & node);

	void
	AnalyzeLoad(const LoadNode & loadNode);

	void
	AnalyzeStore(const StoreNode & storeNode);

	void
	AnalyzeCall(const CallNode & callNode);

	void
	AnalyzeGep(const jlm::rvsdg::simple_node & node);

	void
	AnalyzeBitcast(const jlm::rvsdg::simple_node & node);

	void
	AnalyzeBits2ptr(const jlm::rvsdg::simple_node & node);

	void
	AnalyzeConstantPointerNull(const jlm::rvsdg::simple_node & node);

	void
	AnalyzeUndef(const jlm::rvsdg::simple_node & node);

	void
	AnalyzeMemcpy(const jlm::rvsdg::simple_node & node);

	void
	AnalyzeConstantArray(const jlm::rvsdg::simple_node & node);

	void
	AnalyzeConstantStruct(const jlm::rvsdg::simple_node & node);

	void
	AnalyzeConstantAggregateZero(const jlm::rvsdg::simple_node & node);

	void
	AnalyzeExtractValue(const jlm::rvsdg::simple_node & node);

	static std::unique_ptr<PointsToGraph>
	ConstructPointsToGraph(const LocationSet & locationSets);

	/** \brief Perform a recursive union of Location \p x and \p y.
	*/
	void
	join(Location & x, Location & y);

	LocationSet LocationSet_;
};

}}

#endif
