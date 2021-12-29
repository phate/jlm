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

namespace phi { class node; }
}

namespace jlm {

namespace delta { class node; }
namespace lambda { class node; }

namespace aa {

class location;
class PointsToGraph;

/**
* FIXME: Some documentation
*/
class locationset final {
private:
	using locdjset = typename jlm::disjointset<jlm::aa::location*>;

public:
	using const_iterator = std::unordered_map<
	  const jive::output*
	, jlm::aa::location*
	>::const_iterator;

	~locationset();

	locationset();

	locationset(const locationset &) = delete;

	locationset(locationset &&) = delete;

	locationset &
	operator=(const locationset &) = delete;

	locationset &
	operator=(locationset &&) = delete;

	locdjset::set_iterator
	begin() const
	{
		return djset_.begin();
	}

	locdjset::set_iterator
	end() const
	{
		return djset_.end();
	}

	jlm::aa::location *
	insert(const jive::node * node);

	location *
	insert(const jive::argument * argument);

	location *
	insertDummy();

	bool
	contains(const jive::output * output) const noexcept;

	jlm::aa::location *
	FindOrInsert(const jive::output * output, bool unknown);

	const disjointset<jlm::aa::location*>::set &
	set(jlm::aa::location * l) const
	{
		return *djset_.find(l);
	}

	/*
		FIXME: The implementation of find can be expressed using set().
	*/
	jlm::aa::location *
	find(jlm::aa::location * l) const;

	jlm::aa::location *
	Find(const jive::output * output);

	jlm::aa::location *
	merge(jlm::aa::location * l1, jlm::aa::location * l2);

	std::string
	to_dot() const;

	void
	clear();

private:
	jlm::aa::location *
	Insert(const jive::output * output, bool unknown);

	jlm::aa::location *
	lookup(const jive::output * output);

	disjointset<jlm::aa::location*> djset_;
	std::vector<std::unique_ptr<jlm::aa::location>> locations_;
	std::unordered_map<const jive::output*, jlm::aa::location*> map_;
};

/**
* \brief FIXME: some documentation
*/
class Steensgaard final : public AliasAnalysis {
public:
	virtual
	~Steensgaard();

	Steensgaard() = default;

	Steensgaard(const Steensgaard &) = delete;

	Steensgaard(Steensgaard &&) = delete;

	Steensgaard &
	operator=(const Steensgaard &) = delete;

	Steensgaard &
	operator=(Steensgaard &&) = delete;

	virtual std::unique_ptr<PointsToGraph>
	Analyze(const rvsdg_module & module) override;

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
	Analyze(const jive::phi::node & node);

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
	AnalyzeLoad(const jive::simple_node & node);

	void
	AnalyzeStore(const jive::simple_node & node);

	void
	AnalyzeCall(const jive::simple_node & node);

	void
	AnalyzeGep(const jive::simple_node & node);

	void
	AnalyzeBitcast(const jive::simple_node & node);

	void
	AnalyzeBits2ptr(const jive::simple_node & node);

	void
	AnalyzeNull(const jive::simple_node & node);

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

	std::unique_ptr<PointsToGraph>
	ConstructPointsToGraph(const locationset & lset) const;

	/**
	* \brief Peform a recursive union of location \p x and \p y.
	*
	* FIXME
	*/
	void
	join(location & x, location & y);

	locationset lset_;
};

}}

#endif
