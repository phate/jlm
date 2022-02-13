/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/aggregation.hpp>
#include <jlm/ir/annotation.hpp>
#include <jlm/ir/basic-block.hpp>
#include <jlm/ir/operators/operators.hpp>

#include <algorithm>
#include <functional>
#include <typeindex>

namespace jlm {

DemandSet::~DemandSet()
{}

/* read-write annotation */

static void
annotaterw(const aggnode * node, DemandMap & dm);

static void
annotaterw(const entryaggnode * node, DemandMap & dm)
{
	auto ds = DemandSet::create();
	for (const auto & argument : *node) {
		ds->allwrites.insert(&argument);
		ds->fullwrites.insert(&argument);
	}

	dm[node] = std::move(ds);
}

static void
annotaterw(const exitaggnode * node, DemandMap & dm)
{
	auto ds = DemandSet::create();
	for (const auto & result : *node)
		ds->reads.insert(result);

	dm[node] = std::move(ds);
}

static void
annotaterw(const blockaggnode * node, DemandMap & dm)
{
	auto & bb = node->tacs();

	auto ds = DemandSet::create();
	for (auto it = bb.rbegin(); it != bb.rend(); it++) {
		auto & tac = *it;
		if (is<assignment_op>(tac->operation())) {
			/*
					We need special treatment for assignment operation, since the variable
					they assign the value to is modeled as an argument of the tac.
			*/
			JLM_ASSERT(tac->noperands() == 2 && tac->nresults() == 0);
			ds->reads.remove(tac->operand(0));
			ds->allwrites.insert(tac->operand(0));
			ds->fullwrites.insert(tac->operand(0));
			ds->reads.insert(tac->operand(1));
		} else {
			for (size_t n = 0; n < tac->nresults(); n++) {
				ds->reads.remove(tac->result(n));
				ds->allwrites.insert(tac->result(n));
				ds->fullwrites.insert(tac->result(n));
			}
			for (size_t n = 0; n < tac->noperands(); n++)
				ds->reads.insert(tac->operand(n));
		}
	}

	dm[node] = std::move(ds);
}

static void
annotaterw(const linearaggnode * node, DemandMap & dm)
{
	auto ds = DemandSet::create();
	for (ssize_t n = node->nchildren()-1; n >= 0; n--) {
		auto & cs = *dm[node->child(n)];
		for (const auto & v : cs.fullwrites)
			ds->reads.remove(v);
		ds->reads.insert(cs.reads);
		ds->allwrites.insert(cs.allwrites);
		ds->fullwrites.insert(cs.fullwrites);
	}

	dm[node] = std::move(ds);
}

static void
annotaterw(const branchaggnode * node, DemandMap & dm)
{
	auto ds = DemandSet::create();
	ds->reads = dm[node->child(0)]->reads;
	ds->allwrites = dm[node->child(0)]->allwrites;
	ds->fullwrites = dm[node->child(0)]->fullwrites;
	for (size_t n = 1; n < node->nchildren(); n++) {
		auto & cs = *dm[node->child(n)];
		ds->allwrites.insert(cs.allwrites);
		ds->fullwrites.intersect(cs.fullwrites);
		ds->reads.insert(cs.reads);
	}

	dm[node] = std::move(ds);
}

static void
annotaterw(const loopaggnode * node, DemandMap & dm)
{
	auto ds = DemandSet::create();
	ds->reads = dm[node->child(0)]->reads;
	ds->allwrites = dm[node->child(0)]->allwrites;
	ds->fullwrites = dm[node->child(0)]->fullwrites;
	dm[node] = std::move(ds);
}

template<class T> static void
annotaterw(const aggnode * node, DemandMap & dm)
{
	JLM_ASSERT(is<T>(node));
	annotaterw(static_cast<const T*>(node), dm);
}

static void
annotaterw(const aggnode * node, DemandMap & dm)
{
	static std::unordered_map<
		std::type_index,
		void(*)(const aggnode*, DemandMap&)
	> map({
	  {typeid(entryaggnode), annotaterw<entryaggnode>}
	, {typeid(exitaggnode), annotaterw<exitaggnode>}
	, {typeid(blockaggnode), annotaterw<blockaggnode>}
	, {typeid(linearaggnode), annotaterw<linearaggnode>}
	, {typeid(branchaggnode), annotaterw<branchaggnode>}
	, {typeid(loopaggnode), annotaterw<loopaggnode>}
	});

	for (size_t n = 0; n < node->nchildren(); n++)
		annotaterw(node->child(n), dm);

	JLM_ASSERT(map.find(typeid(*node)) != map.end());
	return map[typeid(*node)](node, dm);
}

/* DemandSet annotation */

static void
annotateds(const aggnode*, VariableSet&, DemandMap&);

static void
annotateds(
  const entryaggnode * node,
  VariableSet & pds,
  DemandMap & dm)
{
	auto & ds = dm[node];
	ds->bottom = pds;

	pds.remove(ds->fullwrites);

	ds->top = pds;
}

static void
annotateds(
  const exitaggnode * node,
  VariableSet & pds,
  DemandMap & dm)
{
	auto & ds = dm[node];
	ds->bottom = pds;

	for (const auto & v : ds->reads)
		pds.insert(v);

	ds->top = pds;
}

static void
annotateds(
  const blockaggnode * node,
  VariableSet & pds,
  DemandMap & dm)
{
	auto & ds = dm[node];
	ds->bottom = pds;

	pds.remove(ds->fullwrites);
	pds.insert(ds->reads);

	ds->top = pds;
}

static void
annotateds(
  const linearaggnode * node,
  VariableSet & pds,
  DemandMap & dm)
{
	auto & ds = dm[node];
	ds->bottom = pds;

	for (ssize_t n = node->nchildren()-1; n >= 0; n--)
		annotateds(node->child(n), pds, dm);

	pds.remove(ds->fullwrites);
	pds.insert(ds->reads);

	ds->top = pds;
}

static void
annotateds(
  const branchaggnode * node,
  VariableSet & pds,
  DemandMap & dm)
{
	auto & ds = dm[node];

	VariableSet passby = pds;
	passby.subtract(ds->allwrites);

	VariableSet bottom = pds;
	bottom.intersect(ds->allwrites);
	ds->bottom = bottom;

	for (size_t n = 0; n < node->nchildren(); n++) {
		auto tmp = bottom;
		annotateds(node->child(n), tmp, dm);
	}


	pds.remove(ds->fullwrites);
	pds.insert(ds->reads);
	ds->top = pds;

	pds.insert(passby);
}

static void
annotateds(
  const loopaggnode * node,
  VariableSet & pds,
  DemandMap & dm)
{
	auto & ds = dm[node];

	pds.insert(ds->reads);
	ds->bottom = ds->top = pds;
	annotateds(node->child(0), pds, dm);

	for (const auto & v : ds->reads)
		JLM_ASSERT(pds.contains(v));

	for (const auto & v : ds->fullwrites)
		if (!ds->reads.contains(v))
			JLM_ASSERT(!pds.contains(v));
}

template<class T> static void
annotateds(
  const aggnode * node,
  VariableSet & pds,
  DemandMap & dm)
{
	JLM_ASSERT(is<T>(node));
	annotateds(static_cast<const T*>(node), pds, dm);
}

static void
annotateds(
  const aggnode * node,
  VariableSet & pds,
  DemandMap & dm)
{
	static std::unordered_map<
		std::type_index,
		void(*)(const aggnode*, VariableSet&, DemandMap&)
	> map({
	  {typeid(entryaggnode), annotateds<entryaggnode>}
	, {typeid(exitaggnode), annotateds<exitaggnode>}
	, {typeid(blockaggnode), annotateds<blockaggnode>}
	, {typeid(linearaggnode), annotateds<linearaggnode>}
	, {typeid(branchaggnode), annotateds<branchaggnode>}
	, {typeid(loopaggnode), annotateds<loopaggnode>}
	});

	JLM_ASSERT(map.find(typeid(*node)) != map.end());
	return map[typeid(*node)](node, pds, dm);
}

DemandMap
Annotate(const aggnode & root)
{
	DemandMap dm;
	VariableSet ds;
	annotaterw(&root, dm);
	annotateds(&root, ds, dm);
	return dm;
}

}
