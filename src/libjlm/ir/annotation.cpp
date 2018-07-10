/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlm/ir/aggregation.hpp>
#include <jlm/jlm/ir/annotation.hpp>
#include <jlm/jlm/ir/basic-block.hpp>
#include <jlm/jlm/ir/operators/operators.hpp>

#include <algorithm>
#include <functional>
#include <typeindex>

namespace jlm {

demandset::~demandset()
{}

static variableset
intersect(const variableset & vs1, const variableset & vs2)
{
	variableset intersect;
	for (const auto & v : vs1) {
		if (vs2.find(v) != vs2.end())
			intersect.insert(v);
	}
	return intersect;
}

/* read-write annotation */

static void
annotaterw(const aggnode * node, demandmap & dm);

static void
annotaterw(const entryaggnode * node, demandmap & dm)
{
	auto ds = demandset::create();
	for (const auto & argument : *node)
		ds->writes.insert(argument);

	dm[node] = std::move(ds);
}

static void
annotaterw(const exitaggnode * node, demandmap & dm)
{
	auto ds = demandset::create();
	for (const auto & result : *node)
		ds->reads.insert(result);

	dm[node] = std::move(ds);
}

static void
annotaterw(const blockaggnode * node, demandmap & dm)
{
	auto & bb = node->tacs();

	auto ds = demandset::create();
	for (auto it = bb.rbegin(); it != bb.rend(); it++) {
		auto & tac = *it;
		if (is<assignment_op>(tac->operation())) {
			/*
					We need special treatment for assignment operation, since the variable
					they assign the value to is modeled as an argument of the tac.
			*/
			JLM_DEBUG_ASSERT(tac->ninputs() == 2 && tac->noutputs() == 0);
			ds->reads.erase(tac->input(0));
			ds->writes.insert(tac->input(0));
			ds->reads.insert(tac->input(1));
		} else {
			for (size_t n = 0; n < tac->noutputs(); n++) {
				ds->reads.erase(tac->output(n));
				ds->writes.insert(tac->output(n));
			}
			for (size_t n = 0; n < tac->ninputs(); n++)
				ds->reads.insert(tac->input(n));
		}
	}

	dm[node] = std::move(ds);
}

static void
annotaterw(const linearaggnode * node, demandmap & dm)
{
	auto ds = demandset::create();
	for (ssize_t n = node->nchildren()-1; n >= 0; n--) {
		auto & cs = *dm[node->child(n)];
		for (const auto & v : cs.writes)
			ds->reads.erase(v);
		ds->reads.insert(cs.reads.begin(), cs.reads.end());
		ds->writes.insert(cs.writes.begin(), cs.writes.end());
	}

	dm[node] = std::move(ds);
}

static void
annotaterw(const branchaggnode * node, demandmap & dm)
{
	auto ds = demandset::create();
	ds->reads = dm[node->child(0)]->reads;
	ds->writes = dm[node->child(0)]->writes;
	for (size_t n = 1; n < node->nchildren(); n++) {
		auto & cs = *dm[node->child(n)];
		ds->writes = intersect(ds->writes, cs.writes);
		ds->reads.insert(cs.reads.begin(), cs.reads.end());
	}

	dm[node] = std::move(ds);
}

static void
annotaterw(const loopaggnode * node, demandmap & dm)
{
	auto ds = demandset::create();
	ds->reads = dm[node->child(0)]->reads;
	ds->writes = dm[node->child(0)]->writes;
	dm[node] = std::move(ds);
}

template<class T> static void
annotaterw(const aggnode * node, demandmap & dm)
{
	JLM_DEBUG_ASSERT(is<T>(node));
	annotaterw(static_cast<const T*>(node), dm);
};

static void
annotaterw(const aggnode * node, demandmap & dm)
{
	static std::unordered_map<
		std::type_index,
		void(*)(const aggnode*, demandmap&)
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

	JLM_DEBUG_ASSERT(map.find(typeid(*node)) != map.end());
	return map[typeid(*node)](node, dm);
}

/* demandset annotation */

static void
annotateds(const aggnode*, variableset&, demandmap&);

static void
annotateds(
	const entryaggnode * node,
	variableset & pds,
	demandmap & dm)
{
	auto & ds = dm[node];
	ds->bottom = pds;

	for (const auto & v : ds->writes)
		pds.erase(v);

	ds->top = pds;
}

static void
annotateds(
	const exitaggnode * node,
	variableset & pds,
	demandmap & dm)
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
	variableset & pds,
	demandmap & dm)
{
	auto & ds = dm[node];
	ds->bottom = pds;

	for (const auto & v : ds->writes)
		pds.erase(v);
	pds.insert(ds->reads.begin(), ds->reads.end());

	ds->top = pds;
}

static void
annotateds(
	const linearaggnode * node,
	variableset & pds,
	demandmap & dm)
{
	auto & ds = dm[node];
	ds->bottom = pds;

	for (ssize_t n = node->nchildren()-1; n >= 0; n--)
		annotateds(node->child(n), pds, dm);

	for (const auto & v : ds->writes)
		pds.erase(v);
	pds.insert(ds->reads.begin(), ds->reads.end());

	ds->top = pds;
}

static void
annotateds(
	const branchaggnode * node,
	variableset & pds,
	demandmap & dm)
{
	auto & ds = dm[node];
	ds->bottom = pds;

	for (size_t n = 0; n < node->nchildren(); n++) {
		auto tmp = pds;
		annotateds(node->child(n), tmp, dm);
	}

	for (const auto & v : ds->writes)
		pds.erase(v);
	pds.insert(ds->reads.begin(), ds->reads.end());

	ds->top = pds;
}

static void
annotateds(
	const loopaggnode * node,
	variableset & pds,
	demandmap & dm)
{
	auto & ds = dm[node];

	variableset loop = ds->reads;
	for (const auto & v : ds->writes) {
		if (pds.find(v) != pds.end())
			loop.insert(v);
	}
	ds->bottom = ds->top = loop;

	annotateds(node->child(0), loop, dm);
	pds.insert(ds->reads.begin(), ds->reads.end());
}

template<class T> static void
annotateds(
	const aggnode * node,
	variableset & pds,
	demandmap & dm)
{
	JLM_DEBUG_ASSERT(is<T>(node));
	annotateds(static_cast<const T*>(node), pds, dm);
}

static void
annotateds(
	const aggnode * node,
	variableset & pds,
	demandmap & dm)
{
	static std::unordered_map<
		std::type_index,
		void(*)(const aggnode*, variableset&, demandmap&)
	> map({
	  {typeid(entryaggnode), annotateds<entryaggnode>}
	, {typeid(exitaggnode), annotateds<exitaggnode>}
	, {typeid(blockaggnode), annotateds<blockaggnode>}
	, {typeid(linearaggnode), annotateds<linearaggnode>}
	, {typeid(branchaggnode), annotateds<branchaggnode>}
	, {typeid(loopaggnode), annotateds<loopaggnode>}
	});

	JLM_DEBUG_ASSERT(map.find(typeid(*node)) != map.end());
	return map[typeid(*node)](node, pds, dm);
}

demandmap
annotate(const aggnode & root)
{
	demandmap dm;
	variableset ds;
	annotaterw(&root, dm);
	annotateds(&root, ds, dm);
	return dm;
}

}
