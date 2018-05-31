/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_AGGREGATION_ANNOTATION_HPP
#define JLM_IR_AGGREGATION_ANNOTATION_HPP

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace jlm {

class variable;

namespace agg {

class aggnode;

typedef std::unordered_set<const jlm::variable*> dset;

class demand_set {
public:
	virtual
	~demand_set();

	dset top;
	dset bottom;
};

static inline std::unique_ptr<demand_set>
create_demand_set(const dset & b)
{
	auto ds = std::make_unique<demand_set>();
	ds->bottom = b;
	return ds;
}

class branch_demand_set final : public demand_set {
public:
	virtual
	~branch_demand_set();

	dset cases_top;
	dset cases_bottom;
};

static inline std::unique_ptr<branch_demand_set>
create_branch_demand_set(const dset & b)
{
	auto ds = std::make_unique<branch_demand_set>();
	ds->bottom = b;
	return ds;
}

typedef std::unordered_map<const agg::aggnode*, std::unique_ptr<demand_set>> demand_map;

demand_map
annotate(jlm::agg::aggnode & root);

}}

#endif
