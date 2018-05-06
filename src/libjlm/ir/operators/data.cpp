/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators/data.hpp>

#include <jive/rvsdg/substitution.h>

namespace jlm {

/* data operator */

std::string
data_op::debug_string() const
{
	return "DATA";
}

std::unique_ptr<jive::operation>
data_op::copy() const
{
	return std::unique_ptr<jive::operation>(new data_op(*this));
}

bool
data_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const data_op*>(&other);
	return op && op->linkage_ == linkage_ && op->constant_ == constant_;
}

/* rvsdg data node */

rvsdg_data_node::~rvsdg_data_node()
{}

rvsdg_data_node *
rvsdg_data_node::copy(jive::region * region, jive::substitution_map & smap) const
{
	auto & op = *static_cast<const data_op*>(&operation());

	data_builder db;
	db.begin(region, op.linkage(), op.constant());

	/* add dependencies */
	jive::substitution_map rmap;
	for (const auto & od : *this) {
		auto nd = db.add_dependency(smap.lookup(od->origin()));
		rmap.insert(dynamic_cast<jive::structural_input*>(od)->arguments.first(), nd);
	}

	/* copy subregion */
	subregion()->copy(db.region(), rmap, false, false);

	auto result = rmap.lookup(subregion()->result(0)->origin());
	auto data = db.end(result);
	smap.insert(output(0), data);
	return static_cast<rvsdg_data_node*>(data->node());
}

}
