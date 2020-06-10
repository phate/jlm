/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators/delta.hpp>

#include <jive/rvsdg/substitution.h>

namespace jlm {

/* delta operator */

std::string
delta_op::debug_string() const
{
	return strfmt("DELTA[", name(), "]");
}

std::unique_ptr<jive::operation>
delta_op::copy() const
{
	return std::unique_ptr<jive::operation>(new delta_op(*this));
}

bool
delta_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const delta_op*>(&other);
	return op
	    && op->name_ == name_
	    && op->linkage_ == linkage_
	    && op->constant_ == constant_
	    && *op->type_ == *type_;
}

/* delta node */

delta_node::~delta_node()
{}

delta_node *
delta_node::copy(jive::region * region, jive::substitution_map & smap) const
{
	auto & op = *static_cast<const delta_op*>(&operation());

	delta_builder db;
	db.begin(region, op.type(), op.name(), op.linkage(), op.constant());

	/* add dependencies */
	jive::substitution_map rmap;
	for (const auto & input : *this) {
		auto nd = db.add_dependency(smap.lookup(input.origin()));
		rmap.insert(input.arguments.first(), nd);
	}

	/* copy subregion */
	subregion()->copy(db.region(), rmap, false, false);

	auto result = rmap.lookup(subregion()->result(0)->origin());
	auto data = db.end(result);
	smap.insert(output(0), data);
	return static_cast<delta_node*>(data->node());
}

}
