/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators/lambda.hpp>

#include <jive/rvsdg/substitution.h>

namespace jlm {

/* lambda operation */

lambda_op::~lambda_op()
{}

bool
lambda_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const lambda_op*>(&other);
	return op
	    && op->fcttype() == fcttype()
	    && op->name() == name()
	    && op->linkage() == linkage();
}

std::string
lambda_op::debug_string() const
{
	return strfmt("LAMBDA[", name(), "]");
}

std::unique_ptr<jive::operation>
lambda_op::copy() const
{
	return std::unique_ptr<jive::operation>(new lambda_op(*this));
}

/* lambda node */

lambda_node::~lambda_node()
{}

lambda_node *
lambda_node::copy(jive::region * region, jive::substitution_map & smap) const
{
	lambda_builder lb;
	auto arguments = lb.begin_lambda(region, *static_cast<const lambda_op*>(&operation()));

	/* add dependencies */
	jive::substitution_map rmap;
	for (size_t n = 0; n < fcttype().narguments(); n++)
		rmap.insert(subregion()->argument(n), arguments[n]);
	for (const auto & odv : *this) {
		auto ndv = lb.add_dependency(smap.lookup(odv.origin()));
		rmap.insert(odv.arguments.first(), ndv);
	}

	/* copy subregion */
	subregion()->copy(lb.subregion(), rmap, false, false);

	/* collect results */
	std::vector<jive::output*> results;
	for (size_t n = 0; n < subregion()->nresults(); n++)
		results.push_back(rmap.lookup(subregion()->result(n)->origin()));

	auto lambda = lb.end_lambda(results);
	smap.insert(output(0), lambda->output(0));
	return lambda;
}

}
