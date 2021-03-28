/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/operators/lambda.hpp>
#include <jlm/util/strfmt.hpp>

namespace jlm {
namespace lambda {

/* lambda operation class */

operation::~operation()
{}

std::string
operation::debug_string() const
{
	return strfmt("LAMBDA[", name(), "]");
}

bool
operation::operator==(const jive::operation & other) const noexcept
{
	auto op = dynamic_cast<const lambda::operation*>(&other);
	return op
	    && op->type() == type()
	    && op->name() == name()
	    && op->linkage() == linkage()
			&& op->attributes() == attributes();
}

std::unique_ptr<jive::operation>
operation::copy() const
{
	return std::unique_ptr<jive::operation>(new operation(*this));
}

/* lambda node class */

node::~node()
{}

node::fctargument_range
node::fctarguments()
{
	fctargiterator end(nullptr);

	if (nfctarguments() == 0)
		return fctargument_range(end, end);

	fctargiterator begin(fctargument(0));
	return fctargument_range(begin, end);
}

node::fctargument_constrange
node::fctarguments() const
{
	fctargconstiterator end(nullptr);

	if (nfctarguments() == 0)
		return fctargument_constrange(end, end);

	fctargconstiterator begin(fctargument(0));
	return fctargument_constrange(begin, end);
}

node::ctxvar_range
node::ctxvars()
{
	cviterator end(nullptr);

	if (ncvarguments() == 0)
		return ctxvar_range(end, end);

	cviterator begin(input(0));
	return ctxvar_range(begin, end);
}

node::ctxvar_constrange
node::ctxvars() const
{
	cvconstiterator end(nullptr);

	if (ncvarguments() == 0)
		return ctxvar_constrange(end, end);

	cvconstiterator begin(input(0));
	return ctxvar_constrange(begin, end);
}

node::fctresiterator
node::begin_res()
{
	if (nfctresults() == 0)
		return end_res();

	auto res = static_cast<lambda::result*>(subregion()->result(0));
	return fctresiterator(res);
}

node::fctresconstiterator
node::begin_res() const
{
	if (nfctresults() == 0)
		return end_res();

	auto res = static_cast<const lambda::result*>(subregion()->result(0));
	return fctresconstiterator(res);
}

node::fctresiterator
node::end_res()
{
	return fctresiterator(nullptr);
}

node::fctresconstiterator
node::end_res() const
{
	return fctresconstiterator(nullptr);
}

cvinput *
node::input(size_t n) const noexcept
{
	return static_cast<cvinput*>(structural_node::input(n));
}

lambda::output *
node::output() const noexcept
{
	return static_cast<lambda::output*>(structural_node::output(0));
}

lambda::fctargument *
node::fctargument(size_t n) const noexcept
{
	return static_cast<lambda::fctargument*>(subregion()->argument(n));
}

lambda::cvargument *
node::cvargument(size_t n) const noexcept
{
	return input(n)->argument();
}

lambda::result *
node::fctresult(size_t n) const noexcept
{
	return static_cast<lambda::result*>(subregion()->result(n));
}

cvargument *
node::add_ctxvar(jive::output * origin)
{
	auto input = cvinput::create(this, origin);
	return cvargument::create(subregion(), input);
}

lambda::node *
node::create(
	jive::region * parent,
	const jive::fcttype & type,
	const std::string & name,
	const jlm::linkage & linkage,
	const attributeset & attributes)
{
	lambda::operation op(type, name, linkage, std::move(attributes));
	auto node = new lambda::node(parent, std::move(op));

	for (size_t n = 0; n < type.narguments(); n++) {
		auto & at = type.argument_type(n);
		lambda::fctargument::create(node->subregion(), at);
	}

	return node;
}

lambda::output *
node::finalize(const std::vector<jive::output*> & results)
{
	/* check if finalized was already called */
	if (noutputs() > 0) {
		JLM_ASSERT(noutputs() == 1);
		return output();
	}

	if (type().nresults() != results.size())
		throw jlm::error("Incorrect number of results.");

	for (size_t n = 0; n < results.size(); n++) {
		auto & expected = type().result_type(n);
		auto & received = results[n]->type();
		if (results[n]->type() != type().result_type(n))
			throw jlm::error("Expected " + expected.debug_string() + ", got " + received.debug_string());
	}

	for (const auto & origin : results)
		lambda::result::create(origin);

	return output::create(this, ptrtype(type()));
}

lambda::node *
node::copy(
	jive::region * region,
	const std::vector<jive::output*> & operands) const
{
	return static_cast<lambda::node*>(jive::node::copy(region, operands));
}

lambda::node *
node::copy(jive::region * region, jive::substitution_map & smap) const
{
	auto lambda = create(region, type(), name(), linkage(), attributes());

	/* add context variables */
	jive::substitution_map subregionmap;
	for (auto & cv : ctxvars()) {
		auto origin = smap.lookup(cv.origin());
		auto newcv = lambda->add_ctxvar(origin);
		subregionmap.insert(cv.argument(), newcv);
	}

	/* collect function arguments */
	for (size_t n = 0; n < nfctarguments(); n++)
		subregionmap.insert(fctargument(n), lambda->fctargument(n));

	/* copy subregion */
	subregion()->copy(lambda->subregion(), subregionmap, false, false);

	/* collect function results */
	std::vector<jive::output*> results;
	for (auto it = begin_res(); it != end_res(); it++)
		results.push_back(subregionmap.lookup(it->origin()));

	/* finalize lambda */
	auto o = lambda->finalize(results);
	smap.insert(output(), o);

	return lambda;
}

/* lambda context variable input class */

cvinput::~cvinput()
{}

cvargument *
cvinput::argument() const noexcept
{
	return static_cast<cvargument*>(arguments.first());
}

/* lambda output class */

output::~output()
{}

/* lambda function argument class */

fctargument::~fctargument()
{}

/* lambda context variable argument class */

cvargument::~cvargument()
{}

/* lambda result class */

result::~result()
{}

}}
