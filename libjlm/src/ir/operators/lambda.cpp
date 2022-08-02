/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/util/strfmt.hpp>

#include <jive/rvsdg/gamma.hpp>
#include <jive/rvsdg/theta.hpp>

#include <deque>

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

node::fctresult_range
node::fctresults()
{
	fctresiterator end(nullptr);

	if (nfctresults() == 0)
		return fctresult_range(end, end);

	fctresiterator begin(fctresult(0));
	return fctresult_range(begin, end);
}

node::fctresult_constrange
node::fctresults() const
{
	fctresconstiterator end(nullptr);

	if (nfctresults() == 0)
		return fctresult_constrange(end, end);

	fctresconstiterator begin(fctresult(0));
	return fctresult_constrange(begin, end);
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
	const FunctionType & type,
	const std::string & name,
	const jlm::linkage & linkage,
	const attributeset & attributes)
{
	lambda::operation op(type, name, linkage, std::move(attributes));
	auto node = new lambda::node(parent, std::move(op));

  for (auto & argumentType : type.Arguments())
		lambda::fctargument::create(node->subregion(), argumentType);

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

	if (type().NumResults() != results.size())
		throw jlm::error("Incorrect number of results.");

	for (size_t n = 0; n < results.size(); n++) {
		auto & expected = type().ResultType(n);
		auto & received = results[n]->type();
		if (results[n]->type() != type().ResultType(n))
			throw jlm::error("Expected " + expected.debug_string() + ", got " + received.debug_string());

		if (results[n]->region() != subregion())
			throw jlm::error("Invalid operand region.");
	}

	for (const auto & origin : results)
		lambda::result::create(origin);

	return output::create(this, PointerType(type()));
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
    for (size_t n = 0; n < nfctarguments(); n++){
        lambda->fctargument(n)->set_attributes(fctargument(n)->attributes());
        subregionmap.insert(fctargument(n), lambda->fctargument(n));
    }

	/* copy subregion */
	subregion()->copy(lambda->subregion(), subregionmap, false, false);

	/* collect function results */
	std::vector<jive::output*> results;
	for (auto & result : fctresults())
		results.push_back(subregionmap.lookup(result.origin()));

	/* finalize lambda */
	auto o = lambda->finalize(results);
	smap.insert(output(), o);

	return lambda;
}

/*
	FIXME: This function should be in jive.
*/
static jive::node *
input_node(const jive::input * input)
{
	using namespace jive;

	auto i = dynamic_cast<const jive::node_input*>(input);
	return i ? i->node() : nullptr;
}

bool
node::direct_calls(std::vector<jive::simple_node*> * calls) const
{
	std::deque<jive::input*> worklist;
	worklist.insert(worklist.end(), output()->begin(), output()->end());

	bool has_only_direct_calls = true;
	while (!worklist.empty()) {
		auto input = worklist.front();
		worklist.pop_front();

		if (auto cvinput = dynamic_cast<lambda::cvinput*>(input)) {
			auto argument = cvinput->argument();
			worklist.insert(worklist.end(), argument->begin(), argument->end());
			continue;
		}

		if (auto gamma_input = dynamic_cast<jive::gamma_input*>(input)) {
			for (auto & argument : *gamma_input)
				worklist.insert(worklist.end(), argument.begin(), argument.end());
			continue;
		}

		if (auto result = is_gamma_result(input)) {
			auto output = result->output();
			worklist.insert(worklist.end(), output->begin(), output->end());
			continue;
		}

		if (auto theta_input = dynamic_cast<jive::theta_input*>(input)) {
			auto argument = theta_input->argument();
			worklist.insert(worklist.end(), argument->begin(), argument->end());
			continue;
		}

		if (auto result = is_theta_result(input)) {
			auto output = result->output();
			worklist.insert(worklist.end(), output->begin(), output->end());
			continue;
		}

		if (auto cvinput = dynamic_cast<phi::cvinput*>(input)) {
			auto argument = cvinput->argument();
			worklist.insert(worklist.end(), argument->begin(), argument->end());
			continue;
		}

		if (auto rvresult = dynamic_cast<phi::rvresult*>(input)) {
			auto argument = rvresult->argument();
			worklist.insert(worklist.end(), argument->begin(), argument->end());

			auto output = rvresult->output();
			worklist.insert(worklist.end(), output->begin(), output->end());
			continue;
		}

		if (auto cvinput = dynamic_cast<delta::cvinput*>(input)) {
			auto argument = cvinput->arguments.first();
			worklist.insert(worklist.end(), argument->begin(), argument->end());
			continue;
		}

		auto node = input_node(input);
		if (is<CallOperation>(node) && input == node->input(0)) {
			if (calls != nullptr)
				calls->push_back(static_cast<jive::simple_node*>(node));
			continue;
		}

		if (is_export(input) || is<jive::simple_op>(node)) {
			if (calls == nullptr)
				return false;

			has_only_direct_calls = false;
			continue;
		}

		JLM_UNREACHABLE("This should have never happend!");
	}

	return has_only_direct_calls;
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
