/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/rvsdg/substitution.hpp>
#include <jive/types/function.hpp>

namespace jive {

/* function type */

fcttype::~fcttype() noexcept
{}

fcttype::fcttype(
	const std::vector<const jive::type*> & argument_types,
	const std::vector<const jive::type*> & result_types)
: jive::valuetype()
{
	for (const auto & type : argument_types)
		argument_types_.push_back(std::unique_ptr<jive::type>(type->copy()));

	for (const auto & type : result_types)
		result_types_.push_back(std::unique_ptr<jive::type>(type->copy()));
}

fcttype::fcttype(
	const std::vector<std::unique_ptr<jive::type>> & argument_types,
	const std::vector<std::unique_ptr<jive::type>> & result_types)
: jive::valuetype()
{
	for (size_t i = 0; i < argument_types.size(); i++)
		argument_types_.push_back(std::unique_ptr<jive::type>(argument_types[i]->copy()));

	for (size_t i = 0; i < result_types.size(); i++)
		result_types_.push_back(std::unique_ptr<jive::type>(result_types[i]->copy()));
}

fcttype::fcttype(const jive::fcttype & rhs)
: jive::valuetype(rhs)
{
	for (size_t i = 0; i < rhs.narguments(); i++)
		argument_types_.push_back(std::unique_ptr<jive::type>(rhs.argument_type(i).copy()));

	for (size_t i = 0; i < rhs.nresults(); i++)
		result_types_.push_back(std::unique_ptr<jive::type>(rhs.result_type(i).copy()));
}

fcttype::fcttype(jive::fcttype && other)
: jive::valuetype(other)
, result_types_(std::move(other.result_types_))
, argument_types_(std::move(other.argument_types_))
{}

std::string
fcttype::debug_string() const
{
	return "fct";
}

bool
fcttype::operator==(const jive::type & _other) const noexcept
{
	auto other = dynamic_cast<const jive::fcttype*>(&_other);
	if (other == nullptr)
		return false;

	if (this->nresults() != other->nresults())
		return false;

	if (this->narguments() != other->narguments())
		return false;

	for (size_t i = 0; i < this->nresults(); i++){
		if (this->result_type(i) != other->result_type(i))
			return false;
	}

	for (size_t i = 0; i < this->narguments(); i++){
		if (this->argument_type(i) != other->argument_type(i))
			return false;
	}

	return true;
}

std::unique_ptr<jive::type>
fcttype::copy() const
{
 return std::unique_ptr<jive::type>(new fcttype(*this));
}

jive::fcttype &
fcttype::operator=(const jive::fcttype & rhs)
{
	result_types_.clear();
	argument_types_.clear();

	for (size_t i = 0; i < rhs.narguments(); i++)
		argument_types_.push_back(std::unique_ptr<jive::type>(rhs.argument_type(i).copy()));

	for (size_t i = 0; i < rhs.nresults(); i++)
		result_types_.push_back(std::unique_ptr<jive::type>(rhs.result_type(i).copy()));

	return *this;
}

jive::fcttype &
fcttype::operator=(jive::fcttype && rhs)
{
	result_types_ = std::move(rhs.result_types_);
	argument_types_ = std::move(rhs.argument_types_);
	return *this;
}

/* apply operator */

apply_op::~apply_op() noexcept
{
}

bool
apply_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const apply_op*>(&other);
	if (!op || op->narguments() != narguments() || op->nresults() != nresults())
		return false;

	for (size_t n = 0; n < narguments(); n++) {
		if (argument(n) != op->argument(n))
			return false;
	}

	for (size_t n = 0; n < nresults(); n++) {
		if (result(n) != op->result(n))
			return false;
	}

	return true;
}

std::string
apply_op::debug_string() const
{
	return "APPLY";
}

std::unique_ptr<jive::operation>
apply_op::copy() const
{
	return std::unique_ptr<jive::operation>(new apply_op(*this));
}

std::vector<jive::port>
apply_op::create_operands(const fcttype & type)
{
	std::vector<jive::port> operands({type});
	for (size_t n = 0; n < type.narguments(); n++)
		operands.push_back(type.argument_type(n));

	return operands;
}

std::vector<jive::port>
apply_op::create_results(const fcttype & type)
{
	std::vector<jive::port> results;
	for (size_t n = 0; n < type.nresults(); n++)
		results.push_back({type.result_type(n)});

	return results;
}

/* lambda operator */

lambda_op::~lambda_op() noexcept
{}

bool
lambda_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const lambda_op*>(&other);
	return op && op->function_type() == function_type();
}

std::string
lambda_op::debug_string() const
{
	return "LAMBDA";
}

std::unique_ptr<jive::operation>
lambda_op::copy() const
{
	return std::unique_ptr<jive::operation>(new lambda_op(*this));
}

/* lambda node class */

lambda_node::~lambda_node()
{}

jive::lambda_node *
lambda_node::copy(jive::region * region, jive::substitution_map & smap) const
{
	jive::lambda_builder lb;
	auto arguments = lb.begin_lambda(region, *static_cast<const lambda_op*>(&operation()));

	/* add dependencies */
	jive::substitution_map rmap;
	for (size_t n = 0; n < function_type().narguments(); n++)
		rmap.insert(subregion()->argument(n), arguments[n]);
	for (const auto & odv : *this) {
		auto ndv = lb.add_dependency(smap.lookup(odv->origin()));
		rmap.insert(dynamic_cast<structural_input*>(odv)->arguments.first(), ndv);
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
