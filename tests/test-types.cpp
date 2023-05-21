/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-types.hpp"

namespace jlm::tests
{

/* valuetype */

valuetype::~valuetype()
{}

std::string
valuetype::debug_string() const
{
	return "valuetype";
}

bool
valuetype::operator==(const rvsdg::type & other) const noexcept
{
	return dynamic_cast<const valuetype*>(&other) != nullptr;
}

std::unique_ptr<rvsdg::type>
valuetype::copy() const
{
	return std::unique_ptr<rvsdg::type>(new valuetype(*this));
}

/* statetype */

statetype::~statetype()
{}

std::string
statetype::debug_string() const
{
	return "statetype";
}

bool
statetype::operator==(const rvsdg::type & other) const noexcept
{
	return dynamic_cast<const statetype*>(&other) != nullptr;
}

std::unique_ptr<rvsdg::type>
statetype::copy() const
{
	return std::unique_ptr<rvsdg::type>(new statetype(*this));
}

}
