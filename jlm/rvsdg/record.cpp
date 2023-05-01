/*
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/record.hpp>

namespace jive {

/* record type */

rcdtype::~rcdtype() noexcept
{}

std::string
rcdtype::debug_string() const
{
	return "rcd";
}

bool
rcdtype::operator==(const jive::type & other) const noexcept
{
	auto type = dynamic_cast<const rcdtype*>(&other);
	return type != nullptr
	    && declaration() == type->declaration();
}

std::unique_ptr<jive::type>
rcdtype::copy() const
{
	return std::unique_ptr<jive::type>(new rcdtype(*this));
}

}
