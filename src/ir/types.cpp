/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/types.hpp>
#include <jlm/util/strfmt.hpp>

namespace jlm {

/* pointer type */

ptrtype::~ptrtype()
{}

std::string
ptrtype::debug_string() const
{
	return pointee_type().debug_string() + "*";
}

bool
ptrtype::operator==(const jive::base::type & other) const noexcept
{
	auto type = dynamic_cast<const jlm::ptrtype*>(&other);
	return type && type->pointee_type() == pointee_type();
}

std::unique_ptr<jive::base::type>
ptrtype::copy() const
{
	return std::unique_ptr<jive::base::type>(new ptrtype(*this));
}

/* array type */

arraytype::~arraytype()
{}

std::string
arraytype::debug_string() const
{
	return strfmt("[ ", nelements(), " x ", type_->debug_string(), " ]");
}

bool
arraytype::operator==(const jive::base::type & other) const noexcept
{
	auto type = dynamic_cast<const jlm::arraytype*>(&other);
	return type && type->element_type() == element_type() && type->nelements() == nelements();
}

std::unique_ptr<jive::base::type>
arraytype::copy() const
{
	return std::unique_ptr<jive::base::type>(new arraytype(*this));
}

}
