/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/common.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/util/strfmt.hpp>

#include <unordered_map>

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
ptrtype::operator==(const jive::type & other) const noexcept
{
	auto type = dynamic_cast<const jlm::ptrtype*>(&other);
	return type && type->pointee_type() == pointee_type();
}

std::unique_ptr<jive::type>
ptrtype::copy() const
{
	return std::unique_ptr<jive::type>(new ptrtype(*this));
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
arraytype::operator==(const jive::type & other) const noexcept
{
	auto type = dynamic_cast<const jlm::arraytype*>(&other);
	return type && type->element_type() == element_type() && type->nelements() == nelements();
}

std::unique_ptr<jive::type>
arraytype::copy() const
{
	return std::unique_ptr<jive::type>(new arraytype(*this));
}

/* floating point type */

fptype::~fptype()
{}

std::string
fptype::debug_string() const
{
	static std::unordered_map<fpsize, std::string> map({
	  {fpsize::half, "half"}
	, {fpsize::flt, "float"}
	, {fpsize::dbl, "double"}
	, {fpsize::x86fp80, "x86fp80"}
	});

	JLM_DEBUG_ASSERT(map.find(size()) != map.end());
	return map[size()];
}

bool
fptype::operator==(const jive::type & other) const noexcept
{
	auto type = dynamic_cast<const jlm::fptype*>(&other);
	return type && type->size() == size();
}

std::unique_ptr<jive::type>
fptype::copy() const
{
	return std::unique_ptr<jive::type>(new fptype(*this));
}

/* vararg type */

varargtype::~varargtype()
{}

bool
varargtype::operator==(const jive::type & other) const noexcept
{
	return dynamic_cast<const jlm::varargtype*>(&other) != nullptr;
}

std::string
varargtype::debug_string() const
{
	return "vararg";
}

std::unique_ptr<jive::type>
varargtype::copy() const
{
	return std::unique_ptr<jive::type>(new jlm::varargtype(*this));
}

/* struct type */

structtype::~structtype()
{}

bool
structtype::operator==(const jive::type & other) const noexcept
{
	auto type = dynamic_cast<const structtype*>(&other);
	return type
	    && type->packed_ == packed_
	    && type->name_ == name_
	    && type->declaration_ == declaration_;
}

std::string
structtype::debug_string() const
{
	return "struct";
}

std::unique_ptr<jive::type>
structtype::copy() const
{
	return std::unique_ptr<jive::type>(new structtype(*this));
}

/* vectortype */

vectortype::~vectortype()
{}

bool
vectortype::operator==(const jive::type & other) const noexcept
{
	auto type = dynamic_cast<const vectortype*>(&other);
	return type
	    && type->size_ == size_
	    && *type->type_ == *type_;
}

std::string
vectortype::debug_string() const
{
	return strfmt("vector[", type().debug_string(), ":", size(), "]");
}

std::unique_ptr<jive::type>
vectortype::copy() const
{
	return std::unique_ptr<jive::type>(new vectortype(*this));
}

/* loop state type */

loopstatetype::~loopstatetype()
{}

bool
loopstatetype::operator==(const jive::type & other) const noexcept
{
	return dynamic_cast<const loopstatetype*>(&other) != nullptr;
}

std::string
loopstatetype::debug_string() const
{
	return "loopstate";
}

std::unique_ptr<jive::type>
loopstatetype::copy() const
{
	return std::unique_ptr<jive::type>(new loopstatetype(*this));
}

/* I/O state type */

iostatetype::~iostatetype()
{}

bool
iostatetype::operator==(const jive::type & other) const noexcept
{
	return is<iostatetype>(other);
}

std::string
iostatetype::debug_string() const
{
	return "iostate";
}

std::unique_ptr<jive::type>
iostatetype::copy() const
{
	return std::unique_ptr<jive::type>(new iostatetype(*this));
}

}
