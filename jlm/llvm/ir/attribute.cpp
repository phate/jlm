/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/attribute.hpp>

namespace jlm {

/* attribute class */

attribute::~attribute()
{}

/* string attribute class */

string_attribute::~string_attribute()
{}

bool
string_attribute::operator==(const attribute & other) const
{
	auto sa = dynamic_cast<const string_attribute*>(&other);
	return sa
	    && sa->kind() == kind()
	    && sa->value() == value();
}

std::unique_ptr<attribute>
string_attribute::copy() const
{
	return std::unique_ptr<attribute>(new string_attribute(kind(), value()));
}

/* enum attribute class */

enum_attribute::~enum_attribute()
{}

bool
enum_attribute::operator==(const attribute & other) const
{
	auto ea = dynamic_cast<const enum_attribute*>(&other);
	return ea
	    && ea->kind() == kind();
}

std::unique_ptr<attribute>
enum_attribute::copy() const
{
	return std::unique_ptr<attribute>(new enum_attribute(kind()));
}

/* integer attribute class */

int_attribute::~int_attribute()
{}

bool
int_attribute::operator==(const attribute & other) const
{
	auto ia = dynamic_cast<const int_attribute*>(&other);
	return ia
	    && ia->kind() == kind()
	    && ia->value() == value();
}

std::unique_ptr<attribute>
int_attribute::copy() const
{
	return std::unique_ptr<attribute>(new int_attribute(kind(), value()));
}

/* type attribute class */

type_attribute::~type_attribute()
{}

bool
type_attribute::operator==(const attribute & other) const
{
	auto ta = dynamic_cast<const type_attribute*>(&other);
	return ta
	    && ta->kind() == kind()
	    && ta->type() == type();
}

std::unique_ptr<attribute>
type_attribute::copy() const
{
	return std::unique_ptr<attribute>(new type_attribute(kind(), type()));
}

/* attribute set class */

attributeset &
attributeset::operator=(const attributeset & other)
{
	if (this == &other)
		return *this;

	attributes_.clear();
	for (auto & attribute : other)
		attributes_.push_back(attribute.copy());

	return *this;
}

attributeset::constiterator
attributeset::begin() const
{
	return constiterator(attributes_.begin());
}

attributeset::constiterator
attributeset::end() const
{
	return constiterator(attributes_.end());
}

}
