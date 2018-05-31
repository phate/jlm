/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlm/ir/aggregation/node.hpp>

namespace jlm {

/* aggnode class */

aggnode::~aggnode()
{}

/* entryaggnode class */

entryaggnode::~entryaggnode()
{}

std::string
entryaggnode::debug_string() const
{
	return "entry";
}

/* exitaggnode class */

exitaggnode::~exitaggnode()
{}

std::string
exitaggnode::debug_string() const
{
	return "exit";
}

/* blockaggnode class */

blockaggnode::~blockaggnode()
{}

std::string
blockaggnode::debug_string() const
{
	return "block";
}

/* linearaggnode class */

linearaggnode::~linearaggnode()
{}

std::string
linearaggnode::debug_string() const
{
	return "linear";
}

/* branchaggnode class */

branchaggnode::~branchaggnode()
{}

std::string
branchaggnode::debug_string() const
{
	return "branch";
}

/* loopaggnode class */

loopaggnode::~loopaggnode()
{}

std::string
loopaggnode::debug_string() const
{
	return "loop";
}

}
