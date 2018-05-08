/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/jlm/ir/aggregation/structure.hpp>

namespace jlm {
namespace agg {

/* structure */

structure::~structure()
{}

/* entry */

entry::~entry()
{}

std::string
entry::debug_string() const
{
	return "entry";
}

/* exit */

exit::~exit()
{}


std::string
exit::debug_string() const
{
	return "exit";
}

/* block */

block::~block()
{}

std::string
block::debug_string() const
{
	return "block";
}

/* linear */

linear::~linear()
{}

std::string
linear::debug_string() const
{
	return "linear";
}

/* branch */

branch::~branch()
{}

std::string
branch::debug_string() const
{
	return "branch";
}

/* loop */

loop::~loop()
{}

std::string
loop::debug_string() const
{
	return "loop";
}

}}
