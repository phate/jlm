/*
 * Copyright 2019 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm {

/* impport class */

impport::~impport()
{}

bool
impport::operator==(const port & other) const noexcept
{
	auto p = dynamic_cast<const impport*>(&other);
	return p
	    && p->type() == type()
	    && p->name() == name()
	    && p->linkage() == linkage();
}

std::unique_ptr<jive::port>
impport::copy() const
{
	return std::unique_ptr<port>(new impport(*this));
}

}
