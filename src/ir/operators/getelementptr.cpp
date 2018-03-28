/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators/getelementptr.hpp>

namespace jlm {

/* getelementptr operator */

getelementptr_op::~getelementptr_op()
{}

bool
getelementptr_op::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const jlm::getelementptr_op*>(&other);
	return op
	    && op->pport_ == pport_
	    && op->rport_ == rport_
	    && op->bports_ == bports_;
}

size_t
getelementptr_op::narguments() const noexcept
{
	return 1 + nindices();
}

const jive::port &
getelementptr_op::argument(size_t index) const noexcept
{
	JLM_DEBUG_ASSERT(index < narguments());
	if (index == 0)
		return pport_;

	return bports_[index-1];
}

size_t
getelementptr_op::nresults() const noexcept
{
	return 1;
}

const jive::port &
getelementptr_op::result(size_t index) const noexcept
{
	return rport_;
}

std::string
getelementptr_op::debug_string() const
{
	return "GETELEMENTPTR";
}

std::unique_ptr<jive::operation>
getelementptr_op::copy() const
{
	return std::unique_ptr<jive::operation>(new getelementptr_op(*this));
}

}
