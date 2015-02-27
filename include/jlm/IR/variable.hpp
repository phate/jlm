/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_VARIABLE_HPP
#define JLM_IR_VARIABLE_HPP

#include <jive/vsdg/basetype.h>

#include <memory>
#include <sstream>

namespace jlm {
namespace frontend {

class variable final {
public:
	inline
	variable(const jive::base::type & type, const std::string & name)
		: name_(name)
		, type_(type.copy())
	{}

	inline std::string
	debug_string() const
	{
		if (!name_.empty())
			return name_;

		std::stringstream sstrm;
		sstrm << "v" << this;

		return sstrm.str();
	}

	inline const std::string &
	name() const noexcept
	{
		return name_;
	}

	inline const jive::base::type &
	type() const noexcept
	{
		return *type_;
	}

	bool
	operator==(const variable & other) const noexcept
	{
		return name_ == other.name_ && *type_ == *other.type_;
	}

	bool
	operator!=(const variable & other) const noexcept
	{
		return !(*this == other);
	}

private:
	std::string name_;
	std::unique_ptr<jive::base::type> type_;
};

}
}

#endif
