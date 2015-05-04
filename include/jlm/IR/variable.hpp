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

class variable {
public:
	virtual
	~variable() noexcept;

	inline
	variable(const jive::base::type & type, const std::string & name)
		: name_(name)
		, type_(type.copy())
	{}

	inline std::string
	debug_string() const
	{
		return name();
	}

	inline const std::string &
	name() const noexcept
	{
		return name_;
	}

	virtual const jive::base::type &
	type() const noexcept;

private:
	std::string name_;
	std::unique_ptr<jive::base::type> type_;
};

}

#endif
