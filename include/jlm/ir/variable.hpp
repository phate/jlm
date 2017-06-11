/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_VARIABLE_HPP
#define JLM_IR_VARIABLE_HPP

#include <jive/vsdg/basetype.h>

#include <jlm/util/strfmt.hpp>

#include <memory>
#include <sstream>

namespace jlm {

class variable {
public:
	virtual
	~variable() noexcept;

	inline
	variable(const jive::base::type & type, const std::string & name, bool exported)
		: exported_(exported)
		, name_(name)
		, type_(type.copy())
	{}

	virtual std::string
	debug_string() const;

	inline const std::string &
	name() const noexcept
	{
		return name_;
	}

	inline bool
	exported() const noexcept
	{
		return exported_;
	}

	virtual const jive::base::type &
	type() const noexcept;

private:
	bool exported_;
	std::string name_;
	std::unique_ptr<jive::base::type> type_;
};

class global_variable : public variable {
public:
	virtual
	~global_variable() noexcept;

	inline
	global_variable(const jive::base::type & type, const std::string & name, bool exported)
		: variable(type, name, exported)
	{}
};

}

#endif
