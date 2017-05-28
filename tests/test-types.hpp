/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef TEST_TEST_TYPES_HPP
#define TEST_TEST_TYPES_HPP

#include <jive/vsdg/basetype.h>

namespace jlm {

class valuetype final : public jive::value::type {
public:
	virtual
	~valuetype() noexcept;

	inline constexpr
	valuetype() noexcept
	: jive::value::type()
	{}

	virtual std::string
	debug_string() const override;

	virtual bool
	operator==(const jive::base::type & other) const noexcept override;

	virtual std::unique_ptr<jive::base::type>
	copy() const override;
};

class statetype final : public jive::state::type {
public:
	virtual
	~statetype() noexcept;

	inline constexpr
	statetype() noexcept
	: jive::state::type()
	{}

	virtual std::string
	debug_string() const override;

	virtual bool
	operator==(const jive::base::type & other) const noexcept override;

	virtual std::unique_ptr<jive::base::type>
	copy() const override;
};

}

#endif
