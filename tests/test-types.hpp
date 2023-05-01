/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef TEST_TEST_TYPES_HPP
#define TEST_TEST_TYPES_HPP

#include <jlm/rvsdg/type.hpp>

namespace jlm {

class valuetype final : public jive::valuetype {
public:
	virtual
	~valuetype();

	inline constexpr
	valuetype() noexcept
	: jive::valuetype()
	{}

	virtual std::string
	debug_string() const override;

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;
};

class statetype final : public jive::statetype {
public:
	virtual
	~statetype();

	inline constexpr
	statetype() noexcept
	: jive::statetype()
	{}

	virtual std::string
	debug_string() const override;

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;
};

}

#endif
