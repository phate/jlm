/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_TYPE_HPP
#define JLM_RVSDG_TYPE_HPP

#include <memory>
#include <string>

namespace jive {

class type {
public:
	virtual
	~type() noexcept;

protected:
	inline constexpr
	type() noexcept
	{}

public:
	virtual bool
	operator==(const jive::type & other) const noexcept = 0;

	inline bool
	operator!=(const jive::type & other) const noexcept
	{
		return !(*this == other);
	}

	virtual std::unique_ptr<type>
	copy() const = 0;

	virtual std::string
	debug_string() const = 0;
};

class valuetype : public jive::type {
public:
	virtual
	~valuetype() noexcept;

protected:
	inline constexpr
	valuetype() noexcept
	: jive::type()
	{}
};

class statetype : public jive::type {
public:
	virtual
	~statetype() noexcept;

protected:
	inline constexpr
	statetype() noexcept
	: jive::type()
	{}
};

template <class T> static inline bool
is(const jive::type & type) noexcept
{
	static_assert(std::is_base_of<jive::type, T>::value,
		"Template parameter T must be derived from jive::type.");

	return dynamic_cast<const T*>(&type) != nullptr;
}

}

#endif
