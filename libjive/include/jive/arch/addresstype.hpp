/*
 * Copyright 2011 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_ARCH_ADDRESSTYPE_HPP
#define JIVE_ARCH_ADDRESSTYPE_HPP

#include <jive/rvsdg/type.hpp>

namespace jive {

class memtype final : public jive::statetype {
public:
	virtual ~memtype() noexcept;

	inline constexpr
	memtype() noexcept
	: jive::statetype()
	{}

	virtual std::string debug_string() const override;

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;

	inline static const memtype & instance() { return instance_; }

private:
	static const memtype instance_;
};

}

#endif
