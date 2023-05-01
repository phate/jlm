/*
 * Copyright 2011 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_BITSTRING_TYPE_HPP
#define JLM_RVSDG_BITSTRING_TYPE_HPP

#include <jlm/rvsdg/type.hpp>
#include <jlm/util/common.hpp>

namespace jive {

/* bitstring type */

class bittype final : public jive::valuetype {
public:
	virtual
	~bittype() noexcept;

	inline constexpr
	bittype(size_t nbits)
	: nbits_(nbits)
	{
		if (nbits == 0)
			throw compiler_error("Number of bits must be greater than zero.");
	}

	inline size_t
	nbits() const noexcept
	{
		return nbits_;
	}

	virtual std::string
	debug_string() const override;

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;

private:
	size_t nbits_;
};

extern const bittype bit1;
extern const bittype bit8;
extern const bittype bit16;
extern const bittype bit32;
extern const bittype bit64;

}

#endif
