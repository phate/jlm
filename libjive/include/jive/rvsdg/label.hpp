/*
 * Copyright 2010 2011 2012 2013 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2011 2012 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_RVSDG_LABEL_HPP
#define JIVE_RVSDG_LABEL_HPP

#include <stdbool.h>
#include <stdint.h>

#include <jive/rvsdg/region.hpp>

namespace jive {
	class output;
}

struct jive_linker_symbol;

typedef uint64_t jive_offset;

typedef struct jive_address jive_address;
typedef struct jive_label jive_label;
typedef struct jive_label_class jive_label_class;
typedef struct jive_label_external jive_label_external;

struct jive_address {
	jive_offset offset;
	jive_stdsectionid section;
};

static inline void
jive_address_init(jive_address * self, jive_stdsectionid section, jive_offset offset)
{
	self->offset = offset;
	self->section = section;
}

typedef enum {
	jive_label_flags_none = 0,
	jive_label_flags_global = 1, /* whether label is supposed to be "globally visible" */
	jive_label_flags_external = 2, /* whether label must be resolved outside the current graph */
} jive_label_flags;

namespace jive {

/* label */

class label {
public:
	virtual
	~label();

protected:
	inline constexpr
	label()
	{}

public:
	label(const label &) = delete;

	label(label &&) = delete;

	virtual jive_label_flags
	flags() const noexcept;
};

/**
	\brief Special label marking position of "current" instruction
*/
class current_label final : public label {
public:
	virtual
	~current_label();

private:
	inline constexpr
	current_label()
	: label()
	{}

	current_label(const current_label &) = delete;

	current_label(current_label &&) = delete;

public:
	static inline const current_label *
	get()
	{
		static const current_label label;
		return &label;
	}
};

/**
	\brief Special label marking offset from frame pointer
*/
class fpoffset_label final : public label {
public:
	virtual
	~fpoffset_label();

private:
	inline constexpr
	fpoffset_label()
	: label()
	{}

	fpoffset_label(const fpoffset_label &) = delete;

	fpoffset_label(fpoffset_label &&) = delete;

public:
	static inline const fpoffset_label *
	get()
	{
		static const fpoffset_label label;
		return &label;
	}
};

/**
	\brief Special label marking offset from stack pointer
*/
class spoffset_label final : public label {
public:
	virtual
	~spoffset_label();

private:
	inline constexpr
	spoffset_label()
	: label()
	{}

	spoffset_label(const spoffset_label &) = delete;

	spoffset_label(spoffset_label &&) = delete;

public:
	static inline const spoffset_label *
	get()
	{
		static const spoffset_label label;
		return &label;
	}
};

/* external label */

class external_label final : public label {
public:
	virtual
	~external_label();

	inline
	external_label(
		const std::string & asmname,
		const struct jive_linker_symbol * symbol)
	: asmname_(asmname)
	, symbol_(symbol)
	{}

	inline const std::string &
	asmname() const noexcept
	{
		return asmname_;
	}

	inline const struct jive_linker_symbol *
	symbol() const noexcept
	{
		return symbol_;
	}

	virtual jive_label_flags
	flags() const noexcept override;

private:
	std::string asmname_;
	const struct jive_linker_symbol * symbol_;
};

}

#endif
