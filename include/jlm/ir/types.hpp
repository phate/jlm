/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_TYPES_HPP
#define JLM_IR_TYPES_HPP

#include <jive/vsdg/basetype.h>

namespace jlm {

class ptrtype : public jive::value::type {
public:
	virtual
	~ptrtype();

	inline
	ptrtype(const jive::value::type & ptype)
	: jive::value::type()
	, ptype_(ptype.copy())
	{}

	inline
	ptrtype(const jlm::ptrtype & other)
	: jive::value::type(other)
	, ptype_(std::move(other.ptype_->copy()))
	{}

	inline
	ptrtype(jlm::ptrtype && other)
	: jive::value::type(other)
	, ptype_(std::move(other.ptype_))
	{}

	inline ptrtype &
	operator=(const jlm::ptrtype & other) = delete;

	inline ptrtype &
	operator=(jlm::ptrtype &&) = delete;

	virtual std::string
	debug_string() const override;

	virtual bool
	operator==(const jive::base::type & other) const noexcept override;

	virtual std::unique_ptr<jive::base::type>
	copy() const override;

	inline const jive::value::type &
	pointee_type() const noexcept
	{
		return *static_cast<const jive::value::type*>(ptype_.get());
	}

private:
	std::unique_ptr<jive::base::type> ptype_;
};

static inline bool
is_ptrtype(const jive::base::type & type)
{
	return dynamic_cast<const jlm::ptrtype*>(&type) != nullptr;
}

static inline std::unique_ptr<jive::base::type>
create_ptrtype(const jive::base::type & vtype)
{
	auto vt = dynamic_cast<const jive::value::type*>(&vtype);
	if (!vt) throw std::logic_error("Expected value type.");

	return std::unique_ptr<jive::base::type>(new jlm::ptrtype(*vt));
}

}

#endif
