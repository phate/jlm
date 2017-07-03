/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_TYPES_HPP
#define JLM_IR_TYPES_HPP

#include <jive/vsdg/basetype.h>

namespace jlm {

/* pointer type */

class ptrtype final : public jive::value::type {
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

/* array type */

class arraytype final : public jive::value::type {
public:
	virtual
	~arraytype();

	inline
	arraytype(const jive::value::type & type, size_t nelements)
	: jive::value::type()
	, nelements_(nelements)
	, type_(type.copy())
	{
		if (nelements == 0)
			throw std::logic_error("Expected at least one element.");
	}

	inline
	arraytype(const jlm::arraytype & other)
	: jive::value::type(other)
	, nelements_(other.nelements_)
	, type_(std::move(other.type_->copy()))
	{}

	inline
	arraytype(jlm::arraytype && other)
	: jive::value::type(other)
	, nelements_(other.nelements_)
	, type_(std::move(other.type_))
	{}

	inline arraytype &
	operator=(const jlm::arraytype &) = delete;

	inline arraytype &
	operator=(jlm::arraytype &&) = delete;

	virtual std::string
	debug_string() const override;

	virtual bool
	operator==(const jive::base::type & other) const noexcept override;

	virtual std::unique_ptr<jive::base::type>
	copy() const override;

	inline size_t
	nelements() const noexcept
	{
		return nelements_;
	}

	inline const jive::value::type &
	element_type() const noexcept
	{
		return *static_cast<const jive::value::type*>(type_.get());
	}

private:
	size_t nelements_;
	std::unique_ptr<jive::base::type> type_;
};

static inline bool
is_arraytype(const jive::base::type & type)
{
	return dynamic_cast<const jlm::arraytype*>(&type) != nullptr;
}

static inline std::unique_ptr<jive::base::type>
create_arraytype(const jive::base::type & type, size_t nelements)
{
	auto vt = dynamic_cast<const jive::value::type*>(&type);
	if (!vt) throw std::logic_error("Expected value type.");

	return std::unique_ptr<jive::base::type>(new arraytype(*vt, nelements));
}

/* floating point type */

enum class fpsize {half, flt, dbl};

class fptype final : public jive::value::type {
public:
	virtual
	~fptype();

	inline
	fptype(const fpsize & size)
	: jive::value::type()
	, size_(size)
	{}

	virtual std::string
	debug_string() const override;

	virtual bool
	operator==(const jive::base::type & other) const noexcept override;

	virtual std::unique_ptr<jive::base::type>
	copy() const override;

	inline const jlm::fpsize &
	size() const noexcept
	{
		return size_;
	}

private:
	jlm::fpsize size_;
};

}

#endif
