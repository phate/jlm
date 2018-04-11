/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_TYPES_HPP
#define JLM_IR_TYPES_HPP

#include <jive/types/function/fcttype.h>
#include <jive/types/record.h>
#include <jive/rvsdg/type.h>

#include <vector>

namespace jlm {

/* pointer type */

class ptrtype final : public jive::valuetype {
public:
	virtual
	~ptrtype();

	inline
	ptrtype(const jive::valuetype & ptype)
	: jive::valuetype()
	, ptype_(ptype.copy())
	{}

	inline
	ptrtype(const jlm::ptrtype & other)
	: jive::valuetype(other)
	, ptype_(std::move(other.ptype_->copy()))
	{}

	inline
	ptrtype(jlm::ptrtype && other)
	: jive::valuetype(other)
	, ptype_(std::move(other.ptype_))
	{}

	inline ptrtype &
	operator=(const jlm::ptrtype & other) = delete;

	inline ptrtype &
	operator=(jlm::ptrtype &&) = delete;

	virtual std::string
	debug_string() const override;

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;

	inline const jive::valuetype &
	pointee_type() const noexcept
	{
		return *static_cast<const jive::valuetype*>(ptype_.get());
	}

private:
	std::unique_ptr<jive::type> ptype_;
};

static inline bool
is_ptrtype(const jive::type & type)
{
	return dynamic_cast<const jlm::ptrtype*>(&type) != nullptr;
}

static inline std::unique_ptr<jive::type>
create_ptrtype(const jive::type & vtype)
{
	auto vt = dynamic_cast<const jive::valuetype*>(&vtype);
	if (!vt) throw std::logic_error("Expected value type.");

	return std::unique_ptr<jive::type>(new jlm::ptrtype(*vt));
}

/* array type */

class arraytype final : public jive::valuetype {
public:
	virtual
	~arraytype();

	inline
	arraytype(const jive::valuetype & type, size_t nelements)
	: jive::valuetype()
	, nelements_(nelements)
	, type_(type.copy())
	{
		if (nelements == 0)
			throw std::logic_error("Expected at least one element.");
	}

	inline
	arraytype(const jlm::arraytype & other)
	: jive::valuetype(other)
	, nelements_(other.nelements_)
	, type_(std::move(other.type_->copy()))
	{}

	inline
	arraytype(jlm::arraytype && other)
	: jive::valuetype(other)
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
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;

	inline size_t
	nelements() const noexcept
	{
		return nelements_;
	}

	inline const jive::valuetype &
	element_type() const noexcept
	{
		return *static_cast<const jive::valuetype*>(type_.get());
	}

private:
	size_t nelements_;
	std::unique_ptr<jive::type> type_;
};

static inline bool
is_arraytype(const jive::type & type)
{
	return dynamic_cast<const jlm::arraytype*>(&type) != nullptr;
}

static inline std::unique_ptr<jive::type>
create_arraytype(const jive::type & type, size_t nelements)
{
	auto vt = dynamic_cast<const jive::valuetype*>(&type);
	if (!vt) throw std::logic_error("Expected value type.");

	return std::unique_ptr<jive::type>(new arraytype(*vt, nelements));
}

/* floating point type */

enum class fpsize {half, flt, dbl, x86fp80};

class fptype final : public jive::valuetype {
public:
	virtual
	~fptype();

	inline
	fptype(const fpsize & size)
	: jive::valuetype()
	, size_(size)
	{}

	virtual std::string
	debug_string() const override;

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;

	inline const jlm::fpsize &
	size() const noexcept
	{
		return size_;
	}

private:
	jlm::fpsize size_;
};

/* vararg type */

class varargtype final : public jive::statetype {
public:
	virtual
	~varargtype();

	inline constexpr
	varargtype()
	: jive::statetype()
	{}

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;

	virtual std::string
	debug_string() const override;
};

static inline bool
is_varargtype(const jive::type & type)
{
	return dynamic_cast<const jlm::varargtype*>(&type) != nullptr;
}

static inline std::unique_ptr<jive::type>
create_varargtype()
{
	return std::unique_ptr<jive::type>(new varargtype());
}

/* struct type */

class structtype final : public jive::valuetype {
public:
	virtual
	~structtype();

	inline
	structtype(
		bool packed,
		const std::shared_ptr<const jive::rcddeclaration> & declaration)
	: jive::valuetype()
	, packed_(packed)
	, declaration_(declaration)
	{}

	inline
	structtype(
		const std::string & name,
		bool packed,
		const std::shared_ptr<const jive::rcddeclaration> & declaration)
	: jive::valuetype()
	, packed_(packed)
	, name_(name)
	, declaration_(declaration)
	{}

	structtype(const structtype &) = default;

	structtype(structtype &&) = delete;

	structtype &
	operator=(const structtype &) = delete;

	structtype &
	operator=(structtype &&) = delete;

	inline bool
	has_name() const noexcept
	{
		return !name_.empty();
	}

	inline const std::string &
	name() const noexcept
	{
		return name_;
	}

	inline bool
	packed() const noexcept
	{
		return packed_;
	}

	inline const std::shared_ptr<const jive::rcddeclaration>
	declaration() const noexcept
	{
		return declaration_;
	}

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;

	virtual std::string
	debug_string() const override;

private:
	bool packed_;
	std::string name_;
	std::shared_ptr<const jive::rcddeclaration> declaration_;
};

/* function type */

/* FIXME: this belongs into jive */

static inline bool
is_fcttype(const jive::type & type)
{
	return dynamic_cast<const jive::fct::type*>(&type) != nullptr;
}

}

#endif
