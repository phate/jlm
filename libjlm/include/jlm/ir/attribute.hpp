/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_ATTRIBUTE_HPP
#define JLM_IR_ATTRIBUTE_HPP

#include <jlm/common.hpp>

#include <jive/rvsdg/type.hpp>

#include <memory>
#include <string>
#include <vector>

namespace jlm {

/** \brief Attribute
*/
class attribute {
public:
	enum class kind {
    None ///< No attributes have been set
	, alignment
	, alloc_size
	, always_inline
	, arg_mem_only
	, builtin
	, by_val
	, cold
	, convergent
	, dereferenceable
	, dereferenceable_or_null
	, imm_arg
	, in_alloca
	, in_reg
	, inaccessible_mem_only
	, inaccessible_mem_or_arg_mem_only
	, inline_hint
	, jump_table
	, min_size
	, naked
	, nest
	, no_alias
	, no_builtin
	, no_capture
	, no_cf_check
	, no_duplicate
	, no_free
	, no_implicit_float
	, no_inline
  , NoMerge
	, no_recurse
	, no_red_zone
	, no_return
	, no_sync
  , NoUndef
	, no_unwind
	, non_lazy_bind
	, non_null
  , NullPointerIsValid
	, opt_for_fuzzing
	, optimize_for_size
	, optimize_none
  , Preallocated
	, read_none
	, read_only
	, returned
	, returns_twice
	, sext
	, safe_stack
	, sanitize_address
	, sanitize_hwaddress
	, sanitize_mem_tag
	, sanitize_memory
	, sanitize_thread
	, shadow_call_stack
	, speculatable
	, speculative_load_hardening
	, stack_alignment
	, stack_protect
	, stack_protect_req
	, stack_protect_strong
	, strict_fp
	, struct_ret
	, swift_error
	, swift_self
	, uwtable
	, will_return
	, write_only
	, zext
  , EndAttrKinds ///< Sentinel value useful for loops
	};

	virtual
	~attribute();

	attribute() = default;

	attribute(const attribute&) = delete;

	attribute(attribute&&) = delete;

	attribute&
	operator=(const attribute&) = delete;

	attribute&
	operator=(attribute&&) = delete;

	virtual bool
	operator==(const attribute&) const = 0;

	virtual bool
	operator!=(const attribute & other) const
	{
		return !operator==(other);
	}

	virtual std::unique_ptr<attribute>
	copy() const = 0;
};

/** \brief String attribute
*/
class string_attribute final : public attribute {
public:
	~string_attribute() override;

private:
	string_attribute(
		const std::string & kind,
		const std::string & value)
	: kind_(kind)
	, value_(value)
	{}

public:
	const std::string &
	kind() const noexcept
	{
		return kind_;
	}

	const std::string &
	value() const noexcept
	{
		return value_;
	}

	virtual bool
	operator==(const attribute&) const override;

	virtual std::unique_ptr<attribute>
	copy() const override;

	static std::unique_ptr<attribute>
	create(
		const std::string & kind,
		const std::string & value)
	{
		return std::unique_ptr<attribute>(new string_attribute(kind, value));
	}

private:
	std::string kind_;
	std::string value_;
};

/** \brief Enum attribute
*/
class enum_attribute : public attribute {
public:
	~enum_attribute() override;

protected:
	enum_attribute(const attribute::kind & kind)
	: kind_(kind)
	{}

public:
	const attribute::kind &
	kind() const noexcept
	{
		return kind_;
	}

	virtual bool
	operator==(const attribute&) const override;

	virtual std::unique_ptr<attribute>
	copy() const override;

	static std::unique_ptr<attribute>
	create(const attribute::kind & kind)
	{
		return std::unique_ptr<attribute>(new enum_attribute(kind));
	}

private:
	attribute::kind kind_;
};

/** \brief Integer attribute
*/
class int_attribute final : public enum_attribute {
public:
	~int_attribute() override;

private:
	int_attribute(
		attribute::kind kind,
		uint64_t value)
	: enum_attribute(kind)
	, value_(value)
	{}

public:
	uint64_t
	value() const noexcept
	{
		return value_;
	}

	virtual bool
	operator==(const attribute&) const override;

	virtual std::unique_ptr<attribute>
	copy() const override;

	static std::unique_ptr<attribute>
	create(
		const attribute::kind & kind,
		uint64_t value)
	{
		return std::unique_ptr<attribute>(new int_attribute(kind, value));
	}

private:
	uint64_t value_;
};

/** \brief Type attribute
*/
class type_attribute final : public enum_attribute {
public:
	~type_attribute() override;

private:
	type_attribute(
		attribute::kind kind,
		std::unique_ptr<jive::valuetype> type)
	: enum_attribute(kind)
	, type_(std::move(type))
	{}

	type_attribute(
		attribute::kind kind,
		const jive::valuetype & type)
	: enum_attribute(kind)
	, type_(static_cast<jive::valuetype*>(type.copy().release()))
	{}

public:
	const jive::valuetype &
	type() const noexcept
	{
		return *type_;
	}

	virtual bool
	operator==(const attribute&) const override;

	virtual std::unique_ptr<attribute>
	copy() const override;

	static std::unique_ptr<attribute>
	create_byval(std::unique_ptr<jive::valuetype> type)
	{
		std::unique_ptr<type_attribute> ta(new type_attribute(kind::by_val, std::move(type)));
		return ta;
	}

private:
	std::unique_ptr<jive::valuetype> type_;
};

/** \brief Attribute set
*/
class attributeset final {
	class constiterator;

public:
	~attributeset()
	{}

	attributeset() = default;

	attributeset(std::vector<std::unique_ptr<attribute>> attributes)
	: attributes_(std::move(attributes))
	{}

	attributeset(const attributeset & other)
	{
		*this = other;
	}

	attributeset(attributeset && other)
	: attributes_(std::move(other.attributes_))
	{}

	attributeset &
	operator=(const attributeset & other);

	attributeset &
	operator=(attributeset && other)
	{
		if (this == &other)
			return *this;

		attributes_ = std::move(other.attributes_);

		return *this;
	}

	constiterator
	begin() const;

	constiterator
	end() const;

	void
	insert(const attribute & a)
	{
		attributes_.push_back(a.copy());
	}

	void
	insert(std::unique_ptr<attribute> a)
	{
		attributes_.push_back(std::move(a));
	}

	bool
	operator==(const attributeset & other) const noexcept
	{
		/*
			FIXME: Ah, since this is not a real set, we cannot cheaply implement a comparison.
		*/
		return false;
	}

	bool
	operator!=(const attributeset & other) const noexcept
	{
		return !(*this == other);
	}

private:
	/*
		FIXME: Implement a proper set. Elements are not unique here.
	*/
	std::vector<std::unique_ptr<attribute>> attributes_;
};


/** \brief Attribute set const iterator
*/
class attributeset::constiterator final : public std::iterator<std::forward_iterator_tag,
	const attribute*, ptrdiff_t> {
	friend ::jlm::attributeset;

private:
	constiterator(const std::vector<std::unique_ptr<attribute>>::const_iterator & it)
	: it_(it)
	{}

public:
	const attribute *
	operator->() const
	{
		return it_->get();
	}

	const attribute &
	operator*()
	{
		return *operator->();
	}

	constiterator &
	operator++()
	{
		it_++;
		return *this;
	}

	constiterator
	operator++(int)
	{
		constiterator tmp = *this;
		++*this;
		return tmp;
	}

	bool
	operator==(const constiterator & other) const
	{
		return it_ == other.it_;
	}

	bool
	operator!=(const constiterator & other) const
	{
		return !operator==(other);
	}

private:
	std::vector<std::unique_ptr<attribute>>::const_iterator it_;
};

}

#endif
