/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_ATTRIBUTE_HPP
#define JLM_LLVM_IR_ATTRIBUTE_HPP

#include <jlm/rvsdg/type.hpp>
#include <jlm/util/common.hpp>

#include <memory>
#include <string>
#include <vector>

namespace jlm {

/** \brief Attribute
*/
class attribute {
public:
	enum class kind {
    None, ///< No attributes have been set

    FirstEnumAttr,
    AllocAlign,
    AllocatedPointer,
    AlwaysInline,
    ArgMemOnly,
    Builtin,
    Cold,
    Convergent,
    DisableSanitizerInstrumentation,
    FnRetThunkExtern,
    Hot,
    ImmArg,
    InReg,
    InaccessibleMemOnly,
    InaccessibleMemOrArgMemOnly,
    InlineHint,
    JumpTable,
    MinSize,
    MustProgress,
    Naked,
    Nest,
    NoAlias,
    NoBuiltin,
    NoCallback,
    NoCapture,
    NoCfCheck,
    NoDuplicate,
    NoFree,
    NoImplicitFloat,
    NoInline,
    NoMerge,
    NoProfile,
    NoRecurse,
    NoRedZone,
    NoReturn,
    NoSanitizeBounds,
    NoSanitizeCoverage,
    NoSync,
    NoUndef,
    NoUnwind,
    NonLazyBind,
    NonNull,
    NullPointerIsValid,
    OptForFuzzing,
    OptimizeForSize,
    OptimizeNone,
    PresplitCoroutine,
    ReadNone,
    ReadOnly,
    Returned,
    ReturnsTwice,
    SExt,
    SafeStack,
    SanitizeAddress,
    SanitizeHWAddress,
    SanitizeMemTag,
    SanitizeMemory,
    SanitizeThread,
    ShadowCallStack,
    Speculatable,
    SpeculativeLoadHardening,
    StackProtect,
    StackProtectReq,
    StackProtectStrong,
    StrictFP,
    SwiftAsync,
    SwiftError,
    SwiftSelf,
    WillReturn,
    WriteOnly,
    ZExt,
    LastEnumAttr,

    FirstTypeAttr,
    ByRef,
    ByVal,
    ElementType,
    InAlloca,
    Preallocated,
    StructRet,
    LastTypeAttr,

    FirstIntAttr,
    Alignment,
    AllocKind,
    AllocSize,
    Dereferenceable,
    DereferenceableOrNull,
    StackAlignment,
    UWTable,
    VScaleRange,
    LastIntAttr,

    EndAttrKinds ///< Sentinel value useful for loops
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
		std::unique_ptr<type_attribute> ta(new type_attribute(kind::ByVal, std::move(type)));
		return ta;
	}

  static std::unique_ptr<attribute>
  CreateStructRetAttribute(std::unique_ptr<jive::valuetype> type)
  {
    return std::unique_ptr<attribute>(new type_attribute(kind::StructRet, std::move(type)));
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
class attributeset::constiterator final
{
public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = const attribute*;
  using difference_type = std::ptrdiff_t;
  using pointer = const attribute**;
  using reference = const attribute*&;

private:
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
