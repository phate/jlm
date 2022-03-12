/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_TYPES_HPP
#define JLM_IR_TYPES_HPP

#include <jive/types/record.hpp>
#include <jive/rvsdg/type.hpp>

#include <jlm/common.hpp>
#include <jlm/util/iterator_range.hpp>

#include <vector>

namespace jlm {

/** \brief Function type class
 *
 */
class FunctionType final : public jive::valuetype {

  class TypeConstIterator final : public std::iterator<std::forward_iterator_tag, jive::type*, ptrdiff_t> {
  public:
    explicit
    TypeConstIterator(const std::vector<std::unique_ptr<jive::type>>::const_iterator & it)
      : It_(it)
    {}

  public:
    jive::type *
    type() const noexcept
    {
      return It_->get();
    }

    jive::type &
    operator*() const
    {
      JLM_ASSERT(type() != nullptr);
      return *type();
    }

    jive::type *
    operator->() const
    {
      return type();
    }

    TypeConstIterator &
    operator++()
    {
      ++It_;
      return *this;
    }

    TypeConstIterator
    operator++(int)
    {
      TypeConstIterator tmp = *this;
      ++*this;
      return tmp;
    }

    bool
    operator==(const TypeConstIterator & other) const
    {
      return It_ == other.It_;
    }

    bool
    operator!=(const TypeConstIterator & other) const
    {
      return !operator==(other);
    }

  private:
    std::vector<std::unique_ptr<jive::type>>::const_iterator It_;
  };

  using ArgumentConstRange = iterator_range<TypeConstIterator>;
  using ResultConstRange = iterator_range<TypeConstIterator>;

public:
  ~FunctionType() noexcept override;

  FunctionType(
    const std::vector<const jive::type*> & argumentTypes,
    const std::vector<const jive::type*> & resultTypes);

  FunctionType(
    std::vector<std::unique_ptr<jive::type>> argumentTypes,
    std::vector<std::unique_ptr<jive::type>> resultTypes);

  FunctionType(const FunctionType & other);

  FunctionType(FunctionType && other) noexcept;

  FunctionType &
  operator=(const FunctionType & other);

  FunctionType &
  operator=(FunctionType && other) noexcept;

  ArgumentConstRange
  Arguments() const;

  ResultConstRange
  Results() const;

  size_t
  NumResults() const noexcept
  {
    return ResultTypes_.size();
  }

  size_t
  NumArguments() const noexcept
  {
    return ArgumentTypes_.size();
  }

  const jive::type &
  ResultType(size_t index) const noexcept
  {
    JLM_ASSERT(index < ResultTypes_.size());
    return *ResultTypes_[index];
  }

  const jive::type &
  ArgumentType(size_t index) const noexcept
  {
    JLM_ASSERT(index < ArgumentTypes_.size());
    return *ArgumentTypes_[index];
  }

  std::string
  debug_string() const override;

  bool
  operator==(const jive::type & other) const noexcept override;

  std::unique_ptr<jive::type>
  copy() const override;

private:
  std::vector<std::unique_ptr<jive::type>> ResultTypes_;
  std::vector<std::unique_ptr<jive::type>> ArgumentTypes_;
};

/** \brief PointerType class
 *
 * This operator is the Jlm equivalent of LLVM's PointerType class.
 */
class PointerType final : public jive::valuetype {
public:
  ~PointerType() noexcept override;

  explicit
  PointerType(const jive::valuetype & elementType)
    : jive::valuetype()
    , ElementType_(elementType.copy())
  {}

  PointerType(const PointerType & other)
    : jive::valuetype(other)
    , ElementType_(other.ElementType_->copy())
  {}

  PointerType(PointerType && other) noexcept
    : jive::valuetype(other)
    , ElementType_(std::move(other.ElementType_))
  {}

  PointerType &
  operator=(const PointerType&) = delete;

  PointerType &
  operator=(PointerType&&) = delete;

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const jive::type & other) const noexcept override;

  [[nodiscard]] std::unique_ptr<jive::type>
  copy() const override;

  [[nodiscard]] const jive::valuetype &
  GetElementType() const noexcept
  {
    return *AssertedCast<const jive::valuetype>(ElementType_.get());
  }

  static std::unique_ptr<jive::type>
  Create(const jive::type & type)
  {
    auto & valueType = CheckAndExtractType(type);
    return std::unique_ptr<jive::type>(new PointerType(valueType));
  }

private:
  static const jive::valuetype &
  CheckAndExtractType(const jive::type & type)
  {
    if (auto valueType = dynamic_cast<const jive::valuetype*>(&type))
      return *valueType;

    throw error("Expected value type.");
  }

  std::unique_ptr<jive::type> ElementType_;
};

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
	{}

	inline
	arraytype(const jlm::arraytype & other)
	: jive::valuetype(other)
	, nelements_(other.nelements_)
	, type_(other.type_->copy())
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

static inline std::unique_ptr<jive::type>
create_arraytype(const jive::type & type, size_t nelements)
{
	auto vt = dynamic_cast<const jive::valuetype*>(&type);
	if (!vt) throw jlm::error("expected value type.");

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
		const jive::rcddeclaration * dcl)
	: jive::valuetype()
	, packed_(packed)
	, declaration_(dcl)
	{}

	inline
	structtype(
		const std::string & name,
		bool packed,
		const jive::rcddeclaration * dcl)
	: jive::valuetype()
	, packed_(packed)
	, name_(name)
	, declaration_(dcl)
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

	inline const jive::rcddeclaration *
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
	const jive::rcddeclaration * declaration_;
};

/* vector type */

class vectortype : public jive::valuetype {
public:
	vectortype(
		const jive::valuetype & type,
		size_t size)
	: size_(size)
	, type_(type.copy())
	{}

	vectortype(const vectortype & other)
	: valuetype(other)
	, size_(other.size_)
	, type_(other.type_->copy())
	{}

	vectortype(vectortype && other)
	: valuetype(other)
	, size_(other.size_)
	, type_(std::move(other.type_))
	{}

	vectortype &
	operator=(const vectortype & other)
	{
		if (this == &other)
			return *this;

		size_ = other.size_;
		type_ = other.type_->copy();
		return *this;
	}

	vectortype &
	operator=(vectortype && other)
	{
		if (this == &other)
			return *this;

		size_ = other.size_;
		type_ = std::move(other.type_);
		return *this;
	}

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	size_t
	size() const noexcept
	{
		return size_;
	}

	const jive::valuetype &
	type() const noexcept
	{
		return *static_cast<const jive::valuetype*>(type_.get());
	}

private:
	size_t size_;
	std::unique_ptr<jive::type> type_;
};

class fixedvectortype final : public vectortype {
public:
	~fixedvectortype() override;

	fixedvectortype(
		const jive::valuetype & type,
		size_t size)
	: vectortype(type, size)
	{}

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;

	virtual std::string
	debug_string() const override;
};

class scalablevectortype final : public vectortype {
public:
	~scalablevectortype() override;

	scalablevectortype(
		const jive::valuetype & type,
		size_t size)
	: vectortype(type, size)
	{}

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;

	virtual std::string
	debug_string() const override;
};

/* loop state type */

class loopstatetype final : public jive::statetype {
public:
	virtual
	~loopstatetype();

	constexpr
	loopstatetype() noexcept
	: statetype()
	{}

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;

	virtual std::string
	debug_string() const override;

	static std::unique_ptr<jive::type>
	create()
	{
		return std::unique_ptr<jive::type>(new loopstatetype());
	}
};

/** \brief Input/Output state type
*
* This type is used for state edges that sequentialize input/output operations.
*/
class iostatetype final : public jive::statetype {
public:
	~iostatetype() override;

	constexpr
	iostatetype() noexcept
	{}

	virtual bool
	operator==(const jive::type & other) const noexcept override;

	virtual std::unique_ptr<jive::type>
	copy() const override;

	virtual std::string
	debug_string() const override;

	static std::unique_ptr<jive::type>
	create()
	{
		return std::make_unique<iostatetype>();
	}

	static const iostatetype &
	instance() noexcept
	{
		static iostatetype iotype;
		return iotype;
	}
};

/** \brief Memory state type class
 *
 * Represents the type of abstract memory locations and is used in state edges for sequentialiazing memory operations,
 * such as load and store operations.
 */
class MemoryStateType final : public jive::statetype {
public:
  ~MemoryStateType() noexcept override;

  constexpr
  MemoryStateType() noexcept
  : jive::statetype()
  {}

  std::string
  debug_string() const override;

  bool
  operator==(const jive::type & other) const noexcept override;

  std::unique_ptr<jive::type>
  copy() const override;

  static std::unique_ptr<MemoryStateType>
  Create()
  {
    return std::make_unique<MemoryStateType>();
  }
};

template <class ELEMENTYPE> static inline bool
IsOrContains(const jive::type & type)
{
  if (jive::is<ELEMENTYPE>(type))
    return true;

  if (auto arrayType = dynamic_cast<const arraytype*>(&type))
    return IsOrContains<ELEMENTYPE>(arrayType->element_type());

  if (auto structType = dynamic_cast<const structtype*>(&type)) {
    auto structDeclaration = structType->declaration();
    for (size_t n = 0; n < structDeclaration->nelements(); n++)
      return IsOrContains<ELEMENTYPE>(structDeclaration->element(n));

    return false;
  }

  if (auto vectorType = dynamic_cast<const vectortype*>(&type))
    return IsOrContains<ELEMENTYPE>(vectorType->type());

  if (auto pointerType = dynamic_cast<const PointerType*>(&type))
    return IsOrContains<ELEMENTYPE>(pointerType->GetElementType());

  return false;
}

}

#endif
