/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_TYPES_HPP
#define JLM_LLVM_IR_TYPES_HPP

#include <jlm/rvsdg/FunctionType.hpp>
#include <jlm/rvsdg/type.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/iterator_range.hpp>

#include <memory>
#include <vector>

namespace jlm::llvm
{

/** \brief PointerType class
 *
 * This operator is the Jlm equivalent of LLVM's PointerType class.
 */
class PointerType final : public jlm::rvsdg::ValueType
{
public:
  ~PointerType() noexcept override;

  PointerType() = default;

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  static std::shared_ptr<const PointerType>
  Create();
};

class ArrayType final : public rvsdg::ValueType
{
public:
  ~ArrayType() noexcept override;

  ArrayType(std::shared_ptr<const ValueType> type, size_t nelements)
      : nelements_(nelements),
        type_(std::move(type))
  {}

  ArrayType(const ArrayType & other) = default;

  ArrayType(ArrayType && other) = default;

  ArrayType &
  operator=(const ArrayType &) = delete;

  ArrayType &
  operator=(ArrayType &&) = delete;

  virtual std::string
  debug_string() const override;

  virtual bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  inline size_t
  nelements() const noexcept
  {
    return nelements_;
  }

  [[nodiscard]] const rvsdg::ValueType &
  element_type() const noexcept
  {
    return *type_;
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::ValueType> &
  GetElementType() const noexcept
  {
    return type_;
  }

  static std::shared_ptr<const ArrayType>
  Create(std::shared_ptr<const ValueType> type, size_t nelements)
  {
    return std::make_shared<ArrayType>(std::move(type), nelements);
  }

private:
  size_t nelements_;
  std::shared_ptr<const rvsdg::ValueType> type_;
};

/* floating point type */

enum class fpsize
{
  half,
  flt,
  dbl,
  x86fp80,
  fp128
};

class FloatingPointType final : public rvsdg::ValueType
{
public:
  ~FloatingPointType() noexcept override;

  explicit FloatingPointType(const fpsize & size)
      : size_(size)
  {}

  virtual std::string
  debug_string() const override;

  virtual bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  inline const fpsize &
  size() const noexcept
  {
    return size_;
  }

  static std::shared_ptr<const FloatingPointType>
  Create(fpsize size);

private:
  fpsize size_;
};

class VariableArgumentType final : public rvsdg::StateType
{
public:
  ~VariableArgumentType() noexcept override;

  constexpr VariableArgumentType() = default;

  virtual bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  virtual std::string
  debug_string() const override;

  static std::shared_ptr<const VariableArgumentType>
  Create();
};

/** \brief StructType class
 *
 * This class is the equivalent of LLVM's StructType class.
 */
class StructType final : public rvsdg::ValueType
{
public:
  class Declaration;

  ~StructType() override;

  StructType(bool isPacked, const Declaration & declaration)
      : rvsdg::ValueType(),
        IsPacked_(isPacked),
        Declaration_(declaration)
  {}

  StructType(std::string name, bool isPacked, const Declaration & declaration)
      : rvsdg::ValueType(),
        IsPacked_(isPacked),
        Name_(std::move(name)),
        Declaration_(declaration)
  {}

  StructType(const StructType &) = default;

  StructType(StructType &&) = delete;

  StructType &
  operator=(const StructType &) = delete;

  StructType &
  operator=(StructType &&) = delete;

  [[nodiscard]] bool
  HasName() const noexcept
  {
    return !Name_.empty();
  }

  [[nodiscard]] const std::string &
  GetName() const noexcept
  {
    return Name_;
  }

  [[nodiscard]] bool
  IsPacked() const noexcept
  {
    return IsPacked_;
  }

  [[nodiscard]] const Declaration &
  GetDeclaration() const noexcept
  {
    return Declaration_;
  }

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  static std::shared_ptr<const StructType>
  Create(const std::string & name, bool isPacked, const Declaration & declaration)
  {
    return std::make_shared<StructType>(name, isPacked, declaration);
  }

  static std::shared_ptr<const StructType>
  Create(bool isPacked, const Declaration & declaration)
  {
    return std::make_shared<StructType>(isPacked, declaration);
  }

private:
  bool IsPacked_;
  std::string Name_;
  const Declaration & Declaration_;
};

class StructType::Declaration final
{
public:
  ~Declaration() = default;

  Declaration(std::vector<std::shared_ptr<const rvsdg::Type>> types)
      : Types_(std::move(types))
  {}

  Declaration() = default;

  Declaration(const Declaration &) = default;

  Declaration &
  operator=(const Declaration &) = default;

  [[nodiscard]] size_t
  NumElements() const noexcept
  {
    return Types_.size();
  }

  [[nodiscard]] const ValueType &
  GetElement(size_t index) const noexcept
  {
    JLM_ASSERT(index < NumElements());
    return *util::AssertedCast<const ValueType>(Types_[index].get());
  }

  [[nodiscard]] std::shared_ptr<const ValueType>
  GetElementType(size_t index) const noexcept
  {
    JLM_ASSERT(index < NumElements());
    auto type = std::dynamic_pointer_cast<const ValueType>(Types_[index]);
    JLM_ASSERT(type);
    return type;
  }

  void
  Append(std::shared_ptr<const ValueType> type)
  {
    Types_.push_back(std::move(type));
  }

  static std::unique_ptr<Declaration>
  Create()
  {
    return std::unique_ptr<Declaration>(new Declaration());
  }

  static std::unique_ptr<Declaration>
  Create(std::vector<std::shared_ptr<const rvsdg::Type>> types)
  {
    return std::make_unique<Declaration>(std::move(types));
  }

private:
  std::vector<std::shared_ptr<const rvsdg::Type>> Types_;
};

class VectorType : public rvsdg::ValueType
{
public:
  VectorType(std::shared_ptr<const ValueType> type, size_t size)
      : size_(size),
        type_(std::move(type))
  {}

  VectorType(const VectorType & other) = default;

  VectorType(VectorType && other) = default;

  VectorType &
  operator=(const VectorType & other) = default;

  VectorType &
  operator=(VectorType && other) = default;

  virtual bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  size_t
  size() const noexcept
  {
    return size_;
  }

  [[nodiscard]] const rvsdg::ValueType &
  type() const noexcept
  {
    return *type_;
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::ValueType> &
  Type() const noexcept
  {
    return type_;
  }

private:
  size_t size_;
  std::shared_ptr<const rvsdg::ValueType> type_;
};

class FixedVectorType final : public VectorType
{
public:
  ~FixedVectorType() noexcept override;

  FixedVectorType(std::shared_ptr<const ValueType> type, size_t size)
      : VectorType(std::move(type), size)
  {}

  virtual bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  virtual std::string
  debug_string() const override;

  static std::shared_ptr<const FixedVectorType>
  Create(std::shared_ptr<const rvsdg::ValueType> type, size_t size)
  {
    return std::make_shared<FixedVectorType>(std::move(type), size);
  }
};

class ScalableVectorType final : public VectorType
{
public:
  ~ScalableVectorType() noexcept override;

  ScalableVectorType(std::shared_ptr<const ValueType> type, size_t size)
      : VectorType(std::move(type), size)
  {}

  virtual bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  virtual std::string
  debug_string() const override;

  static std::shared_ptr<const ScalableVectorType>
  Create(std::shared_ptr<const rvsdg::ValueType> type, size_t size)
  {
    return std::make_shared<ScalableVectorType>(std::move(type), size);
  }
};

/** \brief Input/Output state type
 *
 * This type is used for state edges that sequentialize input/output operations.
 */
class IOStateType final : public rvsdg::StateType
{
public:
  ~IOStateType() noexcept override;

  constexpr IOStateType() noexcept = default;

  virtual bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  virtual std::string
  debug_string() const override;

  static std::shared_ptr<const IOStateType>
  Create();
};

/** \brief Memory state type class
 *
 * Represents the type of abstract memory locations and is used in state edges for sequentialiazing
 * memory operations, such as load and store operations.
 */
class MemoryStateType final : public rvsdg::StateType
{
public:
  ~MemoryStateType() noexcept override;

  constexpr MemoryStateType() noexcept = default;

  std::string
  debug_string() const override;

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  static std::shared_ptr<const MemoryStateType>
  Create();
};

template<class ELEMENTYPE>
inline bool
IsOrContains(const jlm::rvsdg::Type & type)
{
  if (jlm::rvsdg::is<ELEMENTYPE>(type))
    return true;

  if (auto arrayType = dynamic_cast<const ArrayType *>(&type))
    return IsOrContains<ELEMENTYPE>(arrayType->element_type());

  if (auto structType = dynamic_cast<const StructType *>(&type))
  {
    auto & structDeclaration = structType->GetDeclaration();
    for (size_t n = 0; n < structDeclaration.NumElements(); n++)
      if (IsOrContains<ELEMENTYPE>(structDeclaration.GetElement(n)))
        return true;

    return false;
  }

  if (const auto vectorType = dynamic_cast<const VectorType *>(&type))
    return IsOrContains<ELEMENTYPE>(vectorType->type());

  return false;
}

/**
 * Given a type, determines if it is one of LLVM's aggregate types.
 * Vectors are not considered to be aggregate types, despite being based on a subtype.
 * @param type the type to check
 * @return true if the type is an aggregate type, false otherwise
 */
inline bool
IsAggregateType(const jlm::rvsdg::Type & type)
{
  return jlm::rvsdg::is<ArrayType>(type) || jlm::rvsdg::is<StructType>(type);
}

/**
 * Returns the size of the given type's representation, in bytes.
 * The size is always a multiple of the alignment, just like the C operator sizeof().
 * This means the size includes any padding at the end.
 * @param type the ValueType
 * @return the byte size of the type
 */
[[nodiscard]] size_t
GetTypeSize(const rvsdg::ValueType & type);

/**
 * Returns the natural alignment of the given type, in bytes.
 * Types are not guaranteed to be stored at their natural alignment,
 * so instead check the alignment of the store and load operations.
 * A non-packed struct will add padding to maintain alignment.
 * @param type the ValueType
 * @return the byte alignment of the type
 */
[[nodiscard]] size_t
GetTypeAlignment(const rvsdg::ValueType & type);

/**
 * Gets the offset at which the given field is located in memory.
 * Respects alignment of each field, just like in C. Supports packed structs.
 * @param structType the struct type
 * @param fieldIndex the index of the field, must be a valid field
 * @return the byte offset of the field, relative to the beginning of the struct.
 */
[[nodiscard]] size_t
GetStructFieldOffset(const StructType & structType, size_t fieldIndex);

}

#endif
