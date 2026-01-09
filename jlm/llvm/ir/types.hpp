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
class PointerType final : public jlm::rvsdg::Type
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

  rvsdg::TypeKind
  Kind() const noexcept override;

  static std::shared_ptr<const PointerType>
  Create();
};

class ArrayType final : public rvsdg::Type
{
public:
  ~ArrayType() noexcept override;

  ArrayType(std::shared_ptr<const Type> type, size_t nelements)
      : nelements_(nelements),
        type_(std::move(type))
  {}

  ArrayType(const ArrayType & other) = default;

  ArrayType(ArrayType && other) = default;

  ArrayType &
  operator=(const ArrayType &) = delete;

  ArrayType &
  operator=(ArrayType &&) = delete;

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  rvsdg::TypeKind
  Kind() const noexcept override;

  inline size_t
  nelements() const noexcept
  {
    return nelements_;
  }

  [[nodiscard]] const rvsdg::Type &
  element_type() const noexcept
  {
    return *type_;
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::Type> &
  GetElementType() const noexcept
  {
    return type_;
  }

  static std::shared_ptr<const ArrayType>
  Create(std::shared_ptr<const Type> type, size_t nelements)
  {
    return std::make_shared<ArrayType>(std::move(type), nelements);
  }

private:
  size_t nelements_;
  std::shared_ptr<const rvsdg::Type> type_;
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

class FloatingPointType final : public rvsdg::Type
{
public:
  ~FloatingPointType() noexcept override;

  explicit FloatingPointType(const fpsize & size)
      : size_(size)
  {}

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  rvsdg::TypeKind
  Kind() const noexcept override;

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

class VariableArgumentType final : public rvsdg::Type
{
public:
  ~VariableArgumentType() noexcept override;

  constexpr VariableArgumentType() = default;

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  rvsdg::TypeKind
  Kind() const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  static std::shared_ptr<const VariableArgumentType>
  Create();
};

/** \brief StructType class
 *
 * This class is the equivalent of LLVM's StructType class.
 * There are two different kinds of struct types: Literal structs and Identified structs.
 * Literal struct types (e.g., { i32, i32 }) are uniqued structurally.
 * Identified structs (e.g., foo or %42) may optionally have a name and are not uniqued.
 */
class StructType final : public rvsdg::Type
{
public:
  class Declaration;

  ~StructType() override;

private:
  StructType(
      std::string name,
      std::unique_ptr<Declaration> declaration,
      bool isPacked,
      bool isLiteral)
      : name_(std::move(name)),
        declaration_(std::move(declaration)),
        isPacked_(isPacked),
        isLiteral_(isLiteral)
  {
    // Literal structs may not have names
    if (isLiteral)
      JLM_ASSERT(name.empty());
  }

public:
  StructType(const StructType &) = delete;

  StructType(StructType &&) = delete;

  StructType &
  operator=(const StructType &) = delete;

  StructType &
  operator=(StructType &&) = delete;

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  rvsdg::TypeKind
  Kind() const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  /**
   * Checks if the struct is an identified struct that has a name.
   * Will always return false for literal structs.
   * @return true if the struct has a name, false otherwise.
   */
  [[nodiscard]] bool
  HasName() const noexcept
  {
    if (isLiteral_)
      JLM_ASSERT(name_.empty());
    return !name_.empty();
  }

  /**
   * Returns the name of the identified struct.
   * If the struct has no name, an empty string is returned.
   * Must not be called on literal structs.
   * @pre the struct is an identified struct.
   * @return the name of the struct, or an empty string.
   */
  [[nodiscard]] const std::string &
  GetName() const noexcept
  {
    JLM_ASSERT(!isLiteral_);
    return name_;
  }

  [[nodiscard]] bool
  IsPacked() const noexcept
  {
    return isPacked_;
  }

  /**
   * Struct types can either be literal or identified.
   * Literal structs are defined only through their fields, and can not have names.
   * @return true if this struct type is literal, false if it is identified.
   */
  [[nodiscard]] bool
  IsLiteral() const noexcept
  {
    return isLiteral_;
  }

  /**
   * Gets the struct's declaration, which is the list of fields in the struct.
   * @return the struct's declaration
   */
  [[nodiscard]] const Declaration &
  GetDeclaration() const noexcept
  {
    return *declaration_;
  }

  /**
   * Gets the position of the given field, as a byte offset from the start of the struct.
   * Non-packed structs use padding to respect the alignment of each field, just like in C.
   * Packed structs have no padding, and no alignment.
   * @param fieldIndex the index of the field, must be valid
   * @return the byte offset of the given field
   */
  [[nodiscard]] size_t
  GetFieldOffset(size_t fieldIndex) const;

  /**
   * Creates an identified struct, with a name. The name should be unique to this struct.
   * @param name the name of the struct, or an empty string if the struct is unnamed.
   * @param declaration the fields in the struct
   * @param isPacked true if the struct is packed (no padding or alignment), false otherwise
   * @return the created struct type
   */
  static std::shared_ptr<const StructType>
  CreateIdentified(std::string name, std::unique_ptr<Declaration> declaration, bool isPacked)
  {
    return std::shared_ptr<StructType>(
        new StructType(std::move(name), std::move(declaration), isPacked, false));
  }

  /**
   * Creates an identified struct, without a name.
   * @param declaration the fields in the struct
   * @param isPacked true if the struct is packed (no padding or alignment), false otherwise
   * @return the created struct type
   */
  static std::shared_ptr<const StructType>
  CreateIdentified(std::unique_ptr<Declaration> declaration, bool isPacked)
  {
    return CreateIdentified("", std::move(declaration), isPacked);
  }

  /**
   * Creates a literal struct, which is anonymous and only identified through its fields.
   * @param declaration the fields in the struct
   * @param isPacked true if the struct is packed (no padding or alignment), false otherwise
   * @return the created struct type
   */
  static std::shared_ptr<const StructType>
  CreateLiteral(std::unique_ptr<Declaration> declaration, bool isPacked)
  {
    // Literal structs don't have names, so always use the empty string
    return std::shared_ptr<StructType>(new StructType("", std::move(declaration), isPacked, true));
  }

private:
  std::string name_;
  std::unique_ptr<Declaration> declaration_;
  bool isPacked_;
  bool isLiteral_;
};

class StructType::Declaration final
{
public:
  ~Declaration() = default;

  explicit Declaration(std::vector<std::shared_ptr<const rvsdg::Type>> types)
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

  [[nodiscard]] const Type &
  GetElement(size_t index) const noexcept
  {
    JLM_ASSERT(index < NumElements());
    return *Types_[index].get();
  }

  [[nodiscard]] std::shared_ptr<const Type>
  GetElementType(size_t index) const noexcept
  {
    JLM_ASSERT(index < NumElements());
    return Types_[index];
  }

  void
  Append(std::shared_ptr<const Type> type)
  {
    Types_.push_back(std::move(type));
  }

  std::unique_ptr<Declaration>
  copy() const
  {
    return std::make_unique<Declaration>(*this);
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

class VectorType : public rvsdg::Type
{
public:
  VectorType(std::shared_ptr<const Type> type, size_t size)
      : size_(size),
        type_(std::move(type))
  {}

  VectorType(const VectorType & other) = default;

  VectorType(VectorType && other) = default;

  VectorType &
  operator=(const VectorType & other) = default;

  VectorType &
  operator=(VectorType && other) = default;

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  rvsdg::TypeKind
  Kind() const noexcept override;

  size_t
  size() const noexcept
  {
    return size_;
  }

  [[nodiscard]] const rvsdg::Type &
  type() const noexcept
  {
    return *type_;
  }

  [[nodiscard]] const std::shared_ptr<const rvsdg::Type> &
  Type() const noexcept
  {
    return type_;
  }

private:
  size_t size_;
  std::shared_ptr<const rvsdg::Type> type_;
};

class FixedVectorType final : public VectorType
{
public:
  ~FixedVectorType() noexcept override;

  FixedVectorType(std::shared_ptr<const rvsdg::Type> type, size_t size)
      : VectorType(std::move(type), size)
  {}

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  static std::shared_ptr<const FixedVectorType>
  Create(std::shared_ptr<const rvsdg::Type> type, size_t size)
  {
    return std::make_shared<FixedVectorType>(std::move(type), size);
  }
};

class ScalableVectorType final : public VectorType
{
public:
  ~ScalableVectorType() noexcept override;

  ScalableVectorType(std::shared_ptr<const rvsdg::Type> type, size_t size)
      : VectorType(std::move(type), size)
  {}

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  static std::shared_ptr<const ScalableVectorType>
  Create(std::shared_ptr<const rvsdg::Type> type, size_t size)
  {
    return std::make_shared<ScalableVectorType>(std::move(type), size);
  }
};

/** \brief Input/Output state type
 *
 * This type is used for state edges that sequentialize input/output operations.
 */
class IOStateType final : public rvsdg::Type
{
public:
  ~IOStateType() noexcept override;

  constexpr IOStateType() noexcept = default;

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  rvsdg::TypeKind
  Kind() const noexcept override;

  static std::shared_ptr<const IOStateType>
  Create();
};

/** \brief Memory state type class
 *
 * Represents the type of abstract memory locations and is used in state edges for sequentialiazing
 * memory operations, such as load and store operations.
 */
class MemoryStateType final : public rvsdg::Type
{
public:
  ~MemoryStateType() noexcept override;

  constexpr MemoryStateType() noexcept = default;

  [[nodiscard]] std::string
  debug_string() const override;

  bool
  operator==(const jlm::rvsdg::Type & other) const noexcept override;

  [[nodiscard]] std::size_t
  ComputeHash() const noexcept override;

  rvsdg::TypeKind
  Kind() const noexcept override;

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
 * More specifically, the size is the number of bytes affected when storing value of the given type
 * to memory. Unlike C's sizeof() operator, the size is not rounded up to a multiple of alignment.
 * @see GetTypeAllocSize() for the size rounded up to a multiple of alignment.
 * @param type the ValueType
 * @return the byte size of the type
 */
[[nodiscard]] size_t
GetTypeStoreSize(const rvsdg::Type & type);

/**
 * Returns the size of the given type's representation, in bytes.
 * It corresponds to the sizeof() operator in C, so the size is a multiple of the type's alignment.
 * This is the offset between consecutive elements in an array,
 * and also the size of the stack allocation created by an alloca operation with the given type.
 * @see GetTypeStoreSize() for the number of bytes that are actually overwritten when storing.
 * @param type the ValueType
 * @return the byte size of the type
 */
[[nodiscard]] size_t
GetTypeAllocSize(const rvsdg::Type & type);

/**
 * Returns the natural alignment of the given type, in bytes.
 * Types are not guaranteed to be stored at their natural alignment,
 * so instead check the alignment of the store and load operations.
 * A non-packed struct will add padding to maintain alignment.
 * @param type the ValueType
 * @return the byte alignment of the type
 */
[[nodiscard]] size_t
GetTypeAlignment(const rvsdg::Type & type);

}

#endif
