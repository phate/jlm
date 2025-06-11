/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_ATTRIBUTE_HPP
#define JLM_LLVM_IR_ATTRIBUTE_HPP

#include <jlm/rvsdg/type.hpp>
#include <jlm/util/common.hpp>
#include <jlm/util/HashSet.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace jlm::llvm
{

/** \brief Attribute
 */
class Attribute
{
public:
  enum class kind
  {
    None, ///< No attributes have been set

    FirstEnumAttr,
    AllocAlign,
    AllocatedPointer,
    AlwaysInline,
    Builtin,
    Cold,
    Convergent,
    CoroDestroyOnlyWhenComplete,
    DeadOnUnwind,
    DisableSanitizerInstrumentation,
    FnRetThunkExtern,
    Hot,
    ImmArg,
    InReg,
    InlineHint,
    JumpTable,
    Memory,
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
    OptimizeForDebugging,
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
    SkipProfile,
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
    Writable,
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
    NoFPClass,
    StackAlignment,
    UWTable,
    VScaleRange,
    LastIntAttr,

    EndAttrKinds ///< Sentinel value useful for loops
  };

  virtual ~Attribute() noexcept;

  virtual bool
  operator==(const Attribute &) const = 0;

  virtual bool
  operator!=(const Attribute & other) const
  {
    return !operator==(other);
  }
};

/** \brief String attribute
 */
class string_attribute final : public Attribute
{
public:
  ~string_attribute() noexcept override;

  string_attribute(const std::string & kind, const std::string & value)
      : kind_(kind),
        value_(value)
  {}

  [[nodiscard]] const std::string &
  kind() const noexcept
  {
    return kind_;
  }

  [[nodiscard]] const std::string &
  value() const noexcept
  {
    return value_;
  }

  bool
  operator==(const Attribute &) const override;

private:
  std::string kind_;
  std::string value_;
};

/** \brief Enum attribute
 */
class enum_attribute : public Attribute
{
public:
  ~enum_attribute() noexcept override;

  explicit enum_attribute(const Attribute::kind & kind)
      : kind_(kind)
  {}

  [[nodiscard]] const Attribute::kind &
  kind() const noexcept
  {
    return kind_;
  }

  bool
  operator==(const Attribute &) const override;

private:
  Attribute::kind kind_;
};

/** \brief Integer attribute
 */
class int_attribute final : public enum_attribute
{
public:
  ~int_attribute() noexcept override;

  int_attribute(Attribute::kind kind, uint64_t value)
      : enum_attribute(kind),
        value_(value)
  {}

  [[nodiscard]] uint64_t
  value() const noexcept
  {
    return value_;
  }

  bool
  operator==(const Attribute &) const override;

private:
  uint64_t value_;
};

/** \brief Type attribute
 */
class type_attribute final : public enum_attribute
{
public:
  ~type_attribute() noexcept override;

  type_attribute(Attribute::kind kind, std::shared_ptr<const jlm::rvsdg::ValueType> type)
      : enum_attribute(kind),
        type_(std::move(type))
  {}

  [[nodiscard]] const jlm::rvsdg::ValueType &
  type() const noexcept
  {
    return *type_;
  }

  bool
  operator==(const Attribute &) const override;

private:
  std::shared_ptr<const jlm::rvsdg::ValueType> type_;
};

}

namespace jlm::util
{

template<>
struct Hash<jlm::llvm::enum_attribute>
{
  std::size_t
  operator()(const jlm::llvm::enum_attribute & attribute) const noexcept
  {
    return std::hash<jlm::llvm::Attribute::kind>()(attribute.kind());
  }
};

template<>
struct Hash<jlm::llvm::int_attribute>
{
  std::size_t
  operator()(const jlm::llvm::int_attribute & attribute) const noexcept
  {
    auto kindHash = std::hash<jlm::llvm::Attribute::kind>()(attribute.kind());
    auto valueHash = std::hash<uint64_t>()(attribute.value());
    return util::CombineHashes(kindHash, valueHash);
  }
};

template<>
struct Hash<jlm::llvm::string_attribute>
{
  std::size_t
  operator()(const jlm::llvm::string_attribute & attribute) const noexcept
  {
    auto kindHash = std::hash<std::string>()(attribute.kind());
    auto valueHash = std::hash<std::string>()(attribute.value());
    return util::CombineHashes(kindHash, valueHash);
  }
};

template<>
struct Hash<jlm::llvm::type_attribute>
{
  std::size_t
  operator()(const jlm::llvm::type_attribute & attribute) const noexcept
  {
    auto kindHash = std::hash<jlm::llvm::Attribute::kind>()(attribute.kind());
    auto typeHash = attribute.type().ComputeHash();
    return util::CombineHashes(kindHash, typeHash);
  }
};

}

namespace jlm::llvm
{

/** \brief Attribute set
 */
class attributeset final
{
  using EnumAttributeHashSet = util::HashSet<enum_attribute>;
  using IntAttributeHashSet = util::HashSet<int_attribute>;
  using TypeAttributeHashSet = util::HashSet<type_attribute>;
  using StringAttributeHashSet = util::HashSet<string_attribute>;

  using EnumAttributeRange = util::IteratorRange<EnumAttributeHashSet::ItemConstIterator>;
  using IntAttributeRange = util::IteratorRange<IntAttributeHashSet::ItemConstIterator>;
  using TypeAttributeRange = util::IteratorRange<TypeAttributeHashSet::ItemConstIterator>;
  using StringAttributeRange = util::IteratorRange<StringAttributeHashSet::ItemConstIterator>;

public:
  [[nodiscard]] EnumAttributeRange
  EnumAttributes() const;

  [[nodiscard]] IntAttributeRange
  IntAttributes() const;

  [[nodiscard]] TypeAttributeRange
  TypeAttributes() const;

  [[nodiscard]] StringAttributeRange
  StringAttributes() const;

  void
  InsertEnumAttribute(const enum_attribute & attribute)
  {
    EnumAttributes_.Insert(attribute);
  }

  void
  InsertIntAttribute(const int_attribute & attribute)
  {
    IntAttributes_.Insert(attribute);
  }

  void
  InsertTypeAttribute(const type_attribute & attribute)
  {
    TypeAttributes_.Insert(attribute);
  }

  void
  InsertStringAttribute(const string_attribute & attribute)
  {
    StringAttributes_.Insert(attribute);
  }

  bool
  operator==(const attributeset & other) const noexcept
  {
    return IntAttributes_ == other.IntAttributes_ && EnumAttributes_ == other.EnumAttributes_
        && TypeAttributes_ == other.TypeAttributes_ && StringAttributes_ == other.StringAttributes_;
  }

  bool
  operator!=(const attributeset & other) const noexcept
  {
    return !(*this == other);
  }

private:
  EnumAttributeHashSet EnumAttributes_{};
  IntAttributeHashSet IntAttributes_{};
  TypeAttributeHashSet TypeAttributes_{};
  StringAttributeHashSet StringAttributes_{};
};

}

#endif
