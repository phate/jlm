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
class StringAttribute final : public Attribute
{
public:
  ~StringAttribute() noexcept override;

  StringAttribute(const std::string & kind, const std::string & value)
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
class EnumAttribute : public Attribute
{
public:
  ~EnumAttribute() noexcept override;

  explicit EnumAttribute(const Attribute::kind & kind)
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
class IntAttribute final : public EnumAttribute
{
public:
  ~IntAttribute() noexcept override;

  IntAttribute(Attribute::kind kind, uint64_t value)
      : EnumAttribute(kind),
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
class TypeAttribute final : public EnumAttribute
{
public:
  ~TypeAttribute() noexcept override;

  TypeAttribute(Attribute::kind kind, std::shared_ptr<const jlm::rvsdg::ValueType> type)
      : EnumAttribute(kind),
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
struct Hash<jlm::llvm::EnumAttribute>
{
  std::size_t
  operator()(const jlm::llvm::EnumAttribute & attribute) const noexcept
  {
    return std::hash<jlm::llvm::Attribute::kind>()(attribute.kind());
  }
};

template<>
struct Hash<jlm::llvm::IntAttribute>
{
  std::size_t
  operator()(const jlm::llvm::IntAttribute & attribute) const noexcept
  {
    auto kindHash = std::hash<jlm::llvm::Attribute::kind>()(attribute.kind());
    auto valueHash = std::hash<uint64_t>()(attribute.value());
    return util::CombineHashes(kindHash, valueHash);
  }
};

template<>
struct Hash<jlm::llvm::StringAttribute>
{
  std::size_t
  operator()(const jlm::llvm::StringAttribute & attribute) const noexcept
  {
    auto kindHash = std::hash<std::string>()(attribute.kind());
    auto valueHash = std::hash<std::string>()(attribute.value());
    return util::CombineHashes(kindHash, valueHash);
  }
};

template<>
struct Hash<jlm::llvm::TypeAttribute>
{
  std::size_t
  operator()(const jlm::llvm::TypeAttribute & attribute) const noexcept
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
class AttributeSet final
{
  using EnumAttributeHashSet = util::HashSet<EnumAttribute>;
  using IntAttributeHashSet = util::HashSet<IntAttribute>;
  using TypeAttributeHashSet = util::HashSet<TypeAttribute>;
  using StringAttributeHashSet = util::HashSet<StringAttribute>;

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
  InsertEnumAttribute(const EnumAttribute & attribute)
  {
    EnumAttributes_.Insert(attribute);
  }

  void
  InsertIntAttribute(const IntAttribute & attribute)
  {
    IntAttributes_.Insert(attribute);
  }

  void
  InsertTypeAttribute(const TypeAttribute & attribute)
  {
    TypeAttributes_.Insert(attribute);
  }

  void
  InsertStringAttribute(const StringAttribute & attribute)
  {
    StringAttributes_.Insert(attribute);
  }

  bool
  operator==(const AttributeSet & other) const noexcept
  {
    return IntAttributes_ == other.IntAttributes_ && EnumAttributes_ == other.EnumAttributes_
        && TypeAttributes_ == other.TypeAttributes_ && StringAttributes_ == other.StringAttributes_;
  }

  bool
  operator!=(const AttributeSet & other) const noexcept
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
