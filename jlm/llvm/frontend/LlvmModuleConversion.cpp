/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/CallingConvention.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/AggregateOperations.hpp>
#include <jlm/llvm/ir/operators/IntegerOperations.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/SpecializedArithmeticIntrinsicOperations.hpp>
#include <jlm/llvm/ir/TypeConverter.hpp>

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

namespace jlm::llvm
{

using BasicBlockMap = util::BijectiveMap<const ::llvm::BasicBlock *, BasicBlock *>;

class Context final
{
public:
  explicit Context(InterProceduralGraphModule & im)
      : module_(im),
        node_(nullptr),
        iostate_(nullptr),
        memory_state_(nullptr)
  {}

  const llvm::Variable *
  result() const noexcept
  {
    return result_;
  }

  inline void
  set_result(const llvm::Variable * result)
  {
    result_ = result;
  }

  llvm::Variable *
  iostate() const noexcept
  {
    return iostate_;
  }

  void
  set_iostate(llvm::Variable * state)
  {
    iostate_ = state;
  }

  inline llvm::Variable *
  memory_state() const noexcept
  {
    return memory_state_;
  }

  inline void
  set_memory_state(llvm::Variable * state)
  {
    memory_state_ = state;
  }

  inline bool
  has(const ::llvm::BasicBlock * bb) const noexcept
  {
    return bbmap_.HasKey(bb);
  }

  inline bool
  has(BasicBlock * bb) const noexcept
  {
    return bbmap_.HasValue(bb);
  }

  inline BasicBlock *
  get(const ::llvm::BasicBlock * bb) const noexcept
  {
    return bbmap_.LookupKey(bb);
  }

  inline const ::llvm::BasicBlock *
  get(BasicBlock * bb) const noexcept
  {
    return bbmap_.LookupValue(bb);
  }

  inline void
  set_basic_block_map(BasicBlockMap bbmap)
  {
    bbmap_ = std::move(bbmap);
  }

  inline bool
  has_value(const ::llvm::Value * value) const noexcept
  {
    return vmap_.find(value) != vmap_.end();
  }

  inline const llvm::Variable *
  lookup_value(const ::llvm::Value * value) const noexcept
  {
    JLM_ASSERT(has_value(value));
    return vmap_.find(value)->second;
  }

  inline void
  insert_value(const ::llvm::Value * value, const llvm::Variable * variable)
  {
    JLM_ASSERT(!has_value(value));
    vmap_[value] = variable;
  }

  [[nodiscard]] InterProceduralGraphModule &
  module() const noexcept
  {
    return module_;
  }

  inline void
  set_node(InterProceduralGraphNode * node) noexcept
  {
    node_ = node;
  }

  inline InterProceduralGraphNode *
  node() const noexcept
  {
    return node_;
  }

  TypeConverter &
  GetTypeConverter() noexcept
  {
    return TypeConverter_;
  }

private:
  InterProceduralGraphModule & module_;
  BasicBlockMap bbmap_;
  InterProceduralGraphNode * node_;
  const llvm::Variable * result_{};
  llvm::Variable * iostate_;
  llvm::Variable * memory_state_;
  std::unordered_map<const ::llvm::Value *, const llvm::Variable *> vmap_;
  TypeConverter TypeConverter_;
};

static const Variable *
ConvertConstant(
    ::llvm::Constant *,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> &,
    Context &);

static const Variable *
convertInstruction(
    ::llvm::Instruction * instruction,
    std::vector<std::unique_ptr<ThreeAddressCode>> & threeAddressCodes,
    Context & context);

// Converts a value into a variable either representing the llvm
// value or representing a function object.
// The distinction stems from the fact that llvm treats functions simply
// as pointers to the function code while we distinguish between the two.
// This function can return either and caller needs to check / adapt.
const Variable *
ConvertValueOrFunction(::llvm::Value * v, tacsvector_t & tacs, Context & ctx)
{
  auto node = ctx.node();
  if (node && ctx.has_value(v))
  {
    if (auto callee = dynamic_cast<const FunctionVariable *>(ctx.lookup_value(v)))
      node->add_dependency(callee->function());

    if (auto data = dynamic_cast<const GlobalValue *>(ctx.lookup_value(v)))
      node->add_dependency(data->node());
  }

  if (ctx.has_value(v))
    return ctx.lookup_value(v);

  if (auto c = ::llvm::dyn_cast<::llvm::Constant>(v))
    return ConvertConstant(c, tacs, ctx);

  JLM_UNREACHABLE("This should not have happened!");
}

// Converts a value into a variable representing the llvm value.
const Variable *
ConvertValue(::llvm::Value * v, tacsvector_t & tacs, Context & ctx)
{
  const Variable * var = ConvertValueOrFunction(v, tacs, ctx);
  if (auto fntype = std::dynamic_pointer_cast<const rvsdg::FunctionType>(var->Type()))
  {
    auto operation = std::make_unique<FunctionToPointerOperation>(fntype);
    std::unique_ptr<ThreeAddressCode> ptr_cast =
        ThreeAddressCode::create(std::move(operation), { var });
    var = ptr_cast->result(0);
    tacs.push_back(std::move(ptr_cast));
  }
  return var;
}

static rvsdg::BitValueRepresentation
convert_apint(const ::llvm::APInt & value)
{
  ::llvm::APInt v;
  if (value.isNegative())
    v = -value;

  auto str = toString(value, 2, false);
  std::reverse(str.begin(), str.end());

  rvsdg::BitValueRepresentation vr(str.c_str());
  if (value.isNegative())
    vr = vr.sext(value.getBitWidth() - str.size());
  else
    vr = vr.zext(value.getBitWidth() - str.size());

  return vr;
}

Attribute::kind
ConvertAttributeKind(const ::llvm::Attribute::AttrKind & kind)
{
  typedef ::llvm::Attribute::AttrKind ak;

  static std::unordered_map<::llvm::Attribute::AttrKind, Attribute::kind> map(
      { { ak::None, Attribute::kind::None },
        { ak::FirstEnumAttr, Attribute::kind::FirstEnumAttr },
        { ak::AllocAlign, Attribute::kind::AllocAlign },
        { ak::AllocatedPointer, Attribute::kind::AllocatedPointer },
        { ak::AlwaysInline, Attribute::kind::AlwaysInline },
        { ak::Builtin, Attribute::kind::Builtin },
        { ak::Cold, Attribute::kind::Cold },
        { ak::Convergent, Attribute::kind::Convergent },
        { ak::CoroDestroyOnlyWhenComplete, Attribute::kind::CoroDestroyOnlyWhenComplete },
        { ak::DeadOnUnwind, Attribute::kind::DeadOnUnwind },
        { ak::DisableSanitizerInstrumentation, Attribute::kind::DisableSanitizerInstrumentation },
        { ak::FnRetThunkExtern, Attribute::kind::FnRetThunkExtern },
        { ak::Hot, Attribute::kind::Hot },
        { ak::ImmArg, Attribute::kind::ImmArg },
        { ak::InReg, Attribute::kind::InReg },
        { ak::InlineHint, Attribute::kind::InlineHint },
        { ak::JumpTable, Attribute::kind::JumpTable },
        { ak::Memory, Attribute::kind::Memory },
        { ak::MinSize, Attribute::kind::MinSize },
        { ak::MustProgress, Attribute::kind::MustProgress },
        { ak::Naked, Attribute::kind::Naked },
        { ak::Nest, Attribute::kind::Nest },
        { ak::NoAlias, Attribute::kind::NoAlias },
        { ak::NoBuiltin, Attribute::kind::NoBuiltin },
        { ak::NoCallback, Attribute::kind::NoCallback },
        { ak::NoCapture, Attribute::kind::NoCapture },
        { ak::NoCfCheck, Attribute::kind::NoCfCheck },
        { ak::NoDuplicate, Attribute::kind::NoDuplicate },
        { ak::NoFree, Attribute::kind::NoFree },
        { ak::NoImplicitFloat, Attribute::kind::NoImplicitFloat },
        { ak::NoInline, Attribute::kind::NoInline },
        { ak::NoMerge, Attribute::kind::NoMerge },
        { ak::NoProfile, Attribute::kind::NoProfile },
        { ak::NoRecurse, Attribute::kind::NoRecurse },
        { ak::NoRedZone, Attribute::kind::NoRedZone },
        { ak::NoReturn, Attribute::kind::NoReturn },
        { ak::NoSanitizeBounds, Attribute::kind::NoSanitizeBounds },
        { ak::NoSanitizeCoverage, Attribute::kind::NoSanitizeCoverage },
        { ak::NoSync, Attribute::kind::NoSync },
        { ak::NoUndef, Attribute::kind::NoUndef },
        { ak::NoUnwind, Attribute::kind::NoUnwind },
        { ak::NonLazyBind, Attribute::kind::NonLazyBind },
        { ak::NonNull, Attribute::kind::NonNull },
        { ak::NullPointerIsValid, Attribute::kind::NullPointerIsValid },
        { ak::OptForFuzzing, Attribute::kind::OptForFuzzing },
        { ak::OptimizeForDebugging, Attribute::kind::OptimizeForDebugging },
        { ak::OptimizeForSize, Attribute::kind::OptimizeForSize },
        { ak::OptimizeNone, Attribute::kind::OptimizeNone },
        { ak::PresplitCoroutine, Attribute::kind::PresplitCoroutine },
        { ak::ReadNone, Attribute::kind::ReadNone },
        { ak::ReadOnly, Attribute::kind::ReadOnly },
        { ak::Returned, Attribute::kind::Returned },
        { ak::ReturnsTwice, Attribute::kind::ReturnsTwice },
        { ak::SExt, Attribute::kind::SExt },
        { ak::SafeStack, Attribute::kind::SafeStack },
        { ak::SanitizeAddress, Attribute::kind::SanitizeAddress },
        { ak::SanitizeHWAddress, Attribute::kind::SanitizeHWAddress },
        { ak::SanitizeMemTag, Attribute::kind::SanitizeMemTag },
        { ak::SanitizeMemory, Attribute::kind::SanitizeMemory },
        { ak::SanitizeThread, Attribute::kind::SanitizeThread },
        { ak::ShadowCallStack, Attribute::kind::ShadowCallStack },
        { ak::SkipProfile, Attribute::kind::SkipProfile },
        { ak::Speculatable, Attribute::kind::Speculatable },
        { ak::SpeculativeLoadHardening, Attribute::kind::SpeculativeLoadHardening },
        { ak::StackProtect, Attribute::kind::StackProtect },
        { ak::StackProtectReq, Attribute::kind::StackProtectReq },
        { ak::StackProtectStrong, Attribute::kind::StackProtectStrong },
        { ak::StrictFP, Attribute::kind::StrictFP },
        { ak::SwiftAsync, Attribute::kind::SwiftAsync },
        { ak::SwiftError, Attribute::kind::SwiftError },
        { ak::SwiftSelf, Attribute::kind::SwiftSelf },
        { ak::WillReturn, Attribute::kind::WillReturn },
        { ak::Writable, Attribute::kind::Writable },
        { ak::WriteOnly, Attribute::kind::WriteOnly },
        { ak::ZExt, Attribute::kind::ZExt },
        { ak::LastEnumAttr, Attribute::kind::LastEnumAttr },
        { ak::FirstTypeAttr, Attribute::kind::FirstTypeAttr },
        { ak::ByRef, Attribute::kind::ByRef },
        { ak::ByVal, Attribute::kind::ByVal },
        { ak::ElementType, Attribute::kind::ElementType },
        { ak::InAlloca, Attribute::kind::InAlloca },
        { ak::Preallocated, Attribute::kind::Preallocated },
        { ak::StructRet, Attribute::kind::StructRet },
        { ak::LastTypeAttr, Attribute::kind::LastTypeAttr },
        { ak::FirstIntAttr, Attribute::kind::FirstIntAttr },
        { ak::Alignment, Attribute::kind::Alignment },
        { ak::AllocKind, Attribute::kind::AllocKind },
        { ak::AllocSize, Attribute::kind::AllocSize },
        { ak::Dereferenceable, Attribute::kind::Dereferenceable },
        { ak::DereferenceableOrNull, Attribute::kind::DereferenceableOrNull },
        { ak::NoFPClass, Attribute::kind::NoFPClass },
        { ak::StackAlignment, Attribute::kind::StackAlignment },
        { ak::UWTable, Attribute::kind::UWTable },
        { ak::VScaleRange, Attribute::kind::VScaleRange },
        { ak::LastIntAttr, Attribute::kind::LastIntAttr },
        { ak::EndAttrKinds, Attribute::kind::EndAttrKinds } });

  JLM_ASSERT(map.find(kind) != map.end());
  return map[kind];
}

static EnumAttribute
ConvertEnumAttribute(const ::llvm::Attribute & attribute)
{
  JLM_ASSERT(attribute.isEnumAttribute());
  auto kind = ConvertAttributeKind(attribute.getKindAsEnum());
  return EnumAttribute(kind);
}

static IntAttribute
ConvertIntAttribute(const ::llvm::Attribute & attribute)
{
  JLM_ASSERT(attribute.isIntAttribute());
  auto kind = ConvertAttributeKind(attribute.getKindAsEnum());
  return { kind, attribute.getValueAsInt() };
}

static TypeAttribute
ConvertTypeAttribute(const ::llvm::Attribute & attribute, TypeConverter & typeConverter)
{
  JLM_ASSERT(attribute.isTypeAttribute());

  if (attribute.getKindAsEnum() == ::llvm::Attribute::AttrKind::ByVal)
  {
    auto type = typeConverter.ConvertLlvmType(*attribute.getValueAsType());
    return { Attribute::kind::ByVal, std::move(type) };
  }

  if (attribute.getKindAsEnum() == ::llvm::Attribute::AttrKind::StructRet)
  {
    auto type = typeConverter.ConvertLlvmType(*attribute.getValueAsType());
    return { Attribute::kind::StructRet, std::move(type) };
  }

  JLM_UNREACHABLE("Unhandled attribute");
}

static StringAttribute
ConvertStringAttribute(const ::llvm::Attribute & attribute)
{
  JLM_ASSERT(attribute.isStringAttribute());
  return { attribute.getKindAsString().str(), attribute.getValueAsString().str() };
}

static AttributeSet
convert_attributes(const ::llvm::AttributeSet & as, TypeConverter & typeConverter)
{
  AttributeSet attributeSet;
  for (auto & attribute : as)
  {
    if (attribute.isEnumAttribute())
    {
      attributeSet.InsertEnumAttribute(ConvertEnumAttribute(attribute));
    }
    else if (attribute.isIntAttribute())
    {
      attributeSet.InsertIntAttribute(ConvertIntAttribute(attribute));
    }
    else if (attribute.isTypeAttribute())
    {
      attributeSet.InsertTypeAttribute(ConvertTypeAttribute(attribute, typeConverter));
    }
    else if (attribute.isStringAttribute())
    {
      attributeSet.InsertStringAttribute(ConvertStringAttribute(attribute));
    }
    else
    {
      JLM_UNREACHABLE("Unhandled attribute");
    }
  }

  return attributeSet;
}

AttributeList
convertAttributeList(
    const ::llvm::AttributeList & attributeList,
    const size_t numParameters,
    TypeConverter & typeConverter)
{
  auto returnAttributes = convert_attributes(attributeList.getRetAttrs(), typeConverter);
  auto functionAttributes = convert_attributes(attributeList.getFnAttrs(), typeConverter);

  std::vector<AttributeSet> parameterAttributes;
  for (size_t n = 0; n < numParameters; n++)
  {
    parameterAttributes.emplace_back(
        convert_attributes(attributeList.getParamAttrs(n), typeConverter));
  }

  return AttributeList(
      std::move(functionAttributes),
      std::move(returnAttributes),
      std::move(parameterAttributes));
}

static const Variable *
convert_int_constant(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<ThreeAddressCode>> & tacs,
    Context &)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::ConstantIntVal);
  const auto constant = ::llvm::cast<const ::llvm::ConstantInt>(c);

  const auto v = convert_apint(constant->getValue());
  tacs.push_back(
      ThreeAddressCode::create(std::make_unique<IntegerConstantOperation>(std::move(v)), {}));

  return tacs.back()->result(0);
}

static inline const Variable *
convert_undefvalue(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    Context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::UndefValueVal);

  auto t = ctx.GetTypeConverter().ConvertLlvmType(*c->getType());
  tacs.push_back(UndefValueOperation::Create(t));

  return tacs.back()->result(0);
}

static const Variable *
convert_constantExpr(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    Context & ctx)
{
  JLM_ASSERT(constant->getValueID() == ::llvm::Value::ConstantExprVal);
  auto c = ::llvm::cast<::llvm::ConstantExpr>(constant);

  /*
    FIXME: ConvertInstruction currently assumes that a instruction's result variable
           is already added to the context. This is not the case for constants and we
           therefore need to do some poilerplate checking in ConvertInstruction to
           see whether a variable was already declared or we need to create a new
           variable.
  */

  /* FIXME: getAsInstruction is none const, forcing all llvm parameters to be none const */
  /* FIXME: The invocation of getAsInstruction() introduces a memory leak. */
  auto instruction = c->getAsInstruction();
  auto v = convertInstruction(instruction, tacs, ctx);
  instruction->dropAllReferences();
  return v;
}

static const Variable *
convert_constantFP(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    Context & ctx)
{
  JLM_ASSERT(constant->getValueID() == ::llvm::Value::ConstantFPVal);
  auto c = ::llvm::cast<::llvm::ConstantFP>(constant);

  auto type = ctx.GetTypeConverter().ConvertLlvmType(*c->getType());
  tacs.push_back(ConstantFP::create(c->getValueAPF(), type));

  return tacs.back()->result(0);
}

static const Variable *
convert_globalVariable(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    Context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::GlobalVariableVal);
  return ConvertValue(c, tacs, ctx);
}

static const Variable *
convert_constantPointerNull(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    Context & ctx)
{
  JLM_ASSERT(::llvm::dyn_cast<const ::llvm::ConstantPointerNull>(constant));
  auto & c = *::llvm::cast<const ::llvm::ConstantPointerNull>(constant);

  auto t = ctx.GetTypeConverter().ConvertPointerType(*c.getType());
  tacs.push_back(ConstantPointerNullOperation::Create(t));

  return tacs.back()->result(0);
}

static const Variable *
convert_blockAddress(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> &,
    Context &)
{
  JLM_ASSERT(constant->getValueID() == ::llvm::Value::BlockAddressVal);

  JLM_UNREACHABLE("Blockaddress constants are not supported.");
}

static const Variable *
convert_constantAggregateZero(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    Context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::ConstantAggregateZeroVal);

  auto type = ctx.GetTypeConverter().ConvertLlvmType(*c->getType());
  tacs.push_back(ConstantAggregateZeroOperation::create(type));

  return tacs.back()->result(0);
}

static const Variable *
convert_constantArray(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    Context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::ConstantArrayVal);

  std::vector<const Variable *> elements;
  for (size_t n = 0; n < c->getNumOperands(); n++)
  {
    auto operand = c->getOperand(n);
    JLM_ASSERT(::llvm::dyn_cast<const ::llvm::Constant>(operand));
    auto constant = ::llvm::cast<::llvm::Constant>(operand);
    elements.push_back(ConvertConstant(constant, tacs, ctx));
  }

  tacs.push_back(ConstantArrayOperation::create(elements));

  return tacs.back()->result(0);
}

static const Variable *
convert_constantDataArray(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    Context & ctx)
{
  JLM_ASSERT(constant->getValueID() == ::llvm::Value::ConstantDataArrayVal);
  const auto & c = *::llvm::cast<const ::llvm::ConstantDataArray>(constant);

  std::vector<const Variable *> elements;
  for (size_t n = 0; n < c.getNumElements(); n++)
    elements.push_back(ConvertConstant(c.getElementAsConstant(n), tacs, ctx));

  tacs.push_back(ConstantDataArray::create(elements));

  return tacs.back()->result(0);
}

static const Variable *
convert_constantDataVector(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    Context & ctx)
{
  JLM_ASSERT(constant->getValueID() == ::llvm::Value::ConstantDataVectorVal);
  auto c = ::llvm::cast<const ::llvm::ConstantDataVector>(constant);

  std::vector<const Variable *> elements;
  for (size_t n = 0; n < c->getNumElements(); n++)
    elements.push_back(ConvertConstant(c->getElementAsConstant(n), tacs, ctx));

  tacs.push_back(ConstantDataVectorOperation::Create(elements));

  return tacs.back()->result(0);
}

static const Variable *
ConvertConstantStruct(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    Context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::ConstantStructVal);

  std::vector<const Variable *> elements;
  for (size_t n = 0; n < c->getNumOperands(); n++)
    elements.push_back(ConvertConstant(c->getAggregateElement(n), tacs, ctx));

  auto type = ctx.GetTypeConverter().ConvertLlvmType(*c->getType());
  tacs.push_back(ConstantStruct::create(elements, type));

  return tacs.back()->result(0);
}

static const Variable *
convert_constantVector(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    Context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::ConstantVectorVal);

  std::vector<const Variable *> elements;
  for (size_t n = 0; n < c->getNumOperands(); n++)
    elements.push_back(ConvertConstant(c->getAggregateElement(n), tacs, ctx));

  auto type = ctx.GetTypeConverter().ConvertLlvmType(*c->getType());
  tacs.push_back(ConstantVectorOperation::create(elements, type));

  return tacs.back()->result(0);
}

static inline const Variable *
convert_globalAlias(
    ::llvm::Constant * constant,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> &,
    Context &)
{
  JLM_ASSERT(constant->getValueID() == ::llvm::Value::GlobalAliasVal);

  JLM_UNREACHABLE("GlobalAlias constants are not supported.");
}

static inline const Variable *
convert_function(::llvm::Constant * c, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(c->getValueID() == ::llvm::Value::FunctionVal);
  return ConvertValue(c, tacs, ctx);
}

static const Variable *
ConvertConstant(
    ::llvm::PoisonValue * poisonValue,
    tacsvector_t & threeAddressCodeVector,
    Context & context)
{
  auto type = context.GetTypeConverter().ConvertLlvmType(*poisonValue->getType());
  threeAddressCodeVector.push_back(PoisonValueOperation::Create(type));

  return threeAddressCodeVector.back()->result(0);
}

template<class T>
static const Variable *
ConvertConstant(
    ::llvm::Constant * constant,
    tacsvector_t & threeAddressCodeVector,
    Context & context)
{
  JLM_ASSERT(::llvm::dyn_cast<T>(constant));
  return ConvertConstant(::llvm::cast<T>(constant), threeAddressCodeVector, context);
}

static const Variable *
ConvertConstant(
    ::llvm::Constant * c,
    std::vector<std::unique_ptr<llvm::ThreeAddressCode>> & tacs,
    Context & ctx)
{
  static std::unordered_map<
      unsigned,
      const Variable * (*)(::llvm::Constant *,
                           std::vector<std::unique_ptr<llvm::ThreeAddressCode>> &,
                           Context & ctx)>
      constantMap({ { ::llvm::Value::BlockAddressVal, convert_blockAddress },
                    { ::llvm::Value::ConstantAggregateZeroVal, convert_constantAggregateZero },
                    { ::llvm::Value::ConstantArrayVal, convert_constantArray },
                    { ::llvm::Value::ConstantDataArrayVal, convert_constantDataArray },
                    { ::llvm::Value::ConstantDataVectorVal, convert_constantDataVector },
                    { ::llvm::Value::ConstantExprVal, convert_constantExpr },
                    { ::llvm::Value::ConstantFPVal, convert_constantFP },
                    { ::llvm::Value::ConstantIntVal, convert_int_constant },
                    { ::llvm::Value::ConstantPointerNullVal, convert_constantPointerNull },
                    { ::llvm::Value::ConstantStructVal, ConvertConstantStruct },
                    { ::llvm::Value::ConstantVectorVal, convert_constantVector },
                    { ::llvm::Value::FunctionVal, convert_function },
                    { ::llvm::Value::GlobalAliasVal, convert_globalAlias },
                    { ::llvm::Value::GlobalVariableVal, convert_globalVariable },
                    { ::llvm::Value::PoisonValueVal, ConvertConstant<::llvm::PoisonValue> },
                    { ::llvm::Value::UndefValueVal, convert_undefvalue } });

  if (constantMap.find(c->getValueID()) != constantMap.end())
    return constantMap[c->getValueID()](c, tacs, ctx);

  JLM_UNREACHABLE("Unsupported LLVM Constant.");
}

static std::vector<std::unique_ptr<llvm::ThreeAddressCode>>
ConvertConstant(::llvm::Constant * c, Context & ctx)
{
  std::vector<std::unique_ptr<llvm::ThreeAddressCode>> tacs;
  ConvertConstant(c, tacs, ctx);
  return tacs;
}

static inline const Variable *
convert_return_instruction(::llvm::Instruction * instruction, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(instruction->getOpcode() == ::llvm::Instruction::Ret);
  auto i = ::llvm::cast<::llvm::ReturnInst>(instruction);

  auto bb = ctx.get(i->getParent());
  bb->add_outedge(bb->cfg().exit());
  if (!i->getReturnValue())
    return {};

  auto value = ConvertValue(i->getReturnValue(), tacs, ctx);
  tacs.push_back(AssignmentOperation::create(value, ctx.result()));

  return ctx.result();
}

static const Variable *
ConvertBranchInstruction(::llvm::Instruction * instruction, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(instruction->getOpcode() == ::llvm::Instruction::Br);
  auto i = ::llvm::cast<::llvm::BranchInst>(instruction);
  auto bb = ctx.get(i->getParent());

  JLM_ASSERT(bb->NumOutEdges() == 0);

  if (i->isUnconditional())
  {
    bb->add_outedge(ctx.get(i->getSuccessor(0)));
    return nullptr;
  }

  bb->add_outedge(ctx.get(i->getSuccessor(1))); // False out-edge
  bb->add_outedge(ctx.get(i->getSuccessor(0))); // True out-edge

  auto c = ConvertValue(i->getCondition(), tacs, ctx);
  auto nbits = i->getCondition()->getType()->getIntegerBitWidth();
  auto op =
      std::unique_ptr<rvsdg::MatchOperation>(new rvsdg::MatchOperation(nbits, { { 1, 1 } }, 0, 2));
  tacs.push_back(ThreeAddressCode::create(std::move(op), { c }));
  tacs.push_back(BranchOperation::create(2, tacs.back()->result(0)));

  return nullptr;
}

static const Variable *
convertSwitchInstruction(::llvm::SwitchInst * switchInstruction, tacsvector_t & tacs, Context & ctx)
{
  auto jlmSwitchBasicBlock = ctx.get(switchInstruction->getParent());

  JLM_ASSERT(jlmSwitchBasicBlock->NumOutEdges() == 0);
  std::unordered_map<uint64_t, uint64_t> matchMapping;
  std::unordered_map<::llvm::BasicBlock *, ControlFlowGraphEdge *> outEdgeMapping;
  for (auto caseIt = switchInstruction->case_begin(); caseIt != switchInstruction->case_end();
       ++caseIt)
  {
    JLM_ASSERT(caseIt != switchInstruction->case_default());
    auto llvmCaseBasicBlock = caseIt->getCaseSuccessor();

    if (auto outEdgeIt = outEdgeMapping.find(llvmCaseBasicBlock); outEdgeIt != outEdgeMapping.end())
    {
      // We have seen this LLVM basic block already and created an outgoing edge for it. Reuse that
      // edge.
      matchMapping[caseIt->getCaseValue()->getZExtValue()] = outEdgeIt->second->index();
    }
    else
    {
      auto jlmCaseBasicBlock = ctx.get(llvmCaseBasicBlock);
      auto edge = jlmSwitchBasicBlock->add_outedge(jlmCaseBasicBlock);
      outEdgeMapping[llvmCaseBasicBlock] = edge;
      matchMapping[caseIt->getCaseValue()->getZExtValue()] = edge->index();
    }
  }

  auto jlmDefaultBasicBlock = ctx.get(switchInstruction->case_default()->getCaseSuccessor());
  auto defaultEdge = jlmSwitchBasicBlock->add_outedge(jlmDefaultBasicBlock);

  auto c = ConvertValue(switchInstruction->getCondition(), tacs, ctx);
  auto numBits = switchInstruction->getCondition()->getType()->getIntegerBitWidth();
  auto op = std::make_unique<rvsdg::MatchOperation>(
      numBits,
      matchMapping,
      defaultEdge->index(),
      jlmSwitchBasicBlock->NumOutEdges());
  tacs.push_back(ThreeAddressCode::create(std::move(op), { c }));
  tacs.push_back(
      BranchOperation::create(jlmSwitchBasicBlock->NumOutEdges(), tacs.back()->result(0)));

  return nullptr;
}

static inline const Variable *
convert_unreachable_instruction(::llvm::Instruction * i, tacsvector_t &, Context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::Unreachable);
  auto bb = ctx.get(i->getParent());
  bb->add_outedge(bb->cfg().exit());
  return nullptr;
}

static std::unique_ptr<rvsdg::BinaryOperation>
ConvertIntegerIcmpPredicate(const ::llvm::CmpInst::Predicate predicate, const std::size_t numBits)
{
  switch (predicate)
  {
  case ::llvm::CmpInst::ICMP_SLT:
    return std::make_unique<IntegerSltOperation>(numBits);
  case ::llvm::CmpInst::ICMP_ULT:
    return std::make_unique<IntegerUltOperation>(numBits);
  case ::llvm::CmpInst::ICMP_SLE:
    return std::make_unique<IntegerSleOperation>(numBits);
  case ::llvm::CmpInst::ICMP_ULE:
    return std::make_unique<IntegerUleOperation>(numBits);
  case ::llvm::CmpInst::ICMP_EQ:
    return std::make_unique<IntegerEqOperation>(numBits);
  case ::llvm::CmpInst::ICMP_NE:
    return std::make_unique<IntegerNeOperation>(numBits);
  case ::llvm::CmpInst::ICMP_SGE:
    return std::make_unique<IntegerSgeOperation>(numBits);
  case ::llvm::CmpInst::ICMP_UGE:
    return std::make_unique<IntegerUgeOperation>(numBits);
  case ::llvm::CmpInst::ICMP_SGT:
    return std::make_unique<IntegerSgtOperation>(numBits);
  case ::llvm::CmpInst::ICMP_UGT:
    return std::make_unique<IntegerUgtOperation>(numBits);
  default:
    JLM_UNREACHABLE("ConvertIntegerIcmpPredicate: Unsupported icmp predicate.");
  }
}

static std::unique_ptr<rvsdg::BinaryOperation>
ConvertPointerIcmpPredicate(const ::llvm::CmpInst::Predicate predicate)
{
  const auto pred = convertICmpPredicateToJlm(predicate);
  return std::make_unique<PtrCmpOperation>(PointerType::Create(), pred);
}

static const Variable *
convert(const ::llvm::ICmpInst * instruction, tacsvector_t & tacs, Context & ctx)
{
  const auto predicate = instruction->getPredicate();
  const auto operandType = instruction->getOperand(0)->getType();
  auto op1 = ConvertValue(instruction->getOperand(0), tacs, ctx);
  auto op2 = ConvertValue(instruction->getOperand(1), tacs, ctx);

  std::unique_ptr<rvsdg::BinaryOperation> operation;
  if (operandType->isVectorTy() && operandType->getScalarType()->isIntegerTy())
  {
    operation =
        ConvertIntegerIcmpPredicate(predicate, operandType->getScalarType()->getIntegerBitWidth());
  }
  else if (operandType->isVectorTy() && operandType->getScalarType()->isPointerTy())
  {
    operation = ConvertPointerIcmpPredicate(predicate);
  }
  else if (operandType->isIntegerTy())
  {
    operation = ConvertIntegerIcmpPredicate(predicate, operandType->getIntegerBitWidth());
  }
  else if (operandType->isPointerTy())
  {
    operation = ConvertPointerIcmpPredicate(predicate);
  }
  else
  {
    JLM_UNREACHABLE("convert: Unhandled icmp type.");
  }

  if (operandType->isVectorTy())
  {
    const auto instructionType = ctx.GetTypeConverter().ConvertLlvmType(*instruction->getType());
    tacs.push_back(VectorBinaryOperation::create(*operation, op1, op2, instructionType));
  }
  else
  {
    tacs.push_back(ThreeAddressCode::create(std::move(operation), { op1, op2 }));
  }

  return tacs.back()->result(0);
}

static const Variable *
convert_fcmp_instruction(::llvm::Instruction * instruction, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(instruction->getOpcode() == ::llvm::Instruction::FCmp);
  auto & typeConverter = ctx.GetTypeConverter();
  auto i = ::llvm::cast<const ::llvm::FCmpInst>(instruction);
  auto t = i->getOperand(0)->getType();

  static std::unordered_map<::llvm::CmpInst::Predicate, llvm::fpcmp> map(
      { { ::llvm::CmpInst::FCMP_TRUE, fpcmp::TRUE },
        { ::llvm::CmpInst::FCMP_FALSE, fpcmp::FALSE },
        { ::llvm::CmpInst::FCMP_OEQ, fpcmp::oeq },
        { ::llvm::CmpInst::FCMP_OGT, fpcmp::ogt },
        { ::llvm::CmpInst::FCMP_OGE, fpcmp::oge },
        { ::llvm::CmpInst::FCMP_OLT, fpcmp::olt },
        { ::llvm::CmpInst::FCMP_OLE, fpcmp::ole },
        { ::llvm::CmpInst::FCMP_ONE, fpcmp::one },
        { ::llvm::CmpInst::FCMP_ORD, fpcmp::ord },
        { ::llvm::CmpInst::FCMP_UNO, fpcmp::uno },
        { ::llvm::CmpInst::FCMP_UEQ, fpcmp::ueq },
        { ::llvm::CmpInst::FCMP_UGT, fpcmp::ugt },
        { ::llvm::CmpInst::FCMP_UGE, fpcmp::uge },
        { ::llvm::CmpInst::FCMP_ULT, fpcmp::ult },
        { ::llvm::CmpInst::FCMP_ULE, fpcmp::ule },
        { ::llvm::CmpInst::FCMP_UNE, fpcmp::une } });

  auto type = typeConverter.ConvertLlvmType(*i->getType());

  auto op1 = ConvertValue(i->getOperand(0), tacs, ctx);
  auto op2 = ConvertValue(i->getOperand(1), tacs, ctx);

  JLM_ASSERT(map.find(i->getPredicate()) != map.end());
  auto fptype = t->isVectorTy() ? t->getScalarType() : t;
  auto operation = std::make_unique<FCmpOperation>(
      map[i->getPredicate()],
      typeConverter.ExtractFloatingPointSize(*fptype));

  if (t->isVectorTy())
    tacs.push_back(VectorBinaryOperation::create(*operation, op1, op2, type));
  else
    tacs.push_back(ThreeAddressCode::create(std::move(operation), { op1, op2 }));

  return tacs.back()->result(0);
}

static const Variable *
AddIOBarrier(tacsvector_t & tacs, const Variable * operand, const Context & ctx)
{
  auto ioBarrierOperation = std::make_unique<IOBarrierOperation>(operand->Type());
  tacs.push_back(
      ThreeAddressCode::create(std::move(ioBarrierOperation), { operand, ctx.iostate() }));
  return tacs.back()->result(0);
}

static inline const Variable *
convert_load_instruction(::llvm::Instruction * i, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::Load);
  auto instruction = static_cast<::llvm::LoadInst *>(i);

  auto alignment = instruction->getAlign().value();
  auto address = ConvertValue(instruction->getPointerOperand(), tacs, ctx);
  auto loadedType = ctx.GetTypeConverter().ConvertLlvmType(*instruction->getType());

  // We currently do not not support atomic load instructions
  JLM_ASSERT(!instruction->isAtomic());

  const ThreeAddressCodeVariable * loadedValue = nullptr;
  const ThreeAddressCodeVariable * memoryState = nullptr;
  const ThreeAddressCodeVariable * ioState = nullptr;
  if (instruction->isVolatile())
  {
    auto loadVolatileTac = LoadVolatileOperation::Create(
        address,
        ctx.iostate(),
        ctx.memory_state(),
        loadedType,
        alignment);
    tacs.push_back(std::move(loadVolatileTac));

    loadedValue = tacs.back()->result(0);
    ioState = tacs.back()->result(1);
    memoryState = tacs.back()->result(2);
  }
  else
  {
    address = AddIOBarrier(tacs, address, ctx);
    auto loadTac =
        LoadNonVolatileOperation::Create(address, ctx.memory_state(), loadedType, alignment);
    tacs.push_back(std::move(loadTac));
    loadedValue = tacs.back()->result(0);
    memoryState = tacs.back()->result(1);
  }

  if (ioState)
  {
    tacs.push_back(AssignmentOperation::create(ioState, ctx.iostate()));
  }
  tacs.push_back(AssignmentOperation::create(memoryState, ctx.memory_state()));

  return loadedValue;
}

static inline const Variable *
convert_store_instruction(::llvm::Instruction * i, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::Store);
  auto instruction = static_cast<::llvm::StoreInst *>(i);

  auto alignment = instruction->getAlign().value();
  auto address = ConvertValue(instruction->getPointerOperand(), tacs, ctx);
  auto value = ConvertValue(instruction->getValueOperand(), tacs, ctx);

  // We currently do not not support atomic store instructions
  JLM_ASSERT(!instruction->isAtomic());

  const ThreeAddressCodeVariable * memoryState = nullptr;
  const ThreeAddressCodeVariable * ioState = nullptr;
  if (instruction->isVolatile())
  {
    auto storeVolatileTac = StoreVolatileOperation::Create(
        address,
        value,
        ctx.iostate(),
        ctx.memory_state(),
        alignment);
    tacs.push_back(std::move(storeVolatileTac));
    ioState = tacs.back()->result(0);
    memoryState = tacs.back()->result(1);
  }
  else
  {
    address = AddIOBarrier(tacs, address, ctx);
    auto storeTac =
        StoreNonVolatileOperation::Create(address, value, ctx.memory_state(), alignment);
    tacs.push_back(std::move(storeTac));
    memoryState = tacs.back()->result(0);
  }

  if (ioState)
  {
    tacs.push_back(AssignmentOperation::create(ioState, ctx.iostate()));
  }
  tacs.push_back(AssignmentOperation::create(memoryState, ctx.memory_state()));

  return nullptr;
}

static const Variable *
ConvertPhiInstruction(::llvm::Instruction * i, tacsvector_t & tacs, Context & ctx)
{
  // Some of the blocks reaching this phi instruction might not be converted yet,
  // so some of the phi's operands may reference instructions that have not yet been converted.
  // For now, a SsaPhiOperation with no operands is created.
  // Once all basic blocks are converted, all SsaPhiOperations are revisited and given operands.
  auto type = ctx.GetTypeConverter().ConvertLlvmType(*i->getType());
  tacs.push_back(SsaPhiOperation::create({}, std::move(type)));
  return tacs.back()->result(0);
}

static const Variable *
convert_getelementptr_instruction(::llvm::Instruction * inst, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(::llvm::dyn_cast<const ::llvm::GetElementPtrInst>(inst));
  auto & typeConverter = ctx.GetTypeConverter();
  auto i = ::llvm::cast<::llvm::GetElementPtrInst>(inst);

  std::vector<const Variable *> indices;
  auto base = ConvertValue(i->getPointerOperand(), tacs, ctx);
  for (auto it = i->idx_begin(); it != i->idx_end(); it++)
    indices.push_back(ConvertValue(*it, tacs, ctx));

  auto pointeeType = typeConverter.ConvertLlvmType(*i->getSourceElementType());
  auto resultType = typeConverter.ConvertLlvmType(*i->getType());

  tacs.push_back(GetElementPtrOperation::Create(base, indices, pointeeType, resultType));

  return tacs.back()->result(0);
}

static const Variable *
convertMallocCall(
    const ::llvm::CallInst & instruction,
    tacsvector_t & threeAddressCodes,
    Context & context)
{
  auto globalMemoryState = context.memory_state();
  const auto globalIOState = context.iostate();

  const auto size = ConvertValue(instruction.getArgOperand(0), threeAddressCodes, context);

  threeAddressCodes.push_back(MallocOperation::createTac(size, globalIOState));
  const auto mallocAddress = threeAddressCodes.back()->result(0);
  const auto mallocIOState = threeAddressCodes.back()->result(1);
  auto mallocMemoryState = threeAddressCodes.back()->result(2);

  threeAddressCodes.push_back(AssignmentOperation::create(mallocIOState, globalIOState));

  threeAddressCodes.push_back(
      MemoryStateMergeOperation::Create({ mallocMemoryState, globalMemoryState }));
  threeAddressCodes.push_back(
      AssignmentOperation::create(threeAddressCodes.back()->result(0), globalMemoryState));

  return mallocAddress;
}

static const Variable *
convertFreeCall(
    const ::llvm::CallInst & instruction,
    tacsvector_t & threeAddressCodes,
    Context & context)
{
  const auto ioState = context.iostate();
  auto memstate = context.memory_state();

  const auto pointer = ConvertValue(instruction.getArgOperand(0), threeAddressCodes, context);

  threeAddressCodes.push_back(FreeOperation::Create(pointer, { memstate }, ioState));
  const auto & freeThreeAddressCode = *threeAddressCodes.back().get();

  threeAddressCodes.push_back(
      AssignmentOperation::create(freeThreeAddressCode.result(0), memstate));
  threeAddressCodes.push_back(AssignmentOperation::create(freeThreeAddressCode.result(1), ioState));

  return nullptr;
}

/**
 * In LLVM, the memcpy intrinsic is modeled as a call instruction. It expects four arguments, with
 * the fourth argument being a ConstantInt of bit width 1 to encode the volatile flag for the memcpy
 * instruction. This function takes this argument and converts it to a boolean flag.
 *
 * @param value The volatile argument of the memcpy intrinsic.
 * @return Boolean flag indicating whether the memcpy is volatile.
 */
static bool
IsVolatile(const ::llvm::Value & value)
{
  const auto constant = ::llvm::dyn_cast<const ::llvm::ConstantInt>(&value);
  JLM_ASSERT(constant != nullptr && constant->getType()->getIntegerBitWidth() == 1);

  const auto apInt = constant->getValue();
  JLM_ASSERT(apInt.isZero() || apInt.isOne());

  return apInt.isOne();
}

static const Variable *
convertMemCpyCall(
    const ::llvm::IntrinsicInst * instruction,
    tacsvector_t & threeAddressCodes,
    Context & context)
{
  const auto ioState = context.iostate();
  auto memoryState = context.memory_state();

  const auto destination = ConvertValue(instruction->getArgOperand(0), threeAddressCodes, context);
  const auto source = ConvertValue(instruction->getArgOperand(1), threeAddressCodes, context);
  const auto length = ConvertValue(instruction->getArgOperand(2), threeAddressCodes, context);

  if (IsVolatile(*instruction->getArgOperand(3)))
  {
    threeAddressCodes.push_back(MemCpyVolatileOperation::CreateThreeAddressCode(
        *destination,
        *source,
        *length,
        *ioState,
        { memoryState }));
    const auto & memCpyVolatileTac = *threeAddressCodes.back();
    threeAddressCodes.push_back(AssignmentOperation::create(memCpyVolatileTac.result(0), ioState));
    threeAddressCodes.push_back(
        AssignmentOperation::create(memCpyVolatileTac.result(1), memoryState));
  }
  else
  {
    threeAddressCodes.push_back(
        MemCpyNonVolatileOperation::create(destination, source, length, { memoryState }));
    threeAddressCodes.push_back(
        AssignmentOperation::create(threeAddressCodes.back()->result(0), memoryState));
  }

  return nullptr;
}

static bool
isMallocCall(const ::llvm::CallInst & callInstruction)
{
  const auto function = callInstruction.getCalledFunction();
  return function && function->getName() == "malloc";
}

static bool
isFreeCall(const ::llvm::CallInst & callInstruction)
{
  const auto function = callInstruction.getCalledFunction();
  return function && function->getName() == "free";
}

static const Variable *
convertFMulAddIntrinsic(const ::llvm::CallInst & instruction, tacsvector_t & tacs, Context & ctx)
{
  const auto multiplier = ConvertValue(instruction.getArgOperand(0), tacs, ctx);
  const auto multiplicand = ConvertValue(instruction.getArgOperand(1), tacs, ctx);
  const auto summand = ConvertValue(instruction.getArgOperand(2), tacs, ctx);
  tacs.push_back(FMulAddIntrinsicOperation::CreateTac(*multiplier, *multiplicand, *summand));

  return tacs.back()->result(0);
}

std::vector<const Variable *>
convertCallArguments(
    const ::llvm::CallInst & callInstruction,
    tacsvector_t & threeAddressCodes,
    Context & context)
{
  const auto functionType = callInstruction.getFunctionType();

  std::vector<const Variable *> arguments;
  for (size_t n = 0; n < functionType->getNumParams(); n++)
    arguments.push_back(ConvertValue(callInstruction.getArgOperand(n), threeAddressCodes, context));

  if (functionType->isVarArg())
  {
    std::vector<const Variable *> variableArguments;
    for (size_t n = functionType->getNumParams(); n < callInstruction.getNumOperands() - 1; n++)
      variableArguments.push_back(
          ConvertValue(callInstruction.getArgOperand(n), threeAddressCodes, context));

    threeAddressCodes.push_back(VariadicArgumentListOperation::create(variableArguments));
    arguments.push_back(threeAddressCodes.back()->result(0));
  }

  arguments.push_back(context.iostate());
  arguments.push_back(context.memory_state());

  return arguments;
}

static const Variable *
createCall(
    const ::llvm::CallInst & callInstruction,
    tacsvector_t & threeAddressCodes,
    Context & context)
{
  const auto functionType = callInstruction.getFunctionType();

  auto convertedFunctionType = context.GetTypeConverter().ConvertFunctionType(*functionType);
  const auto arguments = convertCallArguments(callInstruction, threeAddressCodes, context);
  const auto callingConvention = convertCallingConventionToJlm(callInstruction.getCallingConv());
  auto attributes = convertAttributeList(
      callInstruction.getAttributes(),
      callInstruction.arg_size(),
      context.GetTypeConverter());

  const Variable * callee =
      ConvertValueOrFunction(callInstruction.getCalledOperand(), threeAddressCodes, context);
  // Llvm does not distinguish between "function objects" and
  // "pointers to functions" while we need to be precise in modeling.
  // If the called object is a function object, then we can just
  // feed it to the call operator directly, otherwise we have
  // to cast it into a function object.
  if (is<PointerType>(*callee->Type()))
  {
    std::unique_ptr<ThreeAddressCode> callee_cast = ThreeAddressCode::create(
        std::make_unique<PointerToFunctionOperation>(convertedFunctionType),
        { callee });
    callee = callee_cast->result(0);
    threeAddressCodes.push_back(std::move(callee_cast));
  }
  else if (auto fntype = std::dynamic_pointer_cast<const rvsdg::FunctionType>(callee->Type()))
  {
    // Llvm also allows argument type mismatches if the function
    // features varargs. The code here could be made more precise by
    // validating and accepting only vararg-related mismatches.
    if (*convertedFunctionType != *fntype)
    {
      // Since vararg passing is not modeled explicitly, simply hide the
      // argument mismatch via pointer casts.
      std::unique_ptr<ThreeAddressCode> ptrCast = ThreeAddressCode::create(
          std::make_unique<FunctionToPointerOperation>(fntype),
          { callee });
      std::unique_ptr<ThreeAddressCode> fnCast = ThreeAddressCode::create(
          std::make_unique<PointerToFunctionOperation>(convertedFunctionType),
          { ptrCast->result(0) });
      callee = fnCast->result(0);
      threeAddressCodes.push_back(std::move(ptrCast));
      threeAddressCodes.push_back(std::move(fnCast));
    }
  }
  else
  {
    throw std::runtime_error("Unexpected callee type: " + callee->Type()->debug_string());
  }

  auto call = CallOperation::create(
      callee,
      convertedFunctionType,
      callingConvention,
      std::move(attributes),
      arguments);

  const auto result = call->result(0);
  const auto ioState = call->result(call->nresults() - 2);
  const auto memoryState = call->result(call->nresults() - 1);

  threeAddressCodes.push_back(std::move(call));
  threeAddressCodes.push_back(AssignmentOperation::create(ioState, context.iostate()));
  threeAddressCodes.push_back(AssignmentOperation::create(memoryState, context.memory_state()));

  return result;
}

/**
 * Checks if the intrinsic with the given ID should be ignored in the frontend.
 * Calls to ignored intrinsics become no-ops, and declarations of ignored intrisincs are skipped.
 * @param intrinsicId the id of the llvm intrinsic
 * @return true if the intrinsic should be ignored, false otherwise
 */
static bool
isIntrinsicIgnored(::llvm::Intrinsic::ID intrinsicId)
{
  switch (intrinsicId)
  {
  // These intrinsics are ignored because they take pointers to local variables,
  // reducing the precision of alias analysis unless specifically handled
  case ::llvm::Intrinsic::lifetime_start:
  case ::llvm::Intrinsic::lifetime_end:
  // This intrinsic is ignored because it takes a parameter of type "metadata"
  case ::llvm::Intrinsic::experimental_noalias_scope_decl:
    return true;
  default:
    return false;
  }
}

static const Variable *
convertIntrinsicInstruction(
    const ::llvm::IntrinsicInst & intrinsicInstruction,
    tacsvector_t & threeAddressCodes,
    Context & context)
{
  if (isIntrinsicIgnored(intrinsicInstruction.getIntrinsicID()))
    return nullptr;

  switch (intrinsicInstruction.getIntrinsicID())
  {
  case ::llvm::Intrinsic::fmuladd:
    return convertFMulAddIntrinsic(intrinsicInstruction, threeAddressCodes, context);
  case ::llvm::Intrinsic::memcpy:
    return convertMemCpyCall(&intrinsicInstruction, threeAddressCodes, context);
  default:
    return createCall(intrinsicInstruction, threeAddressCodes, context);
  }
}

static const Variable *
convertCallInstruction(
    const ::llvm::CallInst & callInstruction,
    tacsvector_t & threeAddressCodes,
    Context & context)
{
  if (const auto intrinsicInstruction = ::llvm::dyn_cast<::llvm::IntrinsicInst>(&callInstruction))
    return convertIntrinsicInstruction(*intrinsicInstruction, threeAddressCodes, context);

  if (isMallocCall(callInstruction))
    return convertMallocCall(callInstruction, threeAddressCodes, context);

  if (isFreeCall(callInstruction))
    return convertFreeCall(callInstruction, threeAddressCodes, context);

  return createCall(callInstruction, threeAddressCodes, context);
}

static inline const Variable *
convert_select_instruction(::llvm::Instruction * i, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::Select);
  auto instruction = static_cast<::llvm::SelectInst *>(i);

  auto p = ConvertValue(instruction->getCondition(), tacs, ctx);
  auto t = ConvertValue(instruction->getTrueValue(), tacs, ctx);
  auto f = ConvertValue(instruction->getFalseValue(), tacs, ctx);

  if (i->getType()->isVectorTy())
    tacs.push_back(VectorSelectOperation::create(p, t, f));
  else
    tacs.push_back(SelectOperation::create(p, t, f));

  return tacs.back()->result(0);
}

static std::unique_ptr<rvsdg::BinaryOperation>
ConvertIntegerBinaryOperation(
    const ::llvm::Instruction::BinaryOps binaryOperation,
    std::size_t numBits)
{
  switch (binaryOperation)
  {
  case ::llvm::Instruction::Add:
    return std::make_unique<IntegerAddOperation>(numBits);
  case ::llvm::Instruction::And:
    return std::make_unique<IntegerAndOperation>(numBits);
  case ::llvm::Instruction::AShr:
    return std::make_unique<IntegerAShrOperation>(numBits);
  case ::llvm::Instruction::LShr:
    return std::make_unique<IntegerLShrOperation>(numBits);
  case ::llvm::Instruction::Mul:
    return std::make_unique<IntegerMulOperation>(numBits);
  case ::llvm::Instruction::Or:
    return std::make_unique<IntegerOrOperation>(numBits);
  case ::llvm::Instruction::SDiv:
    return std::make_unique<IntegerSDivOperation>(numBits);
  case ::llvm::Instruction::Shl:
    return std::make_unique<IntegerShlOperation>(numBits);
  case ::llvm::Instruction::SRem:
    return std::make_unique<IntegerSRemOperation>(numBits);
  case ::llvm::Instruction::Sub:
    return std::make_unique<IntegerSubOperation>(numBits);
  case ::llvm::Instruction::UDiv:
    return std::make_unique<IntegerUDivOperation>(numBits);
  case ::llvm::Instruction::URem:
    return std::make_unique<IntegerURemOperation>(numBits);
  case ::llvm::Instruction::Xor:
    return std::make_unique<IntegerXorOperation>(numBits);
  default:
    JLM_UNREACHABLE("ConvertIntegerBinaryOperation: Unsupported integer binary operation");
  }
}

static std::unique_ptr<rvsdg::BinaryOperation>
ConvertFloatingPointBinaryOperation(
    const ::llvm::Instruction::BinaryOps binaryOperation,
    fpsize floatingPointSize)
{
  switch (binaryOperation)
  {
  case ::llvm::Instruction::FAdd:
    return std::make_unique<FBinaryOperation>(fpop::add, floatingPointSize);
  case ::llvm::Instruction::FSub:
    return std::make_unique<FBinaryOperation>(fpop::sub, floatingPointSize);
  case ::llvm::Instruction::FMul:
    return std::make_unique<FBinaryOperation>(fpop::mul, floatingPointSize);
  case ::llvm::Instruction::FDiv:
    return std::make_unique<FBinaryOperation>(fpop::div, floatingPointSize);
  case ::llvm::Instruction::FRem:
    return std::make_unique<FBinaryOperation>(fpop::mod, floatingPointSize);
  default:
    JLM_UNREACHABLE("ConvertFloatingPointBinaryOperation: Unsupported binary operation");
  }
}

static const Variable *
convert(const ::llvm::BinaryOperator * instruction, tacsvector_t & tacs, Context & ctx)
{
  const auto llvmType = instruction->getType();
  auto & typeConverter = ctx.GetTypeConverter();
  const auto opcode = instruction->getOpcode();

  std::unique_ptr<rvsdg::BinaryOperation> operation;
  if (llvmType->isVectorTy() && llvmType->getScalarType()->isIntegerTy())
  {
    const auto numBits = llvmType->getScalarType()->getIntegerBitWidth();
    operation = ConvertIntegerBinaryOperation(opcode, numBits);
  }
  else if (llvmType->isVectorTy() && llvmType->getScalarType()->isFloatingPointTy())
  {
    const auto size = typeConverter.ExtractFloatingPointSize(*llvmType->getScalarType());
    operation = ConvertFloatingPointBinaryOperation(opcode, size);
  }
  else if (llvmType->isIntegerTy())
  {
    operation = ConvertIntegerBinaryOperation(opcode, llvmType->getIntegerBitWidth());
  }
  else if (llvmType->isFloatingPointTy())
  {
    const auto size = typeConverter.ExtractFloatingPointSize(*llvmType);
    operation = ConvertFloatingPointBinaryOperation(opcode, size);
  }
  else
  {
    JLM_ASSERT("convert: Unhandled binary operation type.");
  }

  const auto jlmType = typeConverter.ConvertLlvmType(*llvmType);
  auto operand1 = ConvertValue(instruction->getOperand(0), tacs, ctx);
  auto operand2 = ConvertValue(instruction->getOperand(1), tacs, ctx);

  if (instruction->getOpcode() == ::llvm::Instruction::SDiv
      || instruction->getOpcode() == ::llvm::Instruction::UDiv
      || instruction->getOpcode() == ::llvm::Instruction::SRem
      || instruction->getOpcode() == ::llvm::Instruction::URem)
  {
    operand1 = AddIOBarrier(tacs, operand1, ctx);
  }

  if (llvmType->isVectorTy())
  {
    tacs.push_back(VectorBinaryOperation::create(*operation, operand1, operand2, jlmType));
  }
  else
  {
    tacs.push_back(ThreeAddressCode::create(std::move(operation), { operand1, operand2 }));
  }

  return tacs.back()->result(0);
}

static inline const Variable *
convert_alloca_instruction(::llvm::Instruction * instruction, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(instruction->getOpcode() == ::llvm::Instruction::Alloca);
  auto i = static_cast<::llvm::AllocaInst *>(instruction);

  auto memstate = ctx.memory_state();
  auto size = ConvertValue(i->getArraySize(), tacs, ctx);
  auto vtype = ctx.GetTypeConverter().ConvertLlvmType(*i->getAllocatedType());
  auto alignment = i->getAlign().value();

  tacs.push_back(AllocaOperation::create(vtype, size, alignment));
  auto result = tacs.back()->result(0);
  auto astate = tacs.back()->result(1);

  tacs.push_back(MemoryStateMergeOperation::Create({ astate, memstate }));
  tacs.push_back(AssignmentOperation::create(tacs.back()->result(0), memstate));

  return result;
}

static const Variable *
convert_extractvalue(::llvm::Instruction * i, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::ExtractValue);
  auto ev = ::llvm::dyn_cast<::llvm::ExtractValueInst>(i);

  auto aggregate = ConvertValue(ev->getOperand(0), tacs, ctx);
  tacs.push_back(ExtractValueOperation::create(aggregate, ev->getIndices()));

  return tacs.back()->result(0);
}

static const Variable *
convertInsertValueInstruction(
    const ::llvm::InsertValueInst & instruction,
    tacsvector_t & tacs,
    Context & context)
{
  const auto aggregateOperand = ConvertValue(instruction.getOperand(0), tacs, context);
  const auto valueOperand = ConvertValue(instruction.getOperand(1), tacs, context);

  tacs.push_back(
      InsertValueOperation::createTac(*aggregateOperand, *valueOperand, instruction.getIndices()));

  return tacs.back()->result(0);
}

static inline const Variable *
convert_extractelement_instruction(::llvm::Instruction * i, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::ExtractElement);

  auto vector = ConvertValue(i->getOperand(0), tacs, ctx);
  auto index = ConvertValue(i->getOperand(1), tacs, ctx);
  tacs.push_back(ExtractElementOperation::create(vector, index));

  return tacs.back()->result(0);
}

static const Variable *
convert(::llvm::ShuffleVectorInst * i, tacsvector_t & tacs, Context & ctx)
{
  auto v1 = ConvertValue(i->getOperand(0), tacs, ctx);
  auto v2 = ConvertValue(i->getOperand(1), tacs, ctx);

  std::vector<int> mask;
  for (auto & element : i->getShuffleMask())
    mask.push_back(element);

  tacs.push_back(ShuffleVectorOperation::create(v1, v2, mask));

  return tacs.back()->result(0);
}

static const Variable *
convert_insertelement_instruction(::llvm::Instruction * i, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::InsertElement);

  auto vector = ConvertValue(i->getOperand(0), tacs, ctx);
  auto value = ConvertValue(i->getOperand(1), tacs, ctx);
  auto index = ConvertValue(i->getOperand(2), tacs, ctx);
  tacs.push_back(InsertElementOperation::create(vector, value, index));

  return tacs.back()->result(0);
}

static const Variable *
convertFreezeInstruction(::llvm::Instruction * i, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(i->getOpcode() == ::llvm::Instruction::Freeze);

  auto operand = ConvertValue(i->getOperand(0), tacs, ctx);
  tacs.push_back(FreezeOperation::createTac(*operand));

  return tacs.back()->result(0);
}

static const Variable *
convert(::llvm::UnaryOperator * unaryOperator, tacsvector_t & threeAddressCodeVector, Context & ctx)
{
  JLM_ASSERT(unaryOperator->getOpcode() == ::llvm::Instruction::FNeg);
  auto & typeConverter = ctx.GetTypeConverter();

  auto type = unaryOperator->getType();
  auto scalarType = typeConverter.ConvertLlvmType(*type->getScalarType());
  auto operand = ConvertValue(unaryOperator->getOperand(0), threeAddressCodeVector, ctx);

  if (type->isVectorTy())
  {
    auto vectorType = typeConverter.ConvertLlvmType(*type);
    threeAddressCodeVector.push_back(VectorUnaryOperation::create(
        FNegOperation(std::static_pointer_cast<const FloatingPointType>(scalarType)),
        operand,
        vectorType));
  }
  else
  {
    threeAddressCodeVector.push_back(FNegOperation::create(operand));
  }

  return threeAddressCodeVector.back()->result(0);
}

template<class OP>
static std::unique_ptr<rvsdg::SimpleOperation>
create_unop(std::shared_ptr<const rvsdg::Type> st, std::shared_ptr<const rvsdg::Type> dt)
{
  return std::unique_ptr<rvsdg::SimpleOperation>(new OP(std::move(st), std::move(dt)));
}

static const Variable *
convert_cast_instruction(::llvm::Instruction * i, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(::llvm::dyn_cast<::llvm::CastInst>(i));
  auto & typeConverter = ctx.GetTypeConverter();
  auto st = i->getOperand(0)->getType();
  auto dt = i->getType();

  // Most cast operations can be performed on vector types, with the cast done per value.
  // Bit casts are an exception, as they can cast between different vector sizes.
  bool isLaneWiseCast = false;
  if (i->getOpcode() != ::llvm::Instruction::BitCast)
  {
    isLaneWiseCast = st->isVectorTy();
    JLM_ASSERT(st->isVectorTy() == dt->isVectorTy());
  }

  static std::unordered_map<
      unsigned,
      std::unique_ptr<rvsdg::SimpleOperation> (*)(
          std::shared_ptr<const rvsdg::Type>,
          std::shared_ptr<const rvsdg::Type>)>
      map({ { ::llvm::Instruction::Trunc, create_unop<TruncOperation> },
            { ::llvm::Instruction::ZExt, create_unop<ZExtOperation> },
            { ::llvm::Instruction::UIToFP, create_unop<UIToFPOperation> },
            { ::llvm::Instruction::SIToFP, create_unop<SIToFPOperation> },
            { ::llvm::Instruction::SExt, create_unop<SExtOperation> },
            { ::llvm::Instruction::PtrToInt, create_unop<PtrToIntOperation> },
            { ::llvm::Instruction::IntToPtr, create_unop<IntegerToPointerOperation> },
            { ::llvm::Instruction::FPTrunc, create_unop<FPTruncOperation> },
            { ::llvm::Instruction::FPToSI, create_unop<FloatingPointToSignedIntegerOperation> },
            { ::llvm::Instruction::FPToUI, create_unop<FloatingPointToUnsignedIntegerOperation> },
            { ::llvm::Instruction::FPExt, create_unop<FPExtOperation> },
            { ::llvm::Instruction::BitCast, create_unop<BitCastOperation> } });

  auto type = ctx.GetTypeConverter().ConvertLlvmType(*i->getType());

  auto op = ConvertValue(i->getOperand(0), tacs, ctx);
  auto srctype = typeConverter.ConvertLlvmType(*(isLaneWiseCast ? st->getScalarType() : st));
  auto dsttype = typeConverter.ConvertLlvmType(*(isLaneWiseCast ? dt->getScalarType() : dt));

  JLM_ASSERT(map.find(i->getOpcode()) != map.end());
  auto unop = map[i->getOpcode()](std::move(srctype), std::move(dsttype));
  JLM_ASSERT(is<rvsdg::UnaryOperation>(*unop));

  if (isLaneWiseCast)
    tacs.push_back(
        VectorUnaryOperation::create(*static_cast<rvsdg::UnaryOperation *>(unop.get()), op, type));
  else
    tacs.push_back(ThreeAddressCode::create(std::move(unop), { op }));

  return tacs.back()->result(0);
}

template<class INSTRUCTIONTYPE>
static const Variable *
convert(::llvm::Instruction * instruction, tacsvector_t & tacs, Context & ctx)
{
  JLM_ASSERT(::llvm::isa<INSTRUCTIONTYPE>(instruction));
  return convert(::llvm::cast<INSTRUCTIONTYPE>(instruction), tacs, ctx);
}

static const Variable *
convertInstruction(
    ::llvm::Instruction * instruction,
    std::vector<std::unique_ptr<ThreeAddressCode>> & threeAddressCodes,
    Context & context)
{
  switch (instruction->getOpcode())
  {
  case ::llvm::Instruction::Trunc:
  case ::llvm::Instruction::ZExt:
  case ::llvm::Instruction::UIToFP:
  case ::llvm::Instruction::SIToFP:
  case ::llvm::Instruction::SExt:
  case ::llvm::Instruction::PtrToInt:
  case ::llvm::Instruction::IntToPtr:
  case ::llvm::Instruction::FPTrunc:
  case ::llvm::Instruction::FPToSI:
  case ::llvm::Instruction::FPToUI:
  case ::llvm::Instruction::FPExt:
  case ::llvm::Instruction::BitCast:
    return convert_cast_instruction(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::Add:
  case ::llvm::Instruction::And:
  case ::llvm::Instruction::AShr:
  case ::llvm::Instruction::Sub:
  case ::llvm::Instruction::UDiv:
  case ::llvm::Instruction::SDiv:
  case ::llvm::Instruction::URem:
  case ::llvm::Instruction::SRem:
  case ::llvm::Instruction::Shl:
  case ::llvm::Instruction::LShr:
  case ::llvm::Instruction::Or:
  case ::llvm::Instruction::Xor:
  case ::llvm::Instruction::Mul:
  case ::llvm::Instruction::FAdd:
  case ::llvm::Instruction::FSub:
  case ::llvm::Instruction::FMul:
  case ::llvm::Instruction::FDiv:
  case ::llvm::Instruction::FRem:
    return convert<::llvm::BinaryOperator>(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::Ret:
    return convert_return_instruction(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::Br:
    return ConvertBranchInstruction(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::Switch:
    return convertSwitchInstruction(
        ::llvm::cast<::llvm::SwitchInst>(instruction),
        threeAddressCodes,
        context);
  case ::llvm::Instruction::Unreachable:
    return convert_unreachable_instruction(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::FNeg:
    return convert<::llvm::UnaryOperator>(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::ICmp:
    return convert<::llvm::ICmpInst>(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::FCmp:
    return convert_fcmp_instruction(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::Load:
    return convert_load_instruction(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::Store:
    return convert_store_instruction(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::PHI:
    return ConvertPhiInstruction(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::GetElementPtr:
    return convert_getelementptr_instruction(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::Call:
    return convertCallInstruction(
        *::llvm::dyn_cast<::llvm::CallInst>(instruction),
        threeAddressCodes,
        context);
  case ::llvm::Instruction::Select:
    return convert_select_instruction(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::Alloca:
    return convert_alloca_instruction(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::ExtractValue:
    return convert_extractvalue(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::InsertValue:
    return convertInsertValueInstruction(
        *::llvm::dyn_cast<::llvm::InsertValueInst>(instruction),
        threeAddressCodes,
        context);
  case ::llvm::Instruction::ExtractElement:
    return convert_extractelement_instruction(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::ShuffleVector:
    return convert<::llvm::ShuffleVectorInst>(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::InsertElement:
    return convert_insertelement_instruction(instruction, threeAddressCodes, context);
  case ::llvm::Instruction::Freeze:
    return convertFreezeInstruction(instruction, threeAddressCodes, context);
  default:
    throw std::runtime_error(util::strfmt(instruction->getOpcodeName(), " is not supported."));
  }
}

static std::vector<::llvm::PHINode *>
convert_instructions(::llvm::Function & function, Context & ctx)
{
  std::vector<::llvm::PHINode *> phis;
  ::llvm::ReversePostOrderTraversal<::llvm::Function *> rpotraverser(&function);
  for (auto & bb : rpotraverser)
  {
    for (auto & instruction : *bb)
    {
      tacsvector_t tacs;
      if (auto result = convertInstruction(&instruction, tacs, ctx))
        ctx.insert_value(&instruction, result);

      // When an LLVM PhiNode is converted to a jlm SsaPhiOperation, some of its operands may not be
      // ready. The created SsaPhiOperation therefore has no operands, but is instead added to a
      // list. Once all basic blocks have been converted, all SsaPhiOperations are revisited and
      // given operands.
      if (!tacs.empty() && is<SsaPhiOperation>(tacs.back()->operation()))
      {
        auto phi = ::llvm::dyn_cast<::llvm::PHINode>(&instruction);
        phis.push_back(phi);
      }

      ctx.get(bb)->append_last(tacs);
    }
  }

  return phis;
}

/**
 * During conversion of LLVM instructions, phi instructions were created without operands.
 * Once all instructions have been converted, this function goes over all phi instructions
 * and assigns proper operands.
 */
static void
PatchPhiOperands(const std::vector<::llvm::PHINode *> & phis, Context & ctx)
{
  for (const auto & phi : phis)
  {
    std::vector<ControlFlowGraphNode *> incomingNodes;
    std::vector<const Variable *> operands;
    for (size_t n = 0; n < phi->getNumOperands(); n++)
    {
      // In LLVM, phi instructions may have incoming basic blocks that are unreachable.
      // These are not visited during convert_basic_blocks, and thus do not have corresponding
      // jlm::llvm::basic_blocks. The SsaPhiOperation can safely ignore these, as they are dead.
      if (!ctx.has(phi->getIncomingBlock(n)))
        continue;

      // The LLVM phi instruction may have multiple operands with the same incoming cfg node.
      // When this happens in valid LLVM IR, all operands from the same basic block are identical.
      // We therefore skip any operands that reference already handled basic blocks.
      auto predecessor = ctx.get(phi->getIncomingBlock(n));
      if (std::find(incomingNodes.begin(), incomingNodes.end(), predecessor) != incomingNodes.end())
        continue;

      // Convert the operand value in the predecessor basic block, as that is where it is "used".
      tacsvector_t tacs;
      operands.push_back(ConvertValue(phi->getIncomingValue(n), tacs, ctx));
      predecessor->insert_before_branch(tacs);
      incomingNodes.push_back(predecessor);
    }

    JLM_ASSERT(operands.size() >= 1);

    auto phi_tac = util::assertedCast<const ThreeAddressCodeVariable>(ctx.lookup_value(phi))->tac();
    phi_tac->replace(
        SsaPhiOperation(std::move(incomingNodes), phi_tac->result(0)->Type()),
        operands);
  }
}

static BasicBlockMap
convert_basic_blocks(::llvm::Function & f, ControlFlowGraph & cfg)
{
  BasicBlockMap bbmap;
  ::llvm::ReversePostOrderTraversal<::llvm::Function *> rpotraverser(&f);
  for (auto & bb : rpotraverser)
    bbmap.Insert(bb, BasicBlock::create(cfg));

  return bbmap;
}

static std::unique_ptr<llvm::Argument>
convert_argument(const ::llvm::Argument & argument, Context & ctx)
{
  auto function = argument.getParent();
  auto name = argument.getName().str();
  auto type = ctx.GetTypeConverter().ConvertLlvmType(*argument.getType());
  auto attributes = convert_attributes(
      function->getAttributes().getParamAttrs(argument.getArgNo()),
      ctx.GetTypeConverter());

  return llvm::Argument::create(name, type, attributes);
}

static void
EnsureSingleInEdgeToExitNode(ControlFlowGraph & cfg)
{
  auto exitNode = cfg.exit();

  if (exitNode->NumInEdges() == 0)
  {
    /*
      LLVM can produce CFGs that have no incoming edge to the exit node. This can happen if
      endless loops are present in the code. For example, this code

      \code{.cpp}
        int foo()
        {
          while (1) {
            printf("foo\n");
          }

          return 0;
        }
      \endcode

      results in a JLM CFG with no incoming edge to the exit node.

      We solve this problem by finding the first SCC with no exit edge, i.e., an endless loop, and
      restructure it to an SCC with an exit edge to the CFG's exit node.
    */
    auto stronglyConnectedComponents = find_sccs(cfg);
    for (auto stronglyConnectedComponent : stronglyConnectedComponents)
    {
      auto sccStructure = StronglyConnectedComponentStructure::Create(stronglyConnectedComponent);

      if (sccStructure->NumExitEdges() == 0)
      {
        auto repetitionEdge = *sccStructure->RepetitionEdges().begin();

        auto basicBlock = BasicBlock::create(cfg);

        auto op = std::make_unique<rvsdg::ControlConstantOperation>(
            rvsdg::ControlValueRepresentation(1, 2));
        auto operand =
            basicBlock->append_last(ThreeAddressCode::create(std::move(op), {}))->result(0);
        basicBlock->append_last(BranchOperation::create(2, operand));

        basicBlock->add_outedge(exitNode);
        basicBlock->add_outedge(repetitionEdge->sink());

        repetitionEdge->divert(basicBlock);
        break;
      }
    }
  }

  if (exitNode->NumInEdges() == 1)
    return;

  /*
    We have multiple incoming edges to the exit node. Insert an empty basic block, divert all
    incoming edges to this block, and add an outgoing edge from this block to the exit node.
  */
  auto basicBlock = BasicBlock::create(cfg);
  exitNode->divert_inedges(basicBlock);
  basicBlock->add_outedge(exitNode);
}

static std::unique_ptr<ControlFlowGraph>
create_cfg(::llvm::Function & f, Context & ctx)
{
  auto node = static_cast<const FunctionVariable *>(ctx.lookup_value(&f))->function();

  auto add_arguments = [](const ::llvm::Function & f, ControlFlowGraph & cfg, Context & ctx)
  {
    auto node = static_cast<const FunctionVariable *>(ctx.lookup_value(&f))->function();

    size_t n = 0;
    for (const auto & arg : f.args())
    {
      auto argument = cfg.entry()->append_argument(convert_argument(arg, ctx));
      ctx.insert_value(&arg, argument);
      n++;
    }

    if (f.isVarArg())
    {
      JLM_ASSERT(n < node->fcttype().NumArguments());
      auto & type = node->fcttype().Arguments()[n++];
      cfg.entry()->append_argument(Argument::create("_varg_", type));
    }
    JLM_ASSERT(n < node->fcttype().NumArguments());

    auto & iotype = node->fcttype().Arguments()[n++];
    auto iostate = cfg.entry()->append_argument(Argument::create("_io_", iotype));

    auto & memtype = node->fcttype().Arguments()[n++];
    auto memstate = cfg.entry()->append_argument(Argument::create("_s_", memtype));

    JLM_ASSERT(n == node->fcttype().NumArguments());
    ctx.set_iostate(iostate);
    ctx.set_memory_state(memstate);
  };

  auto cfg = ControlFlowGraph::create(ctx.module());

  add_arguments(f, *cfg, ctx);
  auto bbmap = convert_basic_blocks(f, *cfg);

  /* create entry block */
  auto entry_block = BasicBlock::create(*cfg);
  cfg->exit()->divert_inedges(entry_block);
  entry_block->add_outedge(bbmap.LookupKey(&f.getEntryBlock()));

  /* add results */
  const ThreeAddressCodeVariable * result = nullptr;
  if (!f.getReturnType()->isVoidTy())
  {
    auto type = ctx.GetTypeConverter().ConvertLlvmType(*f.getReturnType());
    entry_block->append_last(UndefValueOperation::Create(type, "_r_"));
    result = entry_block->last()->result(0);

    JLM_ASSERT(node->fcttype().NumResults() == 3);
    JLM_ASSERT(result->type() == node->fcttype().ResultType(0));
    cfg->exit()->append_result(result);
  }
  cfg->exit()->append_result(ctx.iostate());
  cfg->exit()->append_result(ctx.memory_state());

  /* convert instructions */
  ctx.set_basic_block_map(std::move(bbmap));
  ctx.set_result(result);
  auto phis = convert_instructions(f, ctx);
  PatchPhiOperands(phis, ctx);

  EnsureSingleInEdgeToExitNode(*cfg);

  // Merge basic blocks A -> B when possible
  straighten(*cfg);
  // Remove unreachable nodes
  prune(*cfg);
  return cfg;
}

static void
convert_function(::llvm::Function & function, Context & ctx)
{
  if (function.isDeclaration())
    return;

  auto fv = static_cast<const FunctionVariable *>(ctx.lookup_value(&function));

  ctx.set_node(fv->function());
  fv->function()->add_cfg(create_cfg(function, ctx));
  ctx.set_node(nullptr);
}

static const llvm::Linkage &
convert_linkage(const ::llvm::GlobalValue::LinkageTypes & linkage)
{
  static std::unordered_map<::llvm::GlobalValue::LinkageTypes, llvm::Linkage> map(
      { { ::llvm::GlobalValue::ExternalLinkage, llvm::Linkage::externalLinkage },
        { ::llvm::GlobalValue::AvailableExternallyLinkage,
          llvm::Linkage::availableExternallyLinkage },
        { ::llvm::GlobalValue::LinkOnceAnyLinkage, llvm::Linkage::linkOnceAnyLinkage },
        { ::llvm::GlobalValue::LinkOnceODRLinkage, llvm::Linkage::linkOnceOdrLinkage },
        { ::llvm::GlobalValue::WeakAnyLinkage, llvm::Linkage::weakAnyLinkage },
        { ::llvm::GlobalValue::WeakODRLinkage, llvm::Linkage::weakOdrLinkage },
        { ::llvm::GlobalValue::AppendingLinkage, llvm::Linkage::appendingLinkage },
        { ::llvm::GlobalValue::InternalLinkage, llvm::Linkage::internalLinkage },
        { ::llvm::GlobalValue::PrivateLinkage, llvm::Linkage::privateLinkage },
        { ::llvm::GlobalValue::ExternalWeakLinkage, llvm::Linkage::externalWeakLinkage },
        { ::llvm::GlobalValue::CommonLinkage, llvm::Linkage::commonLinkage } });

  JLM_ASSERT(map.find(linkage) != map.end());
  return map[linkage];
}

static void
declare_globals(::llvm::Module & lm, Context & ctx)
{
  auto create_data_node = [](const ::llvm::GlobalVariable & gv, Context & ctx)
  {
    auto name = gv.getName().str();
    auto constant = gv.isConstant();
    auto type = ctx.GetTypeConverter().ConvertLlvmType(*gv.getValueType());
    auto linkage = convert_linkage(gv.getLinkage());
    auto section = gv.getSection().str();
    const auto alignment = gv.getAlign().valueOrOne().value();

    return DataNode::Create(
        ctx.module().ipgraph(),
        name,
        type,
        linkage,
        std::move(section),
        constant,
        alignment);
  };

  auto create_function_node = [](const ::llvm::Function & f, Context & ctx)
  {
    auto name = f.getName().str();
    auto type = ctx.GetTypeConverter().ConvertFunctionType(*f.getFunctionType());
    auto linkage = convert_linkage(f.getLinkage());
    auto callingConvention = convertCallingConventionToJlm(f.getCallingConv());
    auto attributes = convert_attributes(f.getAttributes().getFnAttrs(), ctx.GetTypeConverter());

    return FunctionNode::create(
        ctx.module().ipgraph(),
        name,
        type,
        linkage,
        callingConvention,
        attributes);
  };

  for (auto & gv : lm.globals())
  {
    auto node = create_data_node(gv, ctx);
    ctx.insert_value(&gv, ctx.module().create_global_value(node));
  }

  for (auto & f : lm.getFunctionList())
  {
    if (f.isIntrinsic() && isIntrinsicIgnored(f.getIntrinsicID()))
      continue;

    auto node = create_function_node(f, ctx);
    ctx.insert_value(&f, ctx.module().create_variable(node));
  }
}

static std::unique_ptr<DataNodeInit>
create_initialization(::llvm::GlobalVariable & gv, Context & ctx)
{
  if (!gv.hasInitializer())
    return nullptr;

  auto init = gv.getInitializer();
  auto tacs = ConvertConstant(init, ctx);
  if (tacs.empty())
    return std::make_unique<DataNodeInit>(ctx.lookup_value(init));

  return std::make_unique<DataNodeInit>(std::move(tacs));
}

static void
convert_global_value(::llvm::GlobalVariable & gv, Context & ctx)
{
  auto v = static_cast<const GlobalValue *>(ctx.lookup_value(&gv));

  ctx.set_node(v->node());
  v->node()->set_initialization(create_initialization(gv, ctx));
  ctx.set_node(nullptr);
}

static void
convert_globals(::llvm::Module & lm, Context & ctx)
{
  for (auto & gv : lm.globals())
    convert_global_value(gv, ctx);

  for (auto & f : lm.getFunctionList())
    convert_function(f, ctx);
}

std::unique_ptr<InterProceduralGraphModule>
ConvertLlvmModule(::llvm::Module & llvmModule)
{
  auto ipgModule = InterProceduralGraphModule::create(
      util::FilePath(llvmModule.getSourceFileName()),
      llvmModule.getTargetTriple(),
      llvmModule.getDataLayoutStr());

  Context ctx(*ipgModule);
  declare_globals(llvmModule, ctx);
  convert_globals(llvmModule, ctx);

  return ipgModule;
}

}
