/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/TypeConverter.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/control.hpp>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

namespace jlm::llvm
{

fpsize
TypeConverter::ExtractFloatingPointSize(const ::llvm::Type & type)
{
  JLM_ASSERT(type.isFloatingPointTy());

  switch (type.getTypeID())
  {
  case ::llvm::Type::HalfTyID:
    return fpsize::half;
  case ::llvm::Type::FloatTyID:
    return fpsize::flt;
  case ::llvm::Type::DoubleTyID:
    return fpsize::dbl;
  case ::llvm::Type::X86_FP80TyID:
    return fpsize::x86fp80;
  case ::llvm::Type::FP128TyID:
    return fpsize::fp128;
  default:
    JLM_UNREACHABLE("TypeConverter::ExtractFloatingPointSize: Unsupported floating point size.");
  }
}

::llvm::IntegerType *
TypeConverter::ConvertBitType(const rvsdg::BitType & bitType, ::llvm::LLVMContext & context)
{
  return ::llvm::Type::getIntNTy(context, bitType.nbits());
}

::llvm::FunctionType *
TypeConverter::ConvertFunctionType(
    const rvsdg::FunctionType & functionType,
    ::llvm::LLVMContext & context)
{
  bool isVariableArgument = false;
  std::vector<::llvm::Type *> argumentTypes;
  for (auto & argumentType : functionType.Arguments())
  {
    if (rvsdg::is<VariableArgumentType>(argumentType))
    {
      isVariableArgument = true;
      continue;
    }

    if (rvsdg::is<IOStateType>(argumentType))
      continue;
    if (rvsdg::is<MemoryStateType>(argumentType))
      continue;

    argumentTypes.push_back(ConvertJlmType(*argumentType, context));
  }

  // The return type can either be (ValueType, StateType, StateType, ...) if the function has
  // a return value, or (StateType, StateType, ...) if the function returns void.
  auto resultType = ::llvm::Type::getVoidTy(context);
  if (functionType.NumResults() > 0 && functionType.ResultType(0).Kind() == rvsdg::TypeKind::Value)
    resultType = ConvertJlmType(functionType.ResultType(0), context);

  return ::llvm::FunctionType::get(resultType, argumentTypes, isVariableArgument);
}

std::shared_ptr<const rvsdg::FunctionType>
TypeConverter::ConvertFunctionType(const ::llvm::FunctionType & functionType)
{
  // Arguments
  std::vector<std::shared_ptr<const rvsdg::Type>> argumentTypes;
  for (size_t n = 0; n < functionType.getNumParams(); n++)
    argumentTypes.push_back(ConvertLlvmType(*functionType.getParamType(n)));
  if (functionType.isVarArg())
    argumentTypes.push_back(VariableArgumentType::Create());
  argumentTypes.push_back(IOStateType::Create());
  argumentTypes.push_back(MemoryStateType::Create());

  // Results
  std::vector<std::shared_ptr<const rvsdg::Type>> resultTypes;
  if (functionType.getReturnType()->getTypeID() != ::llvm::Type::VoidTyID)
    resultTypes.push_back(ConvertLlvmType(*functionType.getReturnType()));
  resultTypes.push_back(IOStateType::Create());
  resultTypes.push_back(MemoryStateType::Create());

  return rvsdg::FunctionType::Create(std::move(argumentTypes), std::move(resultTypes));
}

::llvm::PointerType *
TypeConverter::ConvertPointerType(const PointerType &, ::llvm::LLVMContext & context)
{
  // FIXME: we default the address space to zero
  return ::llvm::PointerType::get(context, 0);
}

std::shared_ptr<const PointerType>
TypeConverter::ConvertPointerType(const ::llvm::PointerType & pointerType)
{
  JLM_ASSERT(pointerType.getAddressSpace() == 0);
  return PointerType::Create();
}

::llvm::ArrayType *
TypeConverter::ConvertArrayType(const ArrayType & type, ::llvm::LLVMContext & context)
{
  return ::llvm::ArrayType::get(ConvertJlmType(type.element_type(), context), type.nelements());
}

::llvm::Type *
TypeConverter::ConvertFloatingPointType(
    const FloatingPointType & type,
    ::llvm::LLVMContext & context)
{
  switch (type.size())
  {
  case fpsize::half:
    return ::llvm::Type::getHalfTy(context);
  case fpsize::flt:
    return ::llvm::Type::getFloatTy(context);
  case fpsize::dbl:
    return ::llvm::Type::getDoubleTy(context);
  case fpsize::fp128:
    return ::llvm::Type::getFP128Ty(context);
  case fpsize::x86fp80:
    return ::llvm::Type::getX86_FP80Ty(context);
  default:
    JLM_UNREACHABLE("TypeConverter::ConvertFloatingPointType: Unhandled floating point size.");
  }
}

::llvm::StructType *
TypeConverter::ConvertStructType(const StructType & type, ::llvm::LLVMContext & context)
{
  const auto sharedPtr = std::shared_ptr<const StructType>(std::shared_ptr<void>(), &type);

  if (StructTypeMap_.HasValue(sharedPtr))
  {
    const auto llvmStructType = StructTypeMap_.LookupValue(sharedPtr);
    JLM_ASSERT(&llvmStructType->getContext() == &context);
    return llvmStructType;
  }

  const auto llvmStructType = ::llvm::StructType::create(context);
  StructTypeMap_.Insert(llvmStructType, sharedPtr);

  std::vector<::llvm::Type *> elements;
  for (size_t n = 0; n < type.numElements(); n++)
    elements.push_back(ConvertJlmType(*type.getElementType(n), context));

  if (type.HasName())
    llvmStructType->setName(type.GetName());
  llvmStructType->setBody(elements, type.IsPacked());

  return llvmStructType;
}

::llvm::Type *
TypeConverter::ConvertJlmType(const rvsdg::Type & type, ::llvm::LLVMContext & context)
{
  if (const auto bitType = dynamic_cast<const rvsdg::BitType *>(&type))
  {
    return ConvertBitType(*bitType, context);
  }

  if (const auto functionType = dynamic_cast<const rvsdg::FunctionType *>(&type))
  {
    return ConvertFunctionType(*functionType, context);
  }

  if (const auto pointerType = dynamic_cast<const PointerType *>(&type))
  {
    return ConvertPointerType(*pointerType, context);
  }

  if (const auto arrayType = dynamic_cast<const ArrayType *>(&type))
  {
    return ConvertArrayType(*arrayType, context);
  }

  if (const auto controlType = dynamic_cast<const rvsdg::ControlType *>(&type))
  {
    return controlType->nalternatives() == 2 ? ::llvm::Type::getInt1Ty(context)
                                             : ::llvm::Type::getInt32Ty(context);
  }

  if (const auto floatingPointType = dynamic_cast<const FloatingPointType *>(&type))
  {
    return ConvertFloatingPointType(*floatingPointType, context);
  }

  if (const auto structType = dynamic_cast<const StructType *>(&type))
  {
    return ConvertStructType(*structType, context);
  }

  if (const auto fixedVectorType = dynamic_cast<const FixedVectorType *>(&type))
  {
    return ::llvm::VectorType::get(
        ConvertJlmType(fixedVectorType->type(), context),
        fixedVectorType->size(),
        false);
  }

  if (const auto scalableVectorType = dynamic_cast<const ScalableVectorType *>(&type))
  {
    return ::llvm::VectorType::get(
        ConvertJlmType(scalableVectorType->type(), context),
        scalableVectorType->size(),
        true);
  }

  JLM_UNREACHABLE("TypeConverter::ConvertJlmType: Unhandled jlm type.");
}

std::shared_ptr<const rvsdg::Type>
TypeConverter::ConvertLlvmType(::llvm::Type & type)
{
  switch (type.getTypeID())
  {
  case ::llvm::Type::IntegerTyID:
  {
    const auto integerType = ::llvm::cast<::llvm::IntegerType>(&type);
    return rvsdg::BitType::Create(integerType->getBitWidth());
  }
  case ::llvm::Type::PointerTyID:
    return ConvertPointerType(*::llvm::cast<::llvm::PointerType>(&type));
  case ::llvm::Type::FunctionTyID:
    return ConvertFunctionType(*::llvm::cast<::llvm::FunctionType>(&type));
  case ::llvm::Type::HalfTyID:
    return FloatingPointType::Create(fpsize::half);
  case ::llvm::Type::FloatTyID:
    return FloatingPointType::Create(fpsize::flt);
  case ::llvm::Type::DoubleTyID:
    return FloatingPointType::Create(fpsize::dbl);
  case ::llvm::Type::X86_FP80TyID:
    return FloatingPointType::Create(fpsize::x86fp80);
  case ::llvm::Type::FP128TyID:
    return FloatingPointType::Create(fpsize::fp128);
  case ::llvm::Type::StructTyID:
  {
    const auto structType = ::llvm::cast<::llvm::StructType>(&type);
    if (StructTypeMap_.HasKey(structType))
    {
      return StructTypeMap_.LookupKey(structType);
    }

    const auto isPacked = structType->isPacked();

    std::vector<std::shared_ptr<const rvsdg::Type>> elementTypes;
    for (const auto elementType : structType->elements())
    {
      elementTypes.push_back(ConvertLlvmType(*elementType));
    }

    auto jlmType =
        structType->hasName()
            ? StructType::Create(structType->getName().str(), isPacked, std::move(elementTypes))
            : StructType::Create(isPacked, std::move(elementTypes));

    StructTypeMap_.Insert(structType, jlmType);
    return jlmType;
  }
  case ::llvm::Type::ArrayTyID:
  {
    auto elementType = ConvertLlvmType(*type.getArrayElementType());
    return ArrayType::Create(std::move(elementType), type.getArrayNumElements());
  }
  case ::llvm::Type::FixedVectorTyID:
  {
    auto scalarType = ConvertLlvmType(*type.getScalarType());
    return FixedVectorType::Create(
        std::move(scalarType),
        ::llvm::cast<::llvm::FixedVectorType>(&type)->getNumElements());
  }
  case ::llvm::Type::ScalableVectorTyID:
  {
    auto scalarType = ConvertLlvmType(*type.getScalarType());
    return ScalableVectorType::Create(
        std::move(scalarType),
        ::llvm::cast<::llvm::ScalableVectorType>(&type)->getMinNumElements());
  }
  default:
    JLM_UNREACHABLE("TypeConverter::ConvertLlvmType: Unhandled llvm type.");
  }
}

}
