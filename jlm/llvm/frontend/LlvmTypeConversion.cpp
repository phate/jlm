/*
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/frontend/LlvmConversionContext.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

#include <map>

namespace jlm::llvm
{

fpsize
ExtractFloatingPointSize(const ::llvm::Type * type)
{
  JLM_ASSERT(type->isFloatingPointTy());

  static std::unordered_map<const ::llvm::Type::TypeID, llvm::fpsize> map(
      { { ::llvm::Type::HalfTyID, fpsize::half },
        { ::llvm::Type::FloatTyID, fpsize::flt },
        { ::llvm::Type::DoubleTyID, fpsize::dbl },
        { ::llvm::Type::X86_FP80TyID, fpsize::x86fp80 },
        { ::llvm::Type::FP128TyID, fpsize::fp128 } });

  auto i = map.find(type->getTypeID());
  JLM_ASSERT(i != map.end());
  return i->second;
}

static std::shared_ptr<const rvsdg::ValueType>
convert_integer_type(const ::llvm::Type * t, context &)
{
  JLM_ASSERT(t->getTypeID() == ::llvm::Type::IntegerTyID);
  auto * type = static_cast<const ::llvm::IntegerType *>(t);

  return rvsdg::bittype::Create(type->getBitWidth());
}

static std::shared_ptr<const rvsdg::ValueType>
convert_pointer_type(const ::llvm::Type * t, context &)
{
  JLM_ASSERT(t->getTypeID() == ::llvm::Type::PointerTyID);
  return PointerType::Create();
}

static std::shared_ptr<const rvsdg::ValueType>
convert_function_type(const ::llvm::Type * t, context & ctx)
{
  JLM_ASSERT(t->getTypeID() == ::llvm::Type::FunctionTyID);
  auto type = ::llvm::cast<const ::llvm::FunctionType>(t);

  /* arguments */
  std::vector<std::shared_ptr<const rvsdg::Type>> argumentTypes;
  for (size_t n = 0; n < type->getNumParams(); n++)
    argumentTypes.push_back(ConvertType(type->getParamType(n), ctx));
  if (type->isVarArg())
    argumentTypes.push_back(create_varargtype());
  argumentTypes.push_back(iostatetype::Create());
  argumentTypes.push_back(MemoryStateType::Create());

  /* results */
  std::vector<std::shared_ptr<const rvsdg::Type>> resultTypes;
  if (type->getReturnType()->getTypeID() != ::llvm::Type::VoidTyID)
    resultTypes.push_back(ConvertType(type->getReturnType(), ctx));
  resultTypes.push_back(iostatetype::Create());
  resultTypes.push_back(MemoryStateType::Create());

  return rvsdg::FunctionType::Create(std::move(argumentTypes), std::move(resultTypes));
}

static std::shared_ptr<const rvsdg::ValueType>
convert_fp_type(const ::llvm::Type * t, context &)
{
  static const std::unordered_map<::llvm::Type::TypeID, fpsize> map(
      { { ::llvm::Type::HalfTyID, fpsize::half },
        { ::llvm::Type::FloatTyID, fpsize::flt },
        { ::llvm::Type::DoubleTyID, fpsize::dbl },
        { ::llvm::Type::X86_FP80TyID, fpsize::x86fp80 },
        { ::llvm::Type::FP128TyID, fpsize::fp128 } });

  auto i = map.find(t->getTypeID());
  JLM_ASSERT(i != map.end());
  return fptype::Create(i->second);
}

static std::shared_ptr<const rvsdg::ValueType>
convert_struct_type(const ::llvm::Type * t, context & ctx)
{
  JLM_ASSERT(t->isStructTy());
  auto type = static_cast<const ::llvm::StructType *>(t);

  auto isPacked = type->isPacked();
  auto & declaration = *ctx.lookup_declaration(type);

  return type->hasName() ? StructType::Create(type->getName().str(), isPacked, declaration)
                         : StructType::Create(isPacked, declaration);
}

static std::shared_ptr<const rvsdg::ValueType>
convert_array_type(const ::llvm::Type * t, context & ctx)
{
  JLM_ASSERT(t->isArrayTy());
  auto etype = ConvertType(t->getArrayElementType(), ctx);
  return arraytype::Create(std::move(etype), t->getArrayNumElements());
}

static std::shared_ptr<const rvsdg::ValueType>
convert_fixed_vector_type(const ::llvm::Type * t, context & ctx)
{
  JLM_ASSERT(t->getTypeID() == ::llvm::Type::FixedVectorTyID);
  auto type = ConvertType(t->getScalarType(), ctx);
  return fixedvectortype::Create(
      std::move(type),
      ::llvm::cast<::llvm::FixedVectorType>(t)->getNumElements());
}

static std::shared_ptr<const rvsdg::ValueType>
convert_scalable_vector_type(const ::llvm::Type * t, context & ctx)
{
  JLM_ASSERT(t->getTypeID() == ::llvm::Type::ScalableVectorTyID);
  auto type = ConvertType(t->getScalarType(), ctx);
  return scalablevectortype::Create(
      std::move(type),
      ::llvm::cast<::llvm::ScalableVectorType>(t)->getMinNumElements());
}

std::shared_ptr<const rvsdg::ValueType>
ConvertType(const ::llvm::Type * t, context & ctx)
{
  static std::unordered_map<
      ::llvm::Type::TypeID,
      std::function<std::shared_ptr<const rvsdg::ValueType>(const ::llvm::Type *, context &)>>
      map({ { ::llvm::Type::IntegerTyID, convert_integer_type },
            { ::llvm::Type::PointerTyID, convert_pointer_type },
            { ::llvm::Type::FunctionTyID, convert_function_type },
            { ::llvm::Type::HalfTyID, convert_fp_type },
            { ::llvm::Type::FloatTyID, convert_fp_type },
            { ::llvm::Type::DoubleTyID, convert_fp_type },
            { ::llvm::Type::X86_FP80TyID, convert_fp_type },
            { ::llvm::Type::FP128TyID, convert_fp_type },
            { ::llvm::Type::StructTyID, convert_struct_type },
            { ::llvm::Type::ArrayTyID, convert_array_type },
            { ::llvm::Type::FixedVectorTyID, convert_fixed_vector_type },
            { ::llvm::Type::ScalableVectorTyID, convert_scalable_vector_type } });

  JLM_ASSERT(map.find(t->getTypeID()) != map.end());
  return map[t->getTypeID()](t, ctx);
}

}
