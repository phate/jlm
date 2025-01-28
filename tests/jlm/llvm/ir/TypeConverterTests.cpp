/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>

#include <jlm/llvm/ir/TypeConverter.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/FunctionType.hpp>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>

static int
LlvmIntegerTypeConversion()
{
  using namespace jlm::llvm;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto i1 = ::llvm::IntegerType::get(context, 1);
  const auto i2 = ::llvm::IntegerType::get(context, 2);
  const auto i4 = ::llvm::IntegerType::get(context, 4);
  const auto i8 = ::llvm::IntegerType::get(context, 8);
  const auto i16 = ::llvm::IntegerType::get(context, 16);
  const auto i32 = ::llvm::IntegerType::get(context, 32);
  const auto i64 = ::llvm::IntegerType::get(context, 64);

  // Act
  const auto i1BitType =
      std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(typeConverter.ConvertLlvmType(*i1));
  const auto i2BitType =
      std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(typeConverter.ConvertLlvmType(*i2));
  const auto i4BitType =
      std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(typeConverter.ConvertLlvmType(*i4));
  const auto i8BitType =
      std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(typeConverter.ConvertLlvmType(*i8));
  const auto i16BitType =
      std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(typeConverter.ConvertLlvmType(*i16));
  const auto i32BitType =
      std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(typeConverter.ConvertLlvmType(*i32));
  const auto i64BitType =
      std::dynamic_pointer_cast<const jlm::rvsdg::bittype>(typeConverter.ConvertLlvmType(*i64));

  // Assert
  assert(i1BitType && i1BitType->nbits() == 1);
  assert(i2BitType && i2BitType->nbits() == 2);
  assert(i4BitType && i4BitType->nbits() == 4);
  assert(i8BitType && i8BitType->nbits() == 8);
  assert(i16BitType && i16BitType->nbits() == 16);
  assert(i32BitType && i32BitType->nbits() == 32);
  assert(i64BitType && i64BitType->nbits() == 64);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-LlvmIntegerTypeConversion",
    LlvmIntegerTypeConversion);

static int
LlvmPointerTypeConversion()
{
  using namespace jlm::llvm;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto pointerTypeLlvm = ::llvm::PointerType::get(context, 0);

  // Act
  const auto pointerTypeJlm =
      std::dynamic_pointer_cast<const PointerType>(typeConverter.ConvertLlvmType(*pointerTypeLlvm));

  // Assert
  assert(pointerTypeJlm);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-LlvmPointerTypeConversion",
    LlvmPointerTypeConversion);

static int
LlvmFunctionTypeConversion()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto voidType = ::llvm::Type::getVoidTy(context);
  auto i32Type = ::llvm::Type::getInt32Ty(context);
  const auto functionType1Llvm = ::llvm::FunctionType::get(voidType, { i32Type, i32Type }, false);
  const auto functionType2Llvm = ::llvm::FunctionType::get(i32Type, {}, false);
  const auto functionType3Llvm = ::llvm::FunctionType::get(i32Type, { i32Type, i32Type }, true);

  // Act
  const auto functionType1Jlm = std::dynamic_pointer_cast<const FunctionType>(
      typeConverter.ConvertLlvmType(*functionType1Llvm));
  const auto functionType2Jlm = std::dynamic_pointer_cast<const FunctionType>(
      typeConverter.ConvertLlvmType(*functionType2Llvm));
  const auto functionType3Jlm = std::dynamic_pointer_cast<const FunctionType>(
      typeConverter.ConvertLlvmType(*functionType3Llvm));

  // Assert
  assert(functionType1Jlm != nullptr);
  assert(functionType1Jlm->NumArguments() == 4);
  assert(functionType1Jlm->NumResults() == 2);
  auto arguments = functionType1Jlm->Arguments();
  assert(is<bittype>(arguments[0]));
  assert(is<bittype>(arguments[1]));
  assert(is<IOStateType>(arguments[2]));
  assert(is<MemoryStateType>(arguments[3]));
  auto results = functionType1Jlm->Results();
  assert(is<IOStateType>(results[0]));
  assert(is<MemoryStateType>(results[1]));

  assert(functionType2Jlm != nullptr);
  assert(functionType2Jlm->NumArguments() == 2);
  assert(functionType2Jlm->NumResults() == 3);
  arguments = functionType2Jlm->Arguments();
  assert(is<IOStateType>(arguments[0]));
  assert(is<MemoryStateType>(arguments[1]));
  results = functionType2Jlm->Results();
  assert(is<bittype>(results[0]));
  assert(is<IOStateType>(results[1]));
  assert(is<MemoryStateType>(results[2]));

  assert(functionType3Jlm != nullptr);
  assert(functionType3Jlm->NumArguments() == 5);
  assert(functionType3Jlm->NumResults() == 3);
  arguments = functionType3Jlm->Arguments();
  assert(is<bittype>(arguments[0]));
  assert(is<bittype>(arguments[1]));
  assert(is<VariableArgumentType>(arguments[2]));
  assert(is<IOStateType>(arguments[3]));
  assert(is<MemoryStateType>(arguments[4]));
  results = functionType3Jlm->Results();
  assert(is<bittype>(results[0]));
  assert(is<IOStateType>(results[1]));
  assert(is<MemoryStateType>(results[2]));

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-LlvmFunctionTypeConversion",
    LlvmFunctionTypeConversion);

static int
LlvmFloatingPointTypeConversion()
{
  using namespace jlm::llvm;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto halfTypeLlvm = ::llvm::Type::getHalfTy(context);
  const auto floatTypeLlvm = ::llvm::Type::getFloatTy(context);
  const auto doubleTypeLlvm = ::llvm::Type::getDoubleTy(context);
  const auto x86fp80TypeLlvm = ::llvm::Type::getX86_FP80Ty(context);
  const auto fp128TypeLlvm = ::llvm::Type::getFP128Ty(context);

  // Act
  const auto halfTypeJlm =
      std::dynamic_pointer_cast<const fptype>(typeConverter.ConvertLlvmType(*halfTypeLlvm));
  const auto floatTypeJlm =
      std::dynamic_pointer_cast<const fptype>(typeConverter.ConvertLlvmType(*floatTypeLlvm));
  const auto doubleTypeJlm =
      std::dynamic_pointer_cast<const fptype>(typeConverter.ConvertLlvmType(*doubleTypeLlvm));
  const auto x86fp80TypeJlm =
      std::dynamic_pointer_cast<const fptype>(typeConverter.ConvertLlvmType(*x86fp80TypeLlvm));
  const auto fp128TypeJlm =
      std::dynamic_pointer_cast<const fptype>(typeConverter.ConvertLlvmType(*fp128TypeLlvm));

  // Assert
  assert(halfTypeJlm && halfTypeJlm->size() == fpsize::half);
  assert(floatTypeJlm && floatTypeJlm->size() == fpsize::flt);
  assert(doubleTypeJlm && doubleTypeJlm->size() == fpsize::dbl);
  assert(x86fp80TypeJlm && x86fp80TypeJlm->size() == fpsize::x86fp80);
  assert(fp128TypeJlm->size() == fpsize::fp128);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-LlvmFloatingPointTypeConversion",
    LlvmFloatingPointTypeConversion);

static int
LlvmStructTypeConversion()
{
  using namespace jlm::llvm;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  auto i32Type = ::llvm::Type::getInt32Ty(context);
  const auto halfType = ::llvm::Type::getHalfTy(context);
  const auto structType1Llvm = ::llvm::StructType::get(context, { i32Type, halfType }, false);
  const auto structType2Llvm =
      ::llvm::StructType::get(context, { i32Type, i32Type, i32Type }, true);
  const auto structType3Llvm = ::llvm::StructType::get(context, { i32Type }, true);
  structType3Llvm->setName("myStruct");

  // Act
  const auto structType1Jlm =
      std::dynamic_pointer_cast<const StructType>(typeConverter.ConvertLlvmType(*structType1Llvm));
  const auto structType2Jlm =
      std::dynamic_pointer_cast<const StructType>(typeConverter.ConvertLlvmType(*structType2Llvm));
  const auto structType3Jlm =
      std::dynamic_pointer_cast<const StructType>(typeConverter.ConvertLlvmType(*structType3Llvm));

  const auto structType4Jlm =
      std::dynamic_pointer_cast<const StructType>(typeConverter.ConvertLlvmType(*structType1Llvm));

  // Assert
  assert(structType1Jlm);
  assert(structType1Jlm->GetDeclaration().NumElements() == 2);
  assert(!structType1Jlm->IsPacked());
  assert(!structType1Jlm->HasName());

  assert(structType2Jlm);
  assert(structType2Jlm->GetDeclaration().NumElements() == 3);
  assert(structType2Jlm->IsPacked());
  assert(!structType2Jlm->HasName());

  assert(structType3Jlm);
  assert(structType3Jlm->GetDeclaration().NumElements() == 1);
  assert(structType3Jlm->IsPacked());
  assert(structType3Jlm->HasName() && structType3Jlm->GetName() == "myStruct");

  assert(&structType1Jlm->GetDeclaration() != &structType2Jlm->GetDeclaration());
  assert(&structType1Jlm->GetDeclaration() != &structType3Jlm->GetDeclaration());
  assert(&structType1Jlm->GetDeclaration() == &structType4Jlm->GetDeclaration());
  assert(&structType2Jlm->GetDeclaration() != &structType3Jlm->GetDeclaration());

  const auto declarations = typeConverter.ReleaseStructTypeDeclarations();
  assert(declarations.size() == 3);

  // We released all struct declarations. After that, translating the same type again should get
  // us a new declarations.
  const auto structType5Jlm =
      std::dynamic_pointer_cast<const StructType>(typeConverter.ConvertLlvmType(*structType1Llvm));

  assert(&structType5Jlm->GetDeclaration() != &structType1Jlm->GetDeclaration());

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-LlvmStructTypeConversion",
    LlvmStructTypeConversion);

static int
LlvmArrayTypeConversion()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto i32Type = ::llvm::Type::getInt32Ty(context);
  const auto halfType = ::llvm::Type::getHalfTy(context);
  const auto arrayType1Llvm = ::llvm::ArrayType::get(i32Type, 4);
  const auto arrayType2Llvm = ::llvm::ArrayType::get(halfType, 9);

  // Act
  const auto arrayType1Jlm =
      std::dynamic_pointer_cast<const ArrayType>(typeConverter.ConvertLlvmType(*arrayType1Llvm));
  const auto arrayType2Jlm =
      std::dynamic_pointer_cast<const ArrayType>(typeConverter.ConvertLlvmType(*arrayType2Llvm));

  // Assert
  assert(arrayType1Jlm);
  assert(is<bittype>(arrayType1Jlm->element_type()));
  assert(arrayType1Jlm->nelements() == 4);

  assert(arrayType2Jlm);
  assert(is<fptype>(arrayType2Jlm->element_type()));
  assert(arrayType2Jlm->nelements() == 9);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-LlvmArrayTypeConversion",
    LlvmArrayTypeConversion);

static int
LlvmVectorTypeConversion()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto i32Type = ::llvm::Type::getInt32Ty(context);
  const auto halfType = ::llvm::Type::getHalfTy(context);
  const auto vectorType1Llvm = ::llvm::VectorType::get(i32Type, 4, false);
  const auto vectorType2Llvm = ::llvm::VectorType::get(halfType, 9, true);

  // Act
  const auto vectorType1Jlm = std::dynamic_pointer_cast<const fixedvectortype>(
      typeConverter.ConvertLlvmType(*vectorType1Llvm));
  const auto vectorType2Jlm = std::dynamic_pointer_cast<const scalablevectortype>(
      typeConverter.ConvertLlvmType(*vectorType2Llvm));

  // Assert
  assert(vectorType1Jlm);
  assert(is<bittype>(vectorType1Jlm->type()));
  assert(vectorType1Jlm->size() == 4);

  assert(vectorType2Jlm);
  assert(is<fptype>(vectorType2Jlm->type()));
  assert(vectorType2Jlm->size() == 9);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-LlvmVectorTypeConversion",
    LlvmVectorTypeConversion);

static int
JLmBitTypeConversion()
{
  using namespace jlm::llvm;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto i1 = jlm::rvsdg::bittype::Create(1);
  const auto i2 = jlm::rvsdg::bittype::Create(2);
  const auto i4 = jlm::rvsdg::bittype::Create(4);
  const auto i8 = jlm::rvsdg::bittype::Create(8);
  const auto i16 = jlm::rvsdg::bittype::Create(16);
  const auto i32 = jlm::rvsdg::bittype::Create(32);
  const auto i64 = jlm::rvsdg::bittype::Create(64);

  // Act
  const auto i1Type = typeConverter.ConvertJlmType(*i1, context);
  const auto i2Type = typeConverter.ConvertJlmType(*i2, context);
  const auto i4Type = typeConverter.ConvertJlmType(*i4, context);
  const auto i8Type = typeConverter.ConvertJlmType(*i8, context);
  const auto i16Type = typeConverter.ConvertJlmType(*i16, context);
  const auto i32Type = typeConverter.ConvertJlmType(*i32, context);
  const auto i64Type = typeConverter.ConvertJlmType(*i64, context);

  // Assert
  assert(i1Type->getTypeID() == llvm::Type::IntegerTyID);
  assert(i1Type->getIntegerBitWidth() == 1);

  assert(i2Type->getTypeID() == llvm::Type::IntegerTyID);
  assert(i2Type->getIntegerBitWidth() == 2);

  assert(i4Type->getTypeID() == llvm::Type::IntegerTyID);
  assert(i4Type->getIntegerBitWidth() == 4);

  assert(i8Type->getTypeID() == llvm::Type::IntegerTyID);
  assert(i8Type->getIntegerBitWidth() == 8);

  assert(i16Type->getTypeID() == llvm::Type::IntegerTyID);
  assert(i16Type->getIntegerBitWidth() == 16);

  assert(i32Type->getTypeID() == llvm::Type::IntegerTyID);
  assert(i32Type->getIntegerBitWidth() == 32);

  assert(i64Type->getTypeID() == llvm::Type::IntegerTyID);
  assert(i64Type->getIntegerBitWidth() == 64);

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/TypeConverterTests-JLmBitTypeConversion", JLmBitTypeConversion);

static int
JlmFunctionTypeConversion()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  auto bit32Type = bittype::Create(32);
  auto ioStateType = IOStateType::Create();
  auto memoryStateType = MemoryStateType::Create();
  auto varArgType = VariableArgumentType::Create();
  const auto functionType1Jlm = FunctionType::Create(
      { bit32Type, bit32Type, ioStateType, memoryStateType },
      { memoryStateType, ioStateType });
  const auto functionType2Jlm = FunctionType::Create(
      { ioStateType, memoryStateType },
      { bit32Type, memoryStateType, ioStateType });
  const auto functionType3Jlm = FunctionType::Create(
      { bit32Type, bit32Type, varArgType, ioStateType, memoryStateType },
      { bit32Type, memoryStateType, ioStateType });

  // Act
  const auto functionType1Llvm =
      llvm::dyn_cast<llvm::FunctionType>(typeConverter.ConvertJlmType(*functionType1Jlm, context));
  const auto functionType2Llvm =
      llvm::dyn_cast<llvm::FunctionType>(typeConverter.ConvertJlmType(*functionType2Jlm, context));
  const auto functionType3Llvm =
      llvm::dyn_cast<llvm::FunctionType>(typeConverter.ConvertJlmType(*functionType3Jlm, context));

  // Assert
  assert(functionType1Llvm != nullptr);
  assert(functionType1Llvm->getNumParams() == 2);
  assert(functionType1Llvm->getParamType(0)->getTypeID() == llvm::Type::IntegerTyID);
  assert(functionType1Llvm->getParamType(1)->getTypeID() == llvm::Type::IntegerTyID);
  assert(functionType1Llvm->getReturnType()->getTypeID() == llvm::Type::VoidTyID);
  assert(!functionType1Llvm->isVarArg());

  assert(functionType2Llvm != nullptr);
  assert(functionType2Llvm->getNumParams() == 0);
  assert(functionType2Llvm->getReturnType()->getTypeID() == llvm::Type::IntegerTyID);
  assert(!functionType2Llvm->isVarArg());

  assert(functionType3Llvm != nullptr);
  assert(functionType3Llvm->getNumParams() == 2);
  assert(functionType3Llvm->getParamType(0)->getTypeID() == llvm::Type::IntegerTyID);
  assert(functionType3Llvm->getParamType(1)->getTypeID() == llvm::Type::IntegerTyID);
  assert(functionType3Llvm->getReturnType()->getTypeID() == llvm::Type::IntegerTyID);
  assert(functionType3Llvm->isVarArg());

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-JlmFunctionTypeConversion",
    JlmFunctionTypeConversion);

static int
JlmPointerTypeConversion()
{
  using namespace jlm::llvm;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto pointerTypeJlm = PointerType::Create();

  // Act
  const auto pointerTypeLlvm =
      llvm::dyn_cast<llvm::PointerType>(typeConverter.ConvertJlmType(*pointerTypeJlm, context));

  // Assert
  assert(pointerTypeLlvm);
  assert(pointerTypeLlvm->getAddressSpace() == 0);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-JlmPointerTypeConversion",
    JlmPointerTypeConversion);

static int
JlmArrayTypeConversion()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto bit32Type = bittype::Create(32);
  const auto halfType = fptype::Create(fpsize::half);
  const auto arrayType1Jlm = ArrayType::Create(bit32Type, 4);
  const auto arrayType2Jlm = ArrayType::Create(halfType, 9);

  // Act
  const auto arrayType1Llvm = typeConverter.ConvertJlmType(*arrayType1Jlm, context);
  const auto arrayType2Llvm = typeConverter.ConvertJlmType(*arrayType2Jlm, context);

  // Assert
  assert(arrayType1Llvm->isArrayTy());
  assert(arrayType1Llvm->getArrayNumElements() == 4);
  assert(arrayType1Llvm->getArrayElementType()->getTypeID() == llvm::Type::IntegerTyID);

  assert(arrayType2Llvm->isArrayTy());
  assert(arrayType2Llvm->getArrayNumElements() == 9);
  assert(arrayType2Llvm->getArrayElementType()->getTypeID() == llvm::Type::HalfTyID);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-JlmArrayTypeConversion",
    JlmArrayTypeConversion);

static int
JlmControlTypeConversion()
{
  using namespace jlm::llvm;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto controlType1 = jlm::rvsdg::ControlType::Create(2);
  const auto controlType10 = jlm::rvsdg::ControlType::Create(10);

  // Act
  const auto integerType1Llvm = typeConverter.ConvertJlmType(*controlType1, context);
  const auto integerType2Llvm = typeConverter.ConvertJlmType(*controlType10, context);

  // Assert
  assert(integerType1Llvm->getTypeID() == llvm::Type::IntegerTyID);
  assert(integerType1Llvm->getIntegerBitWidth() == 1);

  assert(integerType2Llvm->getTypeID() == llvm::Type::IntegerTyID);
  assert(integerType2Llvm->getIntegerBitWidth() == 32);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-JlmControlTypeConversion",
    JlmControlTypeConversion);

static int
JlmFloatingPointTypeConversion()
{
  using namespace jlm::llvm;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto halfTypeJlm = fptype::Create(fpsize::half);
  const auto floatTypeJlm = fptype::Create(fpsize::flt);
  const auto doubleTypeJlm = fptype::Create(fpsize::dbl);
  const auto x86fp80TypeJlm = fptype::Create(fpsize::x86fp80);
  const auto fp128TypeJlm = fptype::Create(fpsize::fp128);

  // Act
  const auto halfTypeLlvm = typeConverter.ConvertJlmType(*halfTypeJlm, context);
  const auto floatTypeLlvm = typeConverter.ConvertJlmType(*floatTypeJlm, context);
  const auto doubleTypeLlvm = typeConverter.ConvertJlmType(*doubleTypeJlm, context);
  const auto x86fp80TypeLlvm = typeConverter.ConvertJlmType(*x86fp80TypeJlm, context);
  const auto fp128TypeLlvm = typeConverter.ConvertJlmType(*fp128TypeJlm, context);

  // Assert
  assert(halfTypeLlvm->getTypeID() == llvm::Type::HalfTyID);
  assert(floatTypeLlvm->getTypeID() == llvm::Type::FloatTyID);
  assert(doubleTypeLlvm->getTypeID() == llvm::Type::DoubleTyID);
  assert(x86fp80TypeLlvm->getTypeID() == llvm::Type::X86_FP80TyID);
  assert(fp128TypeLlvm->getTypeID() == llvm::Type::FP128TyID);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-JlmFloatingPointTypeConversion",
    JlmFloatingPointTypeConversion);

static int
JlmStructTypeConversion()
{
  using namespace jlm::llvm;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto bit32Type = jlm::rvsdg::bittype::Create(32);
  const auto halfType = fptype::Create(fpsize::half);

  const auto declaration1 = StructType::Declaration::Create({ bit32Type, halfType });
  const auto declaration2 = StructType::Declaration::Create({ bit32Type, bit32Type, bit32Type });
  const auto declaration3 = StructType::Declaration::Create({ bit32Type });

  const auto structType1Jlm = StructType::Create(false, *declaration1);
  const auto structType2Jlm = StructType::Create(true, *declaration2);
  const auto structType3Jlm = StructType::Create("myStruct", true, *declaration3);

  // Act
  const auto structType1Llvm = typeConverter.ConvertJlmType(*structType1Jlm, context);
  const auto structType2Llvm = typeConverter.ConvertJlmType(*structType2Jlm, context);
  const auto structType3Llvm = typeConverter.ConvertJlmType(*structType3Jlm, context);

  const auto structType4Llvm = typeConverter.ConvertJlmType(*structType1Jlm, context);

  // Assert
  assert(structType1Llvm->getTypeID() == llvm::Type::StructTyID);
  assert(structType1Llvm->getStructNumElements() == 2);
  assert(structType1Llvm->getStructElementType(0)->getTypeID() == llvm::Type::IntegerTyID);
  assert(structType1Llvm->getStructElementType(1)->getTypeID() == llvm::Type::HalfTyID);
  assert(!llvm::dyn_cast<llvm::StructType>(structType1Llvm)->isPacked());

  assert(structType2Llvm->getTypeID() == llvm::Type::StructTyID);
  assert(structType2Llvm->getStructNumElements() == 3);
  assert(structType2Llvm->getStructElementType(0)->getTypeID() == llvm::Type::IntegerTyID);
  assert(structType2Llvm->getStructElementType(1)->getTypeID() == llvm::Type::IntegerTyID);
  assert(structType2Llvm->getStructElementType(2)->getTypeID() == llvm::Type::IntegerTyID);
  assert(llvm::dyn_cast<llvm::StructType>(structType2Llvm)->isPacked());

  assert(structType3Llvm->getTypeID() == llvm::Type::StructTyID);
  assert(structType3Llvm->getStructNumElements() == 1);
  assert(structType3Llvm->getStructElementType(0)->getTypeID() == llvm::Type::IntegerTyID);
  assert(structType3Llvm->getStructName() == "myStruct");
  assert(llvm::dyn_cast<llvm::StructType>(structType3Llvm)->isPacked());

  assert(structType4Llvm == structType1Llvm);

  // The type converter created no jlm struct types. It is therefore not the owner of any
  // declarations.
  const auto declarations = typeConverter.ReleaseStructTypeDeclarations();
  assert(declarations.size() == 0);

  // Converting the same type again after the declaration release should give us a new Llvm type
  const auto structType5Llvm = typeConverter.ConvertJlmType(*structType1Jlm, context);
  assert(structType5Llvm != structType1Llvm);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-JlmStructTypeConversion",
    JlmStructTypeConversion);

static int
JlmFixedVectorTypeConversion()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto bit32Type = bittype::Create(32);
  const auto fixedVectorType1 = fixedvectortype::Create(bit32Type, 2);
  const auto fixedVectorType2 = fixedvectortype::Create(bit32Type, 4);

  // Act
  const auto vectorType1 =
      llvm::dyn_cast<llvm::VectorType>(typeConverter.ConvertJlmType(*fixedVectorType1, context));
  const auto vectorType2 =
      llvm::dyn_cast<llvm::VectorType>(typeConverter.ConvertJlmType(*fixedVectorType2, context));

  // Assert
  assert(vectorType1->getTypeID() == llvm::Type::FixedVectorTyID);
  assert(vectorType1->getElementType()->getTypeID() == llvm::Type::IntegerTyID);
  assert(vectorType1->getElementCount().getFixedValue() == 2);

  assert(vectorType2->getTypeID() == llvm::Type::FixedVectorTyID);
  assert(vectorType2->getElementType()->getTypeID() == llvm::Type::IntegerTyID);
  assert(vectorType2->getElementCount().getFixedValue() == 4);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-JlmFixedVectorTypeConversion",
    JlmFixedVectorTypeConversion);

static int
JlmScalableVectorTypeConversion()
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto bit32Type = bittype::Create(32);
  const auto scalableVectorType1 = scalablevectortype::Create(bit32Type, 2);
  const auto scalableVectorType2 = scalablevectortype::Create(bit32Type, 4);

  // Act
  const auto vectorType1 =
      llvm::dyn_cast<llvm::VectorType>(typeConverter.ConvertJlmType(*scalableVectorType1, context));
  const auto vectorType2 =
      llvm::dyn_cast<llvm::VectorType>(typeConverter.ConvertJlmType(*scalableVectorType2, context));

  // Assert
  assert(vectorType1->getTypeID() == llvm::Type::ScalableVectorTyID);
  assert(vectorType1->getElementType()->getTypeID() == llvm::Type::IntegerTyID);
  assert(vectorType1->getElementCount().getKnownMinValue() == 2);

  assert(vectorType2->getTypeID() == llvm::Type::ScalableVectorTyID);
  assert(vectorType2->getElementType()->getTypeID() == llvm::Type::IntegerTyID);
  assert(vectorType2->getElementCount().getKnownMinValue() == 4);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/ir/TypeConverterTests-JlmScalableVectorTypeConversion",
    JlmScalableVectorTypeConversion);
