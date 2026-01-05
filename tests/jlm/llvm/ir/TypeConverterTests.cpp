/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <gtest/gtest.h>

#include <jlm/llvm/ir/TypeConverter.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/control.hpp>
#include <jlm/rvsdg/FunctionType.hpp>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>

TEST(TypeConverterTests, LlvmIntegerTypeConversion)
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
      std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(typeConverter.ConvertLlvmType(*i1));
  const auto i2BitType =
      std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(typeConverter.ConvertLlvmType(*i2));
  const auto i4BitType =
      std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(typeConverter.ConvertLlvmType(*i4));
  const auto i8BitType =
      std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(typeConverter.ConvertLlvmType(*i8));
  const auto i16BitType =
      std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(typeConverter.ConvertLlvmType(*i16));
  const auto i32BitType =
      std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(typeConverter.ConvertLlvmType(*i32));
  const auto i64BitType =
      std::dynamic_pointer_cast<const jlm::rvsdg::BitType>(typeConverter.ConvertLlvmType(*i64));

  // Assert
  EXPECT_TRUE(i1BitType && i1BitType->nbits() == 1);
  EXPECT_TRUE(i2BitType && i2BitType->nbits() == 2);
  EXPECT_TRUE(i4BitType && i4BitType->nbits() == 4);
  EXPECT_TRUE(i8BitType && i8BitType->nbits() == 8);
  EXPECT_TRUE(i16BitType && i16BitType->nbits() == 16);
  EXPECT_TRUE(i32BitType && i32BitType->nbits() == 32);
  EXPECT_TRUE(i64BitType && i64BitType->nbits() == 64);
}

TEST(TypeConverterTests, LlvmPointerTypeConversion)
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
  EXPECT_NE(pointerTypeJlm, nullptr);
}

TEST(TypeConverterTests, LlvmFunctionTypeConversion)
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
  EXPECT_NE(functionType1Jlm, nullptr);
  EXPECT_EQ(functionType1Jlm->NumArguments(), 4);
  EXPECT_EQ(functionType1Jlm->NumResults(), 2);
  auto arguments = functionType1Jlm->Arguments();
  EXPECT_TRUE(is<BitType>(arguments[0]));
  EXPECT_TRUE(is<BitType>(arguments[1]));
  EXPECT_TRUE(is<IOStateType>(arguments[2]));
  EXPECT_TRUE(is<MemoryStateType>(arguments[3]));
  auto results = functionType1Jlm->Results();
  EXPECT_TRUE(is<IOStateType>(results[0]));
  EXPECT_TRUE(is<MemoryStateType>(results[1]));

  EXPECT_NE(functionType2Jlm, nullptr);
  EXPECT_EQ(functionType2Jlm->NumArguments(), 2);
  EXPECT_EQ(functionType2Jlm->NumResults(), 3);
  arguments = functionType2Jlm->Arguments();
  EXPECT_TRUE(is<IOStateType>(arguments[0]));
  EXPECT_TRUE(is<MemoryStateType>(arguments[1]));
  results = functionType2Jlm->Results();
  EXPECT_TRUE(is<BitType>(results[0]));
  EXPECT_TRUE(is<IOStateType>(results[1]));
  EXPECT_TRUE(is<MemoryStateType>(results[2]));

  EXPECT_NE(functionType3Jlm, nullptr);
  EXPECT_EQ(functionType3Jlm->NumArguments(), 5);
  EXPECT_EQ(functionType3Jlm->NumResults(), 3);
  arguments = functionType3Jlm->Arguments();
  EXPECT_TRUE(is<BitType>(arguments[0]));
  EXPECT_TRUE(is<BitType>(arguments[1]));
  EXPECT_TRUE(is<VariableArgumentType>(arguments[2]));
  EXPECT_TRUE(is<IOStateType>(arguments[3]));
  EXPECT_TRUE(is<MemoryStateType>(arguments[4]));
  results = functionType3Jlm->Results();
  EXPECT_TRUE(is<BitType>(results[0]));
  EXPECT_TRUE(is<IOStateType>(results[1]));
  EXPECT_TRUE(is<MemoryStateType>(results[2]));
}

TEST(TypeConverterTests, LlvmFloatingPointTypeConversion)
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
  const auto halfTypeJlm = std::dynamic_pointer_cast<const FloatingPointType>(
      typeConverter.ConvertLlvmType(*halfTypeLlvm));
  const auto floatTypeJlm = std::dynamic_pointer_cast<const FloatingPointType>(
      typeConverter.ConvertLlvmType(*floatTypeLlvm));
  const auto doubleTypeJlm = std::dynamic_pointer_cast<const FloatingPointType>(
      typeConverter.ConvertLlvmType(*doubleTypeLlvm));
  const auto x86fp80TypeJlm = std::dynamic_pointer_cast<const FloatingPointType>(
      typeConverter.ConvertLlvmType(*x86fp80TypeLlvm));
  const auto fp128TypeJlm = std::dynamic_pointer_cast<const FloatingPointType>(
      typeConverter.ConvertLlvmType(*fp128TypeLlvm));

  // Assert
  EXPECT_TRUE(halfTypeJlm && halfTypeJlm->size() == fpsize::half);
  EXPECT_TRUE(floatTypeJlm && floatTypeJlm->size() == fpsize::flt);
  EXPECT_TRUE(doubleTypeJlm && doubleTypeJlm->size() == fpsize::dbl);
  EXPECT_TRUE(x86fp80TypeJlm && x86fp80TypeJlm->size() == fpsize::x86fp80);
  EXPECT_TRUE(fp128TypeJlm->size() == fpsize::fp128);
}

TEST(TypeConverterTests, LlvmStructTypeConversion)
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
  const auto structType3Llvm = ::llvm::StructType::create(context, { i32Type }, "myStruct", true);

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
  EXPECT_NE(structType1Jlm, nullptr);
  EXPECT_EQ(structType1Jlm->GetDeclaration().NumElements(), 2);
  EXPECT_FALSE(structType1Jlm->IsPacked());
  EXPECT_FALSE(structType1Jlm->HasName());

  EXPECT_NE(structType2Jlm, nullptr);
  EXPECT_EQ(structType2Jlm->GetDeclaration().NumElements(), 3);
  EXPECT_TRUE(structType2Jlm->IsPacked());
  EXPECT_FALSE(structType2Jlm->HasName());

  EXPECT_NE(structType3Jlm, nullptr);
  EXPECT_EQ(structType3Jlm->GetDeclaration().NumElements(), 1);
  EXPECT_TRUE(structType3Jlm->IsPacked());
  EXPECT_TRUE(structType3Jlm->HasName() && structType3Jlm->GetName() == "myStruct");

  EXPECT_NE(&structType1Jlm->GetDeclaration(), &structType2Jlm->GetDeclaration());
  EXPECT_NE(&structType1Jlm->GetDeclaration(), &structType3Jlm->GetDeclaration());
  EXPECT_EQ(&structType1Jlm->GetDeclaration(), &structType4Jlm->GetDeclaration());
  EXPECT_NE(&structType2Jlm->GetDeclaration(), &structType3Jlm->GetDeclaration());

  const auto declarations = typeConverter.ReleaseStructTypeDeclarations();
  EXPECT_EQ(declarations.size(), 3);

  // We released all struct declarations. After that, translating the same type again should get
  // us a new declarations.
  const auto structType5Jlm =
      std::dynamic_pointer_cast<const StructType>(typeConverter.ConvertLlvmType(*structType1Llvm));

  EXPECT_NE(&structType5Jlm->GetDeclaration(), &structType1Jlm->GetDeclaration());
}

TEST(TypeConverterTests, LlvmArrayTypeConversion)
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
  EXPECT_NE(arrayType1Jlm, nullptr);
  EXPECT_TRUE(is<BitType>(arrayType1Jlm->element_type()));
  EXPECT_EQ(arrayType1Jlm->nelements(), 4);

  EXPECT_NE(arrayType2Jlm, nullptr);
  EXPECT_TRUE(is<FloatingPointType>(arrayType2Jlm->element_type()));
  EXPECT_EQ(arrayType2Jlm->nelements(), 9);
}

TEST(TypeConverterTests, LlvmVectorTypeConversion)
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
  const auto vectorType1Jlm = std::dynamic_pointer_cast<const FixedVectorType>(
      typeConverter.ConvertLlvmType(*vectorType1Llvm));
  const auto vectorType2Jlm = std::dynamic_pointer_cast<const ScalableVectorType>(
      typeConverter.ConvertLlvmType(*vectorType2Llvm));

  // Assert
  EXPECT_NE(vectorType1Jlm, nullptr);
  EXPECT_TRUE(is<BitType>(vectorType1Jlm->type()));
  EXPECT_EQ(vectorType1Jlm->size(), 4);

  EXPECT_NE(vectorType2Jlm, nullptr);
  EXPECT_TRUE(is<FloatingPointType>(vectorType2Jlm->type()));
  EXPECT_EQ(vectorType2Jlm->size(), 9);
}

TEST(TypeConverterTests, JLmBitTypeConversion)
{
  using namespace jlm::llvm;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto i1 = jlm::rvsdg::BitType::Create(1);
  const auto i2 = jlm::rvsdg::BitType::Create(2);
  const auto i4 = jlm::rvsdg::BitType::Create(4);
  const auto i8 = jlm::rvsdg::BitType::Create(8);
  const auto i16 = jlm::rvsdg::BitType::Create(16);
  const auto i32 = jlm::rvsdg::BitType::Create(32);
  const auto i64 = jlm::rvsdg::BitType::Create(64);

  // Act
  const auto i1Type = typeConverter.ConvertJlmType(*i1, context);
  const auto i2Type = typeConverter.ConvertJlmType(*i2, context);
  const auto i4Type = typeConverter.ConvertJlmType(*i4, context);
  const auto i8Type = typeConverter.ConvertJlmType(*i8, context);
  const auto i16Type = typeConverter.ConvertJlmType(*i16, context);
  const auto i32Type = typeConverter.ConvertJlmType(*i32, context);
  const auto i64Type = typeConverter.ConvertJlmType(*i64, context);

  // Assert
  EXPECT_EQ(i1Type->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(i1Type->getIntegerBitWidth(), 1);

  EXPECT_EQ(i2Type->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(i2Type->getIntegerBitWidth(), 2);

  EXPECT_EQ(i4Type->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(i4Type->getIntegerBitWidth(), 4);

  EXPECT_EQ(i8Type->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(i8Type->getIntegerBitWidth(), 8);

  EXPECT_EQ(i16Type->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(i16Type->getIntegerBitWidth(), 16);

  EXPECT_EQ(i32Type->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(i32Type->getIntegerBitWidth(), 32);

  EXPECT_EQ(i64Type->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(i64Type->getIntegerBitWidth(), 64);
}

TEST(TypeConverterTests, JlmFunctionTypeConversion)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  auto bit32Type = BitType::Create(32);
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
  EXPECT_NE(functionType1Llvm, nullptr);
  EXPECT_EQ(functionType1Llvm->getNumParams(), 2);
  EXPECT_EQ(functionType1Llvm->getParamType(0)->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(functionType1Llvm->getParamType(1)->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(functionType1Llvm->getReturnType()->getTypeID(), llvm::Type::VoidTyID);
  EXPECT_FALSE(functionType1Llvm->isVarArg());

  EXPECT_NE(functionType2Llvm, nullptr);
  EXPECT_EQ(functionType2Llvm->getNumParams(), 0);
  EXPECT_EQ(functionType2Llvm->getReturnType()->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_FALSE(functionType2Llvm->isVarArg());

  EXPECT_NE(functionType3Llvm, nullptr);
  EXPECT_EQ(functionType3Llvm->getNumParams(), 2);
  EXPECT_EQ(functionType3Llvm->getParamType(0)->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(functionType3Llvm->getParamType(1)->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(functionType3Llvm->getReturnType()->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_TRUE(functionType3Llvm->isVarArg());
}

TEST(TypeConverterTests, JlmPointerTypeConversion)
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
  EXPECT_NE(pointerTypeLlvm, nullptr);
  EXPECT_EQ(pointerTypeLlvm->getAddressSpace(), 0);
}

TEST(TypeConverterTests, JlmArrayTypeConversion)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto bit32Type = BitType::Create(32);
  const auto halfType = FloatingPointType::Create(fpsize::half);
  const auto arrayType1Jlm = ArrayType::Create(bit32Type, 4);
  const auto arrayType2Jlm = ArrayType::Create(halfType, 9);

  // Act
  const auto arrayType1Llvm = typeConverter.ConvertJlmType(*arrayType1Jlm, context);
  const auto arrayType2Llvm = typeConverter.ConvertJlmType(*arrayType2Jlm, context);

  // Assert
  EXPECT_TRUE(arrayType1Llvm->isArrayTy());
  EXPECT_EQ(arrayType1Llvm->getArrayNumElements(), 4);
  EXPECT_EQ(arrayType1Llvm->getArrayElementType()->getTypeID(), llvm::Type::IntegerTyID);

  EXPECT_TRUE(arrayType2Llvm->isArrayTy());
  EXPECT_EQ(arrayType2Llvm->getArrayNumElements(), 9);
  EXPECT_EQ(arrayType2Llvm->getArrayElementType()->getTypeID(), llvm::Type::HalfTyID);
}

TEST(TypeConverterTests, JlmControlTypeConversion)
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
  EXPECT_EQ(integerType1Llvm->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(integerType1Llvm->getIntegerBitWidth(), 1);

  EXPECT_EQ(integerType2Llvm->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(integerType2Llvm->getIntegerBitWidth(), 32);
}

TEST(TypeConverterTests, JlmFloatingPointTypeConversion)
{
  using namespace jlm::llvm;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto halfTypeJlm = FloatingPointType::Create(fpsize::half);
  const auto floatTypeJlm = FloatingPointType::Create(fpsize::flt);
  const auto doubleTypeJlm = FloatingPointType::Create(fpsize::dbl);
  const auto x86fp80TypeJlm = FloatingPointType::Create(fpsize::x86fp80);
  const auto fp128TypeJlm = FloatingPointType::Create(fpsize::fp128);

  // Act
  const auto halfTypeLlvm = typeConverter.ConvertJlmType(*halfTypeJlm, context);
  const auto floatTypeLlvm = typeConverter.ConvertJlmType(*floatTypeJlm, context);
  const auto doubleTypeLlvm = typeConverter.ConvertJlmType(*doubleTypeJlm, context);
  const auto x86fp80TypeLlvm = typeConverter.ConvertJlmType(*x86fp80TypeJlm, context);
  const auto fp128TypeLlvm = typeConverter.ConvertJlmType(*fp128TypeJlm, context);

  // Assert
  EXPECT_EQ(halfTypeLlvm->getTypeID(), llvm::Type::HalfTyID);
  EXPECT_EQ(floatTypeLlvm->getTypeID(), llvm::Type::FloatTyID);
  EXPECT_EQ(doubleTypeLlvm->getTypeID(), llvm::Type::DoubleTyID);
  EXPECT_EQ(x86fp80TypeLlvm->getTypeID(), llvm::Type::X86_FP80TyID);
  EXPECT_EQ(fp128TypeLlvm->getTypeID(), llvm::Type::FP128TyID);
}

TEST(TypeConverterTests, JlmStructTypeConversion)
{
  using namespace jlm::llvm;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto bit32Type = jlm::rvsdg::BitType::Create(32);
  const auto halfType = FloatingPointType::Create(fpsize::half);

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
  EXPECT_EQ(structType1Llvm->getTypeID(), llvm::Type::StructTyID);
  EXPECT_EQ(structType1Llvm->getStructNumElements(), 2);
  EXPECT_EQ(structType1Llvm->getStructElementType(0)->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(structType1Llvm->getStructElementType(1)->getTypeID(), llvm::Type::HalfTyID);
  EXPECT_FALSE(llvm::dyn_cast<llvm::StructType>(structType1Llvm)->isPacked());

  EXPECT_EQ(structType2Llvm->getTypeID(), llvm::Type::StructTyID);
  EXPECT_EQ(structType2Llvm->getStructNumElements(), 3);
  EXPECT_EQ(structType2Llvm->getStructElementType(0)->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(structType2Llvm->getStructElementType(1)->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(structType2Llvm->getStructElementType(2)->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_TRUE(llvm::dyn_cast<llvm::StructType>(structType2Llvm)->isPacked());

  EXPECT_EQ(structType3Llvm->getTypeID(), llvm::Type::StructTyID);
  EXPECT_EQ(structType3Llvm->getStructNumElements(), 1);
  EXPECT_EQ(structType3Llvm->getStructElementType(0)->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(structType3Llvm->getStructName(), "myStruct");
  EXPECT_TRUE(llvm::dyn_cast<llvm::StructType>(structType3Llvm)->isPacked());

  EXPECT_EQ(structType4Llvm, structType1Llvm);

  // The type converter created no jlm struct types. It is therefore not the owner of any
  // declarations.
  const auto declarations = typeConverter.ReleaseStructTypeDeclarations();
  EXPECT_EQ(declarations.size(), 0);

  // Converting the same type again after the declaration release should give us a new Llvm type
  const auto structType5Llvm = typeConverter.ConvertJlmType(*structType1Jlm, context);
  EXPECT_NE(structType5Llvm, structType1Llvm);
}

TEST(TypeConverterTests, JlmFixedVectorTypeConversion)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto bit32Type = BitType::Create(32);
  const auto fixedVectorType1 = FixedVectorType::Create(bit32Type, 2);
  const auto fixedVectorType2 = FixedVectorType::Create(bit32Type, 4);

  // Act
  const auto vectorType1 =
      llvm::dyn_cast<llvm::VectorType>(typeConverter.ConvertJlmType(*fixedVectorType1, context));
  const auto vectorType2 =
      llvm::dyn_cast<llvm::VectorType>(typeConverter.ConvertJlmType(*fixedVectorType2, context));

  // Assert
  EXPECT_EQ(vectorType1->getTypeID(), llvm::Type::FixedVectorTyID);
  EXPECT_EQ(vectorType1->getElementType()->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(vectorType1->getElementCount().getFixedValue(), 2);

  EXPECT_EQ(vectorType2->getTypeID(), llvm::Type::FixedVectorTyID);
  EXPECT_EQ(vectorType2->getElementType()->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(vectorType2->getElementCount().getFixedValue(), 4);
}

TEST(TypeConverterTests, JlmScalableVectorTypeConversion)
{
  using namespace jlm::llvm;
  using namespace jlm::rvsdg;

  // Arrange
  llvm::LLVMContext context;
  TypeConverter typeConverter;

  const auto bit32Type = BitType::Create(32);
  const auto scalableVectorType1 = ScalableVectorType::Create(bit32Type, 2);
  const auto scalableVectorType2 = ScalableVectorType::Create(bit32Type, 4);

  // Act
  const auto vectorType1 =
      llvm::dyn_cast<llvm::VectorType>(typeConverter.ConvertJlmType(*scalableVectorType1, context));
  const auto vectorType2 =
      llvm::dyn_cast<llvm::VectorType>(typeConverter.ConvertJlmType(*scalableVectorType2, context));

  // Assert
  EXPECT_EQ(vectorType1->getTypeID(), llvm::Type::ScalableVectorTyID);
  EXPECT_EQ(vectorType1->getElementType()->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(vectorType1->getElementCount().getKnownMinValue(), 2);

  EXPECT_EQ(vectorType2->getTypeID(), llvm::Type::ScalableVectorTyID);
  EXPECT_EQ(vectorType2->getElementType()->getTypeID(), llvm::Type::IntegerTyID);
  EXPECT_EQ(vectorType2->getElementCount().getKnownMinValue(), 4);
}
