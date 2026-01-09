/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_IR_TYPECONVERTER_HPP
#define JLM_LLVM_IR_TYPECONVERTER_HPP

#include <jlm/llvm/ir/types.hpp>
#include <jlm/util/BijectiveMap.hpp>

#include <vector>

namespace llvm
{
class ArrayType;
class FunctionType;
class IntegerType;
class LLVMContext;
class PointerType;
class StructType;
class Type;
}

namespace jlm::rvsdg
{
class BitType;
class ControlType;
class FunctionType;
class Type;
}

namespace jlm::llvm
{

/**
 * Converts Llvm to Jlm types and vice versa.
 */
class TypeConverter final
{
public:
  TypeConverter() = default;

  TypeConverter(const TypeConverter &) = delete;

  TypeConverter(const TypeConverter &&) = delete;

  TypeConverter &
  operator=(const TypeConverter &) = delete;

  TypeConverter &
  operator=(const TypeConverter &&) = delete;

  static fpsize
  ExtractFloatingPointSize(const ::llvm::Type & type);

  static ::llvm::IntegerType *
  ConvertBitType(const rvsdg::BitType & bitType, ::llvm::LLVMContext & context);

  ::llvm::FunctionType *
  ConvertFunctionType(const rvsdg::FunctionType & functionType, ::llvm::LLVMContext & context);

  static ::llvm::PointerType *
  ConvertPointerType(const PointerType & type, ::llvm::LLVMContext & context);

  ::llvm::StructType *
  ConvertStructType(const StructType & type, ::llvm::LLVMContext & context);

  ::llvm::ArrayType *
  ConvertArrayType(const ArrayType & type, ::llvm::LLVMContext & context);

  ::llvm::Type *
  ConvertJlmType(const rvsdg::Type & type, ::llvm::LLVMContext & context);

  std::shared_ptr<const rvsdg::FunctionType>
  ConvertFunctionType(const ::llvm::FunctionType & functionType);

  static std::shared_ptr<const PointerType>
  ConvertPointerType(const ::llvm::PointerType & pointerType);

  std::shared_ptr<const rvsdg::Type>
  ConvertLlvmType(::llvm::Type & type);

private:
  static ::llvm::Type *
  ConvertFloatingPointType(const FloatingPointType & type, ::llvm::LLVMContext & context);

  std::unique_ptr<StructType::Declaration>
  CreateStructDeclaration(::llvm::StructType & structType);

  // Mapping between LLVM's StructType and jlm's StructType
  util::BijectiveMap<::llvm::StructType *, std::shared_ptr<const StructType>> StructTypeMap_;
};

}

#endif // JLM_LLVM_IR_TYPECONVERTER_HPP
