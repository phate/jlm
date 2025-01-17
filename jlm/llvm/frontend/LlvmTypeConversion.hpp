/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_FRONTEND_LLVMTYPECONVERSION_HPP
#define JLM_LLVM_FRONTEND_LLVMTYPECONVERSION_HPP

#include <jlm/llvm/ir/types.hpp>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

#include <memory>

namespace llvm
{
class ArrayType;
class Type;
}

namespace jlm::llvm
{

class context;

fpsize
ExtractFloatingPointSize(const ::llvm::Type * type);

std::shared_ptr<const rvsdg::ValueType>
ConvertType(const ::llvm::Type * type, context & ctx);

static inline std::shared_ptr<const rvsdg::FunctionType>
ConvertFunctionType(const ::llvm::FunctionType * type, context & ctx)
{
  auto t = ConvertType(::llvm::cast<::llvm::Type>(type), ctx);
  JLM_ASSERT(dynamic_cast<const rvsdg::FunctionType *>(t.get()));
  return std::dynamic_pointer_cast<const rvsdg::FunctionType>(t);
}

static inline std::shared_ptr<const PointerType>
ConvertPointerType(const ::llvm::PointerType * type, context & ctx)
{
  auto t = ConvertType(::llvm::cast<::llvm::Type>(type), ctx);
  JLM_ASSERT(dynamic_cast<const PointerType *>(t.get()));
  return std::dynamic_pointer_cast<const PointerType>(t);
}

}

#endif
