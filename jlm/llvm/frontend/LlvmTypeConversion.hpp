/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_FRONTEND_LLVMTYPECONVERSION_HPP
#define JLM_LLVM_FRONTEND_LLVMTYPECONVERSION_HPP

#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/record.hpp>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

#include <memory>

namespace jive {
namespace base {
	class type;
}
}

namespace llvm {
	class ArrayType;
	class Type;
}

namespace jlm {

class context;

jlm::fpsize
ExtractFloatingPointSize(const llvm::Type * type);

std::unique_ptr<jive::valuetype>
ConvertType(const llvm::Type * type, context & ctx);

static inline std::unique_ptr<FunctionType>
ConvertFunctionType(const llvm::FunctionType * type, context & ctx)
{
	auto t = ConvertType(llvm::cast<llvm::Type>(type), ctx);
	JLM_ASSERT(dynamic_cast<const FunctionType*>(t.get()));
	return std::unique_ptr<FunctionType>(static_cast<FunctionType*>(t.release()));
}

static inline std::unique_ptr<PointerType>
ConvertPointerType(const llvm::PointerType * type, context & ctx)
{
	auto t = ConvertType(llvm::cast<llvm::Type>(type), ctx);
	JLM_ASSERT(dynamic_cast<const PointerType*>(t.get()));
	return std::unique_ptr<PointerType>(static_cast<PointerType*>(t.release()));
}

}

#endif
