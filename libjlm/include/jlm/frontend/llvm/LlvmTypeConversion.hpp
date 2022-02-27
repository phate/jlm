/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_FRONTEND_LLVM_LLVM2JLM_LLVMTYPECONVERSION_HPP
#define JLM_FRONTEND_LLVM_LLVM2JLM_LLVMTYPECONVERSION_HPP

#include <jive/types/record.hpp>

#include <jlm/ir/types.hpp>

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
convert_fpsize(const llvm::Type * type);

std::unique_ptr<jive::valuetype>
ConvertType(const llvm::Type * type, context & ctx);

static inline std::unique_ptr<jlm::arraytype>
convert_arraytype(const llvm::ArrayType * type, context & ctx)
{
	auto t = ConvertType(llvm::cast<llvm::Type>(type), ctx);
	JLM_ASSERT(jive::is<arraytype>(*t));
	return std::unique_ptr<jlm::arraytype>(static_cast<jlm::arraytype*>(t.release()));
}

static inline std::unique_ptr<FunctionType>
convert_type(const llvm::FunctionType * type, context & ctx)
{
	auto t = ConvertType(llvm::cast<llvm::Type>(type), ctx);
	JLM_ASSERT(dynamic_cast<const FunctionType*>(t.get()));
	return std::unique_ptr<FunctionType>(static_cast<FunctionType*>(t.release()));
}

static inline std::unique_ptr<structtype>
convert_type(const llvm::StructType * type, context & ctx)
{
	auto t = ConvertType(llvm::cast<llvm::Type>(type), ctx);
	JLM_ASSERT(dynamic_cast<const structtype*>(t.get()));
	return std::unique_ptr<structtype>(static_cast<structtype*>(t.release()));
}

static inline std::unique_ptr<vectortype>
convert_type(const llvm::VectorType * type, context & ctx)
{
	auto t = ConvertType(llvm::cast<llvm::Type>(type), ctx);
	JLM_ASSERT(dynamic_cast<const vectortype*>(t.get()));
	return std::unique_ptr<vectortype>(static_cast<vectortype*>(t.release()));
}

static inline std::unique_ptr<ptrtype>
convert_type(const llvm::PointerType * type, context & ctx)
{
	auto t = ConvertType(llvm::cast<llvm::Type>(type), ctx);
	JLM_ASSERT(dynamic_cast<const ptrtype*>(t.get()));
	return std::unique_ptr<ptrtype>(static_cast<ptrtype*>(t.release()));
}

}

#endif
