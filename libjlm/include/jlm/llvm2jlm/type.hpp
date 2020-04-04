/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM2JLM_TYPE_HPP
#define JLM_LLVM2JLM_TYPE_HPP

#include <jive/types/record.h>

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
convert_type(const llvm::Type * type, context & ctx);

static inline std::unique_ptr<jlm::arraytype>
convert_arraytype(const llvm::ArrayType * type, context & ctx)
{
	auto t = convert_type(llvm::cast<llvm::Type>(type), ctx);
	JLM_DEBUG_ASSERT(is<arraytype>(*t));
	return std::unique_ptr<jlm::arraytype>(static_cast<jlm::arraytype*>(t.release()));
}

static inline std::unique_ptr<jive::fcttype>
convert_type(const llvm::FunctionType * type, context & ctx)
{
	auto t = convert_type(llvm::cast<llvm::Type>(type), ctx);
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::fcttype*>(t.get()));
	return std::unique_ptr<jive::fcttype>(static_cast<jive::fcttype*>(t.release()));
}

static inline std::unique_ptr<structtype>
convert_type(const llvm::StructType * type, context & ctx)
{
	auto t = convert_type(llvm::cast<llvm::Type>(type), ctx);
	JLM_DEBUG_ASSERT(dynamic_cast<const structtype*>(t.get()));
	return std::unique_ptr<structtype>(static_cast<structtype*>(t.release()));
}

static inline std::unique_ptr<vectortype>
convert_type(const llvm::VectorType * type, context & ctx)
{
	auto t = convert_type(llvm::cast<llvm::Type>(type), ctx);
	JLM_DEBUG_ASSERT(dynamic_cast<const vectortype*>(t.get()));
	return std::unique_ptr<vectortype>(static_cast<vectortype*>(t.release()));
}

static inline std::unique_ptr<ptrtype>
convert_type(const llvm::PointerType * type, context & ctx)
{
	auto t = convert_type(llvm::cast<llvm::Type>(type), ctx);
	JLM_DEBUG_ASSERT(dynamic_cast<const ptrtype*>(t.get()));
	return std::unique_ptr<ptrtype>(static_cast<ptrtype*>(t.release()));
}

}

#endif
