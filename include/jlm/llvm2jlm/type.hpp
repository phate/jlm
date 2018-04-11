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

std::unique_ptr<jive::valuetype>
convert_type(const llvm::Type * type, context & ctx);

static inline std::unique_ptr<jlm::arraytype>
convert_arraytype(const llvm::ArrayType * type, context & ctx)
{
	auto t = convert_type(llvm::cast<llvm::Type>(type), ctx);
	JLM_DEBUG_ASSERT(is_arraytype(*t));
	return std::unique_ptr<jlm::arraytype>(static_cast<jlm::arraytype*>(t.release()));
}

static inline std::unique_ptr<jive::fct::type>
convert_type(const llvm::FunctionType * type, context & ctx)
{
	auto t = convert_type(llvm::cast<llvm::Type>(type), ctx);
	JLM_DEBUG_ASSERT(dynamic_cast<const jive::fct::type*>(t.get()));
	return std::unique_ptr<jive::fct::type>(static_cast<jive::fct::type*>(t.release()));
}

static inline std::unique_ptr<structtype>
convert_type(const llvm::StructType * type, context & ctx)
{
	auto t = convert_type(llvm::cast<llvm::Type>(type), ctx);
	JLM_DEBUG_ASSERT(dynamic_cast<const structtype*>(t.get()));
	return std::unique_ptr<structtype>(static_cast<structtype*>(t.release()));
}

}

#endif
