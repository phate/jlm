/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_FRONTEND_LLVMMODULECONVERSION_HPP
#define JLM_LLVM_FRONTEND_LLVMMODULECONVERSION_HPP

#include <jlm/llvm/ir/attribute.hpp>

#include <llvm/IR/Attributes.h>

#include <memory>

namespace llvm
{
class Module;
}

namespace jlm::llvm
{

class InterProceduralGraphModule;

Attribute::kind
ConvertAttributeKind(const ::llvm::Attribute::AttrKind & kind);

std::unique_ptr<InterProceduralGraphModule>
ConvertLlvmModule(::llvm::Module & module);

}

#endif
