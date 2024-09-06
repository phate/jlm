/*
 * Copyright 2024 Halvor Linder Henriksen <halvor_lh@hotmail.no>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-util.hpp>

#include <jlm/llvm/frontend/LlvmConversionContext.hpp>
#include <jlm/llvm/frontend/LlvmTypeConversion.hpp>

static void
TestTypeConversion(
    jlm::llvm::context & jlm_context,
    llvm::Type * llvm_type,
    jlm::llvm::fpsize jlm_type_size)
{
  using namespace llvm;

  auto jlm_type = jlm::llvm::ConvertType(llvm_type, jlm_context);
  auto floating_point_type = dynamic_cast<const jlm::llvm::fptype *>(jlm_type.get());

  assert(floating_point_type && floating_point_type->size() == jlm_type_size);
}

static int
TypeConversion()
{
  using namespace jlm::llvm;

  llvm::LLVMContext llvm_ctx;
  llvm::Module lm("module", llvm_ctx);

  ipgraph_module im(jlm::util::filepath(""), "", "");
  auto jlm_ctx = context(im);

  TestTypeConversion(jlm_ctx, ::llvm::Type::getHalfTy(llvm_ctx), jlm::llvm::fpsize::half);
  TestTypeConversion(jlm_ctx, ::llvm::Type::getFloatTy(llvm_ctx), jlm::llvm::fpsize::flt);
  TestTypeConversion(jlm_ctx, ::llvm::Type::getDoubleTy(llvm_ctx), jlm::llvm::fpsize::dbl);
  TestTypeConversion(jlm_ctx, ::llvm::Type::getX86_FP80Ty(llvm_ctx), jlm::llvm::fpsize::x86fp80);
  TestTypeConversion(jlm_ctx, ::llvm::Type::getFP128Ty(llvm_ctx), jlm::llvm::fpsize::fp128);

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/llvm/LlvmTypeConversionTests-TypeConversion",
    TypeConversion)
