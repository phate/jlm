/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/control.hpp>

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <jlm/llvm/backend/jlm2llvm/context.hpp>
#include <jlm/llvm/backend/jlm2llvm/instruction.hpp>
#include <jlm/llvm/backend/jlm2llvm/jlm2llvm.hpp>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <deque>
#include <unordered_map>

namespace jlm::llvm
{

namespace jlm2llvm
{

static const llvm::tac *
get_match(const llvm::tac * branch)
{
  JLM_ASSERT(is<tacvariable>(branch->operand(0)));
  auto tv = static_cast<const tacvariable *>(branch->operand(0));
  return tv->tac();
}

static bool
has_return_value(const llvm::cfg & cfg)
{
  for (size_t n = 0; n < cfg.exit()->nresults(); n++)
  {
    auto result = cfg.exit()->result(n);
    if (rvsdg::is<rvsdg::ValueType>(result->type()))
      return true;
  }

  return false;
}

static void
create_return(const cfg_node * node, context & ctx)
{
  JLM_ASSERT(node->NumOutEdges() == 1);
  JLM_ASSERT(node->OutEdge(0)->sink() == node->cfg().exit());
  ::llvm::IRBuilder<> builder(ctx.basic_block(node));
  auto & cfg = node->cfg();

  /* return without result */
  if (!has_return_value(cfg))
  {
    builder.CreateRetVoid();
    return;
  }

  auto result = cfg.exit()->result(0);
  JLM_ASSERT(rvsdg::is<rvsdg::ValueType>(result->type()));
  builder.CreateRet(ctx.value(result));
}

static void
create_unconditional_branch(const cfg_node * node, context & ctx)
{
  JLM_ASSERT(node->NumOutEdges() == 1);
  JLM_ASSERT(node->OutEdge(0)->sink() != node->cfg().exit());
  ::llvm::IRBuilder<> builder(ctx.basic_block(node));
  auto target = node->OutEdge(0)->sink();

  builder.CreateBr(ctx.basic_block(target));
}

static void
create_conditional_branch(const cfg_node * node, context & ctx)
{
  JLM_ASSERT(node->NumOutEdges() == 2);
  JLM_ASSERT(node->OutEdge(0)->sink() != node->cfg().exit());
  JLM_ASSERT(node->OutEdge(1)->sink() != node->cfg().exit());
  ::llvm::IRBuilder<> builder(ctx.basic_block(node));

  auto branch = static_cast<const basic_block *>(node)->tacs().last();
  JLM_ASSERT(branch && is<branch_op>(branch));
  JLM_ASSERT(ctx.value(branch->operand(0))->getType()->isIntegerTy(1));

  auto condition = ctx.value(branch->operand(0));
  auto bbfalse = ctx.basic_block(node->OutEdge(0)->sink());
  auto bbtrue = ctx.basic_block(node->OutEdge(1)->sink());
  builder.CreateCondBr(condition, bbtrue, bbfalse);
}

static void
create_switch(const cfg_node * node, context & ctx)
{
  JLM_ASSERT(node->NumOutEdges() >= 2);
  ::llvm::LLVMContext & llvmContext = ctx.llvm_module().getContext();
  auto & typeConverter = ctx.GetTypeConverter();
  auto bb = static_cast<const basic_block *>(node);
  ::llvm::IRBuilder<> builder(ctx.basic_block(node));

  auto branch = bb->tacs().last();
  JLM_ASSERT(branch && is<branch_op>(branch));
  auto condition = ctx.value(branch->operand(0));
  auto match = get_match(branch);

  if (is<rvsdg::match_op>(match))
  {
    JLM_ASSERT(match->result(0) == branch->operand(0));
    auto mop = static_cast<const rvsdg::match_op *>(&match->operation());

    auto defbb = ctx.basic_block(node->OutEdge(mop->default_alternative())->sink());
    auto sw = builder.CreateSwitch(condition, defbb);
    for (const auto & alt : *mop)
    {
      auto & type = *std::static_pointer_cast<const rvsdg::bittype>(mop->argument(0));
      auto value =
          ::llvm::ConstantInt::get(typeConverter.ConvertBitType(type, llvmContext), alt.first);
      sw->addCase(value, ctx.basic_block(node->OutEdge(alt.second)->sink()));
    }
  }
  else
  {
    auto defbb = ctx.basic_block(node->OutEdge(node->NumOutEdges() - 1)->sink());
    auto sw = builder.CreateSwitch(condition, defbb);
    for (size_t n = 0; n < node->NumOutEdges() - 1; n++)
    {
      auto value = ::llvm::ConstantInt::get(::llvm::Type::getInt32Ty(builder.getContext()), n);
      sw->addCase(value, ctx.basic_block(node->OutEdge(n)->sink()));
    }
  }
}

static void
create_terminator_instruction(const llvm::cfg_node * node, context & ctx)
{
  JLM_ASSERT(is<basic_block>(node));
  auto & tacs = static_cast<const basic_block *>(node)->tacs();
  auto & cfg = node->cfg();

  /* unconditional branch or return statement */
  if (node->NumOutEdges() == 1)
  {
    auto target = node->OutEdge(0)->sink();
    if (target == cfg.exit())
      return create_return(node, ctx);

    return create_unconditional_branch(node, ctx);
  }

  auto branch = tacs.last();
  JLM_ASSERT(branch && is<branch_op>(branch));

  /* conditional branch */
  if (ctx.value(branch->operand(0))->getType()->isIntegerTy(1))
    return create_conditional_branch(node, ctx);

  /* switch */
  create_switch(node, ctx);
}

::llvm::Attribute::AttrKind
convert_attribute_kind(const attribute::kind & kind)
{
  typedef ::llvm::Attribute::AttrKind ak;

  static std::unordered_map<attribute::kind, ::llvm::Attribute::AttrKind> map(
      { { attribute::kind::None, ak::None },

        { attribute::kind::FirstEnumAttr, ak::FirstEnumAttr },
        { attribute::kind::AllocAlign, ak::AllocAlign },
        { attribute::kind::AllocatedPointer, ak::AllocatedPointer },
        { attribute::kind::AlwaysInline, ak::AlwaysInline },
        { attribute::kind::Builtin, ak::Builtin },
        { attribute::kind::Cold, ak::Cold },
        { attribute::kind::Convergent, ak::Convergent },
        { attribute::kind::CoroDestroyOnlyWhenComplete, ak::CoroDestroyOnlyWhenComplete },
        { attribute::kind::DeadOnUnwind, ak::DeadOnUnwind },
        { attribute::kind::DisableSanitizerInstrumentation, ak::DisableSanitizerInstrumentation },
        { attribute::kind::FnRetThunkExtern, ak::FnRetThunkExtern },
        { attribute::kind::Hot, ak::Hot },
        { attribute::kind::ImmArg, ak::ImmArg },
        { attribute::kind::InReg, ak::InReg },
        { attribute::kind::InlineHint, ak::InlineHint },
        { attribute::kind::JumpTable, ak::JumpTable },
        { attribute::kind::Memory, ak::Memory },
        { attribute::kind::MinSize, ak::MinSize },
        { attribute::kind::MustProgress, ak::MustProgress },
        { attribute::kind::Naked, ak::Naked },
        { attribute::kind::Nest, ak::Nest },
        { attribute::kind::NoAlias, ak::NoAlias },
        { attribute::kind::NoBuiltin, ak::NoBuiltin },
        { attribute::kind::NoCallback, ak::NoCallback },
        { attribute::kind::NoCapture, ak::NoCapture },
        { attribute::kind::NoCfCheck, ak::NoCfCheck },
        { attribute::kind::NoDuplicate, ak::NoDuplicate },
        { attribute::kind::NoFree, ak::NoFree },
        { attribute::kind::NoImplicitFloat, ak::NoImplicitFloat },
        { attribute::kind::NoInline, ak::NoInline },
        { attribute::kind::NoMerge, ak::NoMerge },
        { attribute::kind::NoProfile, ak::NoProfile },
        { attribute::kind::NoRecurse, ak::NoRecurse },
        { attribute::kind::NoRedZone, ak::NoRedZone },
        { attribute::kind::NoReturn, ak::NoReturn },
        { attribute::kind::NoSanitizeBounds, ak::NoSanitizeBounds },
        { attribute::kind::NoSanitizeCoverage, ak::NoSanitizeCoverage },
        { attribute::kind::NoSync, ak::NoSync },
        { attribute::kind::NoUndef, ak::NoUndef },
        { attribute::kind::NoUnwind, ak::NoUnwind },
        { attribute::kind::NonLazyBind, ak::NonLazyBind },
        { attribute::kind::NonNull, ak::NonNull },
        { attribute::kind::NullPointerIsValid, ak::NullPointerIsValid },
        { attribute::kind::OptForFuzzing, ak::OptForFuzzing },
        { attribute::kind::OptimizeForDebugging, ak::OptimizeForDebugging },
        { attribute::kind::OptimizeForSize, ak::OptimizeForSize },
        { attribute::kind::OptimizeNone, ak::OptimizeNone },
        { attribute::kind::PresplitCoroutine, ak::PresplitCoroutine },
        { attribute::kind::ReadNone, ak::ReadNone },
        { attribute::kind::ReadOnly, ak::ReadOnly },
        { attribute::kind::Returned, ak::Returned },
        { attribute::kind::ReturnsTwice, ak::ReturnsTwice },
        { attribute::kind::SExt, ak::SExt },
        { attribute::kind::SafeStack, ak::SafeStack },
        { attribute::kind::SanitizeAddress, ak::SanitizeAddress },
        { attribute::kind::SanitizeHWAddress, ak::SanitizeHWAddress },
        { attribute::kind::SanitizeMemTag, ak::SanitizeMemTag },
        { attribute::kind::SanitizeMemory, ak::SanitizeMemory },
        { attribute::kind::SanitizeThread, ak::SanitizeThread },
        { attribute::kind::ShadowCallStack, ak::ShadowCallStack },
        { attribute::kind::SkipProfile, ak::SkipProfile },
        { attribute::kind::Speculatable, ak::Speculatable },
        { attribute::kind::SpeculativeLoadHardening, ak::SpeculativeLoadHardening },
        { attribute::kind::StackProtect, ak::StackProtect },
        { attribute::kind::StackProtectReq, ak::StackProtectReq },
        { attribute::kind::StackProtectStrong, ak::StackProtectStrong },
        { attribute::kind::StrictFP, ak::StrictFP },
        { attribute::kind::SwiftAsync, ak::SwiftAsync },
        { attribute::kind::SwiftError, ak::SwiftError },
        { attribute::kind::SwiftSelf, ak::SwiftSelf },
        { attribute::kind::WillReturn, ak::WillReturn },
        { attribute::kind::Writable, ak::Writable },
        { attribute::kind::WriteOnly, ak::WriteOnly },
        { attribute::kind::ZExt, ak::ZExt },
        { attribute::kind::LastEnumAttr, ak::LastEnumAttr },

        { attribute::kind::FirstTypeAttr, ak::FirstTypeAttr },
        { attribute::kind::ByRef, ak::ByRef },
        { attribute::kind::ByVal, ak::ByVal },
        { attribute::kind::ElementType, ak::ElementType },
        { attribute::kind::InAlloca, ak::InAlloca },
        { attribute::kind::Preallocated, ak::Preallocated },
        { attribute::kind::StructRet, ak::StructRet },
        { attribute::kind::LastTypeAttr, ak::LastTypeAttr },

        { attribute::kind::FirstIntAttr, ak::FirstIntAttr },
        { attribute::kind::Alignment, ak::Alignment },
        { attribute::kind::AllocKind, ak::AllocKind },
        { attribute::kind::AllocSize, ak::AllocSize },
        { attribute::kind::Dereferenceable, ak::Dereferenceable },
        { attribute::kind::DereferenceableOrNull, ak::DereferenceableOrNull },
        { attribute::kind::NoFPClass, ak::NoFPClass },
        { attribute::kind::StackAlignment, ak::StackAlignment },
        { attribute::kind::UWTable, ak::UWTable },
        { attribute::kind::VScaleRange, ak::VScaleRange },
        { attribute::kind::LastIntAttr, ak::LastIntAttr },

        { attribute::kind::EndAttrKinds, ak::EndAttrKinds } });

  JLM_ASSERT(map.find(kind) != map.end());
  return map[kind];
}

static ::llvm::Attribute
ConvertEnumAttribute(const llvm::enum_attribute & attribute, context & ctx)
{
  auto & llvmContext = ctx.llvm_module().getContext();
  auto kind = convert_attribute_kind(attribute.kind());
  return ::llvm::Attribute::get(llvmContext, kind);
}

static ::llvm::Attribute
ConvertIntAttribute(const llvm::int_attribute & attribute, context & ctx)
{
  auto & llvmContext = ctx.llvm_module().getContext();
  auto kind = convert_attribute_kind(attribute.kind());
  return ::llvm::Attribute::get(llvmContext, kind, attribute.value());
}

static ::llvm::Attribute
ConvertTypeAttribute(const llvm::type_attribute & attribute, context & ctx)
{
  auto & typeConverter = ctx.GetTypeConverter();
  auto & llvmContext = ctx.llvm_module().getContext();

  auto kind = convert_attribute_kind(attribute.kind());
  auto type = typeConverter.ConvertJlmType(attribute.type(), llvmContext);
  return ::llvm::Attribute::get(llvmContext, kind, type);
}

static ::llvm::Attribute
ConvertStringAttribute(const llvm::string_attribute & attribute, context & ctx)
{
  auto & llvmContext = ctx.llvm_module().getContext();
  return ::llvm::Attribute::get(llvmContext, attribute.kind(), attribute.value());
}

static ::llvm::AttributeSet
convert_attributes(const attributeset & attributeSet, context & ctx)
{
  ::llvm::AttrBuilder builder(ctx.llvm_module().getContext());
  for (auto & attribute : attributeSet.EnumAttributes())
    builder.addAttribute(ConvertEnumAttribute(attribute, ctx));

  for (auto & attribute : attributeSet.IntAttributes())
    builder.addAttribute(ConvertIntAttribute(attribute, ctx));

  for (auto & attribute : attributeSet.TypeAttributes())
    builder.addAttribute(ConvertTypeAttribute(attribute, ctx));

  for (auto & attribute : attributeSet.StringAttributes())
    builder.addAttribute(ConvertStringAttribute(attribute, ctx));

  return ::llvm::AttributeSet::get(ctx.llvm_module().getContext(), builder);
}

static ::llvm::AttributeList
convert_attributes(const function_node & f, context & ctx)
{
  JLM_ASSERT(f.cfg());

  auto & llvmctx = ctx.llvm_module().getContext();

  auto fctset = convert_attributes(f.attributes(), ctx);
  /*
    FIXME: return value attributes are currently not supported
  */
  auto retset = ::llvm::AttributeSet();

  std::vector<::llvm::AttributeSet> argsets;
  for (size_t n = 0; n < f.cfg()->entry()->narguments(); n++)
  {
    auto argument = f.cfg()->entry()->argument(n);

    if (rvsdg::is<rvsdg::StateType>(argument->type()))
      continue;

    argsets.push_back(convert_attributes(argument->attributes(), ctx));
  }

  return ::llvm::AttributeList::get(llvmctx, fctset, retset, argsets);
}

static std::vector<cfg_node *>
ConvertBasicBlocks(
    const llvm::cfg & controlFlowGraph,
    ::llvm::Function & function,
    jlm2llvm::context & context)
{
  auto nodes = breadth_first(controlFlowGraph);

  uint64_t basicBlockCounter = 0;
  for (const auto & node : nodes)
  {
    if (node == controlFlowGraph.entry())
      continue;
    if (node == controlFlowGraph.exit())
      continue;

    auto name = util::strfmt("bb", basicBlockCounter++);
    auto * basicBlock = ::llvm::BasicBlock::Create(function.getContext(), name, &function);
    context.insert(node, basicBlock);
  }

  return nodes;
}

static inline void
convert_cfg(llvm::cfg & cfg, ::llvm::Function & f, context & ctx)
{
  JLM_ASSERT(is_closed(cfg));

  auto add_arguments = [](const llvm::cfg & cfg, ::llvm::Function & f, context & ctx)
  {
    size_t n = 0;
    for (auto & llvmarg : f.args())
    {
      auto jlmarg = cfg.entry()->argument(n++);
      ctx.insert(jlmarg, &llvmarg);
    }
  };

  straighten(cfg);

  auto nodes = ConvertBasicBlocks(cfg, f, ctx);

  add_arguments(cfg, f, ctx);

  /* create non-terminator instructions */
  for (const auto & node : nodes)
  {
    if (node == cfg.entry() || node == cfg.exit())
      continue;

    JLM_ASSERT(is<basic_block>(node));
    auto & tacs = static_cast<const basic_block *>(node)->tacs();
    for (const auto & tac : tacs)
      convert_instruction(*tac, node, ctx);
  }

  /* create cfg structure */
  for (const auto & node : nodes)
  {
    if (node == cfg.entry() || node == cfg.exit())
      continue;

    create_terminator_instruction(node, ctx);
  }

  /* patch phi instructions */
  for (const auto & node : nodes)
  {
    if (node == cfg.entry() || node == cfg.exit())
      continue;

    JLM_ASSERT(is<basic_block>(node));
    auto & tacs = static_cast<const basic_block *>(node)->tacs();
    for (const auto & tac : tacs)
    {
      if (!is<SsaPhiOperation>(tac->operation()))
        continue;

      if (rvsdg::is<IOStateType>(tac->result(0)->type()))
        continue;
      if (rvsdg::is<MemoryStateType>(tac->result(0)->type()))
        continue;

      JLM_ASSERT(node->NumInEdges() == tac->noperands());
      auto & op = *static_cast<const SsaPhiOperation *>(&tac->operation());
      auto phi = ::llvm::dyn_cast<::llvm::PHINode>(ctx.value(tac->result(0)));
      for (size_t n = 0; n < tac->noperands(); n++)
        phi->addIncoming(ctx.value(tac->operand(n)), ctx.basic_block(op.node(n)));
    }
  }
}

static inline void
convert_function(const function_node & node, context & ctx)
{
  if (!node.cfg())
    return;

  auto & im = ctx.module();
  auto f = ::llvm::cast<::llvm::Function>(ctx.value(im.variable(&node)));

  auto attributes = convert_attributes(node, ctx);
  f->setAttributes(attributes);

  convert_cfg(*node.cfg(), *f, ctx);
}

static void
convert_data_node(const data_node & node, context & ctx)
{
  if (!node.initialization())
    return;

  auto & jm = ctx.module();
  auto init = node.initialization();
  convert_tacs(init->tacs(), ctx);

  auto gv = ::llvm::dyn_cast<::llvm::GlobalVariable>(ctx.value(jm.variable(&node)));
  gv->setInitializer(::llvm::dyn_cast<::llvm::Constant>(ctx.value(init->value())));
}

static const ::llvm::GlobalValue::LinkageTypes &
convert_linkage(const llvm::linkage & linkage)
{
  static std::unordered_map<llvm::linkage, ::llvm::GlobalValue::LinkageTypes> map(
      { { llvm::linkage::external_linkage, ::llvm::GlobalValue::ExternalLinkage },
        { llvm::linkage::available_externally_linkage,
          ::llvm::GlobalValue::AvailableExternallyLinkage },
        { llvm::linkage::link_once_any_linkage, ::llvm::GlobalValue::LinkOnceAnyLinkage },
        { llvm::linkage::link_once_odr_linkage, ::llvm::GlobalValue::LinkOnceODRLinkage },
        { llvm::linkage::weak_any_linkage, ::llvm::GlobalValue::WeakAnyLinkage },
        { llvm::linkage::weak_odr_linkage, ::llvm::GlobalValue::WeakODRLinkage },
        { llvm::linkage::appending_linkage, ::llvm::GlobalValue::AppendingLinkage },
        { llvm::linkage::internal_linkage, ::llvm::GlobalValue::InternalLinkage },
        { llvm::linkage::private_linkage, ::llvm::GlobalValue::PrivateLinkage },
        { llvm::linkage::external_weak_linkage, ::llvm::GlobalValue::ExternalWeakLinkage },
        { llvm::linkage::common_linkage, ::llvm::GlobalValue::CommonLinkage } });

  JLM_ASSERT(map.find(linkage) != map.end());
  return map[linkage];
}

static void
convert_ipgraph(context & ctx)
{
  auto & typeConverter = ctx.GetTypeConverter();
  auto & jm = ctx.module();
  auto & lm = ctx.llvm_module();

  /* forward declare all nodes */
  for (const auto & node : jm.ipgraph())
  {
    auto v = jm.variable(&node);

    if (auto dataNode = dynamic_cast<const data_node *>(&node))
    {
      auto type = typeConverter.ConvertJlmType(*dataNode->GetValueType(), lm.getContext());
      auto linkage = convert_linkage(dataNode->linkage());

      auto gv = new ::llvm::GlobalVariable(
          lm,
          type,
          dataNode->constant(),
          linkage,
          nullptr,
          dataNode->name());
      gv->setSection(dataNode->Section());
      ctx.insert(v, gv);
    }
    else if (auto n = dynamic_cast<const function_node *>(&node))
    {
      auto type = typeConverter.ConvertFunctionType(n->fcttype(), lm.getContext());
      auto linkage = convert_linkage(n->linkage());
      auto f = ::llvm::Function::Create(type, linkage, n->name(), &lm);
      ctx.insert(v, f);
    }
    else
      JLM_ASSERT(0);
  }

  /* convert all nodes */
  for (const auto & node : jm.ipgraph())
  {
    if (auto n = dynamic_cast<const data_node *>(&node))
    {
      convert_data_node(*n, ctx);
    }
    else if (auto n = dynamic_cast<const function_node *>(&node))
    {
      convert_function(*n, ctx);
    }
    else
      JLM_ASSERT(0);
  }
}

std::unique_ptr<::llvm::Module>
convert(ipgraph_module & im, ::llvm::LLVMContext & lctx)
{
  std::unique_ptr<::llvm::Module> lm(new ::llvm::Module("module", lctx));
  lm->setSourceFileName(im.source_filename().to_str());
  lm->setTargetTriple(im.target_triple());
  lm->setDataLayout(im.data_layout());

  context ctx(im, *lm);
  convert_ipgraph(ctx);

  return lm;
}

}
}
