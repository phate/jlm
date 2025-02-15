/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/frontend/LlvmConversionContext.hpp>
#include <jlm/llvm/frontend/LlvmInstructionConversion.hpp>
#include <jlm/llvm/frontend/LlvmModuleConversion.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/TypeConverter.hpp>

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

namespace jlm::llvm
{

static std::vector<::llvm::PHINode *>
convert_instructions(::llvm::Function & function, context & ctx)
{
  std::vector<::llvm::PHINode *> phis;
  ::llvm::ReversePostOrderTraversal<::llvm::Function *> rpotraverser(&function);
  for (auto & bb : rpotraverser)
  {
    for (auto & instruction : *bb)
    {
      tacsvector_t tacs;
      if (auto result = ConvertInstruction(&instruction, tacs, ctx))
        ctx.insert_value(&instruction, result);

      // When an LLVM PhiNode is converted to a jlm SsaPhiOperation, some of its operands may not be
      // ready. The created SsaPhiOperation therefore has no operands, but is instead added to a
      // list. Once all basic blocks have been converted, all phi_ops are revisited and given
      // operands.
      if (!tacs.empty() && is<SsaPhiOperation>(tacs.back()->operation()))
      {
        auto phi = ::llvm::dyn_cast<::llvm::PHINode>(&instruction);
        phis.push_back(phi);
      }

      ctx.get(bb)->append_last(tacs);
    }
  }

  return phis;
}

static void
patch_phi_operands(const std::vector<::llvm::PHINode *> & phis, context & ctx)
{
  for (const auto & phi : phis)
  {
    std::vector<cfg_node *> nodes;
    std::vector<const variable *> operands;
    for (size_t n = 0; n < phi->getNumOperands(); n++)
    {
      // In LLVM, phi instructions may have incoming basic blocks that are unreachable.
      // These are not visited during convert_basic_blocks, and thus do not have corresponding
      // jlm::llvm::basic_blocks. The SsaPhiOperation can safely ignore these, as they are dead.
      if (!ctx.has(phi->getIncomingBlock(n)))
        continue;

      auto bb = ctx.get(phi->getIncomingBlock(n));
      tacsvector_t tacs;
      operands.push_back(ConvertValue(phi->getIncomingValue(n), tacs, ctx));
      bb->insert_before_branch(tacs);
      nodes.push_back(bb);
    }

    // Phi instructions with a single reachable predecessor should have already been elided
    JLM_ASSERT(operands.size() >= 2);

    auto phi_tac = util::AssertedCast<const tacvariable>(ctx.lookup_value(phi))->tac();
    phi_tac->replace(SsaPhiOperation(nodes, phi_tac->result(0)->Type()), operands);
  }
}

static basic_block_map
convert_basic_blocks(::llvm::Function & f, llvm::cfg & cfg)
{
  basic_block_map bbmap;
  ::llvm::ReversePostOrderTraversal<::llvm::Function *> rpotraverser(&f);
  for (auto & bb : rpotraverser)
    bbmap.insert(bb, basic_block::create(cfg));

  return bbmap;
}

attribute::kind
ConvertAttributeKind(const ::llvm::Attribute::AttrKind & kind)
{
  typedef ::llvm::Attribute::AttrKind ak;

  static std::unordered_map<::llvm::Attribute::AttrKind, attribute::kind> map(
      { { ak::None, attribute::kind::None },
        { ak::FirstEnumAttr, attribute::kind::FirstEnumAttr },
        { ak::AllocAlign, attribute::kind::AllocAlign },
        { ak::AllocatedPointer, attribute::kind::AllocatedPointer },
        { ak::AlwaysInline, attribute::kind::AlwaysInline },
        { ak::Builtin, attribute::kind::Builtin },
        { ak::Cold, attribute::kind::Cold },
        { ak::Convergent, attribute::kind::Convergent },
        { ak::CoroDestroyOnlyWhenComplete, attribute::kind::CoroDestroyOnlyWhenComplete },
        { ak::DeadOnUnwind, attribute::kind::DeadOnUnwind },
        { ak::DisableSanitizerInstrumentation, attribute::kind::DisableSanitizerInstrumentation },
        { ak::FnRetThunkExtern, attribute::kind::FnRetThunkExtern },
        { ak::Hot, attribute::kind::Hot },
        { ak::ImmArg, attribute::kind::ImmArg },
        { ak::InReg, attribute::kind::InReg },
        { ak::InlineHint, attribute::kind::InlineHint },
        { ak::JumpTable, attribute::kind::JumpTable },
        { ak::Memory, attribute::kind::Memory },
        { ak::MinSize, attribute::kind::MinSize },
        { ak::MustProgress, attribute::kind::MustProgress },
        { ak::Naked, attribute::kind::Naked },
        { ak::Nest, attribute::kind::Nest },
        { ak::NoAlias, attribute::kind::NoAlias },
        { ak::NoBuiltin, attribute::kind::NoBuiltin },
        { ak::NoCallback, attribute::kind::NoCallback },
        { ak::NoCapture, attribute::kind::NoCapture },
        { ak::NoCfCheck, attribute::kind::NoCfCheck },
        { ak::NoDuplicate, attribute::kind::NoDuplicate },
        { ak::NoFree, attribute::kind::NoFree },
        { ak::NoImplicitFloat, attribute::kind::NoImplicitFloat },
        { ak::NoInline, attribute::kind::NoInline },
        { ak::NoMerge, attribute::kind::NoMerge },
        { ak::NoProfile, attribute::kind::NoProfile },
        { ak::NoRecurse, attribute::kind::NoRecurse },
        { ak::NoRedZone, attribute::kind::NoRedZone },
        { ak::NoReturn, attribute::kind::NoReturn },
        { ak::NoSanitizeBounds, attribute::kind::NoSanitizeBounds },
        { ak::NoSanitizeCoverage, attribute::kind::NoSanitizeCoverage },
        { ak::NoSync, attribute::kind::NoSync },
        { ak::NoUndef, attribute::kind::NoUndef },
        { ak::NoUnwind, attribute::kind::NoUnwind },
        { ak::NonLazyBind, attribute::kind::NonLazyBind },
        { ak::NonNull, attribute::kind::NonNull },
        { ak::NullPointerIsValid, attribute::kind::NullPointerIsValid },
        { ak::OptForFuzzing, attribute::kind::OptForFuzzing },
        { ak::OptimizeForDebugging, attribute::kind::OptimizeForDebugging },
        { ak::OptimizeForSize, attribute::kind::OptimizeForSize },
        { ak::OptimizeNone, attribute::kind::OptimizeNone },
        { ak::PresplitCoroutine, attribute::kind::PresplitCoroutine },
        { ak::ReadNone, attribute::kind::ReadNone },
        { ak::ReadOnly, attribute::kind::ReadOnly },
        { ak::Returned, attribute::kind::Returned },
        { ak::ReturnsTwice, attribute::kind::ReturnsTwice },
        { ak::SExt, attribute::kind::SExt },
        { ak::SafeStack, attribute::kind::SafeStack },
        { ak::SanitizeAddress, attribute::kind::SanitizeAddress },
        { ak::SanitizeHWAddress, attribute::kind::SanitizeHWAddress },
        { ak::SanitizeMemTag, attribute::kind::SanitizeMemTag },
        { ak::SanitizeMemory, attribute::kind::SanitizeMemory },
        { ak::SanitizeThread, attribute::kind::SanitizeThread },
        { ak::ShadowCallStack, attribute::kind::ShadowCallStack },
        { ak::SkipProfile, attribute::kind::SkipProfile },
        { ak::Speculatable, attribute::kind::Speculatable },
        { ak::SpeculativeLoadHardening, attribute::kind::SpeculativeLoadHardening },
        { ak::StackProtect, attribute::kind::StackProtect },
        { ak::StackProtectReq, attribute::kind::StackProtectReq },
        { ak::StackProtectStrong, attribute::kind::StackProtectStrong },
        { ak::StrictFP, attribute::kind::StrictFP },
        { ak::SwiftAsync, attribute::kind::SwiftAsync },
        { ak::SwiftError, attribute::kind::SwiftError },
        { ak::SwiftSelf, attribute::kind::SwiftSelf },
        { ak::WillReturn, attribute::kind::WillReturn },
        { ak::Writable, attribute::kind::Writable },
        { ak::WriteOnly, attribute::kind::WriteOnly },
        { ak::ZExt, attribute::kind::ZExt },
        { ak::LastEnumAttr, attribute::kind::LastEnumAttr },
        { ak::FirstTypeAttr, attribute::kind::FirstTypeAttr },
        { ak::ByRef, attribute::kind::ByRef },
        { ak::ByVal, attribute::kind::ByVal },
        { ak::ElementType, attribute::kind::ElementType },
        { ak::InAlloca, attribute::kind::InAlloca },
        { ak::Preallocated, attribute::kind::Preallocated },
        { ak::StructRet, attribute::kind::StructRet },
        { ak::LastTypeAttr, attribute::kind::LastTypeAttr },
        { ak::FirstIntAttr, attribute::kind::FirstIntAttr },
        { ak::Alignment, attribute::kind::Alignment },
        { ak::AllocKind, attribute::kind::AllocKind },
        { ak::AllocSize, attribute::kind::AllocSize },
        { ak::Dereferenceable, attribute::kind::Dereferenceable },
        { ak::DereferenceableOrNull, attribute::kind::DereferenceableOrNull },
        { ak::NoFPClass, attribute::kind::NoFPClass },
        { ak::StackAlignment, attribute::kind::StackAlignment },
        { ak::UWTable, attribute::kind::UWTable },
        { ak::VScaleRange, attribute::kind::VScaleRange },
        { ak::LastIntAttr, attribute::kind::LastIntAttr },
        { ak::EndAttrKinds, attribute::kind::EndAttrKinds } });

  JLM_ASSERT(map.find(kind) != map.end());
  return map[kind];
}

static enum_attribute
ConvertEnumAttribute(const ::llvm::Attribute & attribute)
{
  JLM_ASSERT(attribute.isEnumAttribute());
  auto kind = ConvertAttributeKind(attribute.getKindAsEnum());
  return enum_attribute(kind);
}

static int_attribute
ConvertIntAttribute(const ::llvm::Attribute & attribute)
{
  JLM_ASSERT(attribute.isIntAttribute());
  auto kind = ConvertAttributeKind(attribute.getKindAsEnum());
  return { kind, attribute.getValueAsInt() };
}

static type_attribute
ConvertTypeAttribute(const ::llvm::Attribute & attribute, context & ctx)
{
  JLM_ASSERT(attribute.isTypeAttribute());

  if (attribute.getKindAsEnum() == ::llvm::Attribute::AttrKind::ByVal)
  {
    auto type = ctx.GetTypeConverter().ConvertLlvmType(*attribute.getValueAsType());
    return { attribute::kind::ByVal, std::move(type) };
  }

  if (attribute.getKindAsEnum() == ::llvm::Attribute::AttrKind::StructRet)
  {
    auto type = ctx.GetTypeConverter().ConvertLlvmType(*attribute.getValueAsType());
    return { attribute::kind::StructRet, std::move(type) };
  }

  JLM_UNREACHABLE("Unhandled attribute");
}

static string_attribute
ConvertStringAttribute(const ::llvm::Attribute & attribute)
{
  JLM_ASSERT(attribute.isStringAttribute());
  return { attribute.getKindAsString().str(), attribute.getValueAsString().str() };
}

static attributeset
convert_attributes(const ::llvm::AttributeSet & as, context & ctx)
{
  attributeset attributeSet;
  for (auto & attribute : as)
  {
    if (attribute.isEnumAttribute())
    {
      attributeSet.InsertEnumAttribute(ConvertEnumAttribute(attribute));
    }
    else if (attribute.isIntAttribute())
    {
      attributeSet.InsertIntAttribute(ConvertIntAttribute(attribute));
    }
    else if (attribute.isTypeAttribute())
    {
      attributeSet.InsertTypeAttribute(ConvertTypeAttribute(attribute, ctx));
    }
    else if (attribute.isStringAttribute())
    {
      attributeSet.InsertStringAttribute(ConvertStringAttribute(attribute));
    }
    else
    {
      JLM_UNREACHABLE("Unhandled attribute");
    }
  }

  return attributeSet;
}

static std::unique_ptr<llvm::argument>
convert_argument(const ::llvm::Argument & argument, context & ctx)
{
  auto function = argument.getParent();
  auto name = argument.getName().str();
  auto type = ctx.GetTypeConverter().ConvertLlvmType(*argument.getType());
  auto attributes =
      convert_attributes(function->getAttributes().getParamAttrs(argument.getArgNo()), ctx);

  return llvm::argument::create(name, type, attributes);
}

static void
EnsureSingleInEdgeToExitNode(llvm::cfg & cfg)
{
  auto exitNode = cfg.exit();

  if (exitNode->ninedges() == 0)
  {
    /*
      LLVM can produce CFGs that have no incoming edge to the exit node. This can happen if
      endless loops are present in the code. For example, this code

      \code{.cpp}
        int foo()
        {
          while (1) {
            printf("foo\n");
          }

          return 0;
        }
      \endcode

      results in a JLM CFG with no incoming edge to the exit node.

      We solve this problem by finding the first SCC with no exit edge, i.e., an endless loop, and
      restructure it to an SCC with an exit edge to the CFG's exit node.
    */
    auto stronglyConnectedComponents = find_sccs(cfg);
    for (auto stronglyConnectedComponent : stronglyConnectedComponents)
    {
      auto structure = sccstructure::create(stronglyConnectedComponent);

      if (structure->nxedges() == 0)
      {
        auto repetitionEdge = *structure->redges().begin();

        auto basicBlock = basic_block::create(cfg);

        rvsdg::ctlconstant_op op(rvsdg::ctlvalue_repr(1, 2));
        auto operand = basicBlock->append_last(tac::create(op, {}))->result(0);
        basicBlock->append_last(branch_op::create(2, operand));

        basicBlock->add_outedge(exitNode);
        basicBlock->add_outedge(repetitionEdge->sink());

        repetitionEdge->divert(basicBlock);
        break;
      }
    }
  }

  if (exitNode->ninedges() == 1)
    return;

  /*
    We have multiple incoming edges to the exit node. Insert an empty basic block, divert all
    incoming edges to this block, and add an outgoing edge from this block to the exit node.
  */
  auto basicBlock = basic_block::create(cfg);
  exitNode->divert_inedges(basicBlock);
  basicBlock->add_outedge(exitNode);
}

static std::unique_ptr<llvm::cfg>
create_cfg(::llvm::Function & f, context & ctx)
{
  auto node = static_cast<const fctvariable *>(ctx.lookup_value(&f))->function();

  auto add_arguments = [](const ::llvm::Function & f, llvm::cfg & cfg, context & ctx)
  {
    auto node = static_cast<const fctvariable *>(ctx.lookup_value(&f))->function();

    size_t n = 0;
    for (const auto & arg : f.args())
    {
      auto argument = cfg.entry()->append_argument(convert_argument(arg, ctx));
      ctx.insert_value(&arg, argument);
      n++;
    }

    if (f.isVarArg())
    {
      JLM_ASSERT(n < node->fcttype().NumArguments());
      auto & type = node->fcttype().Arguments()[n++];
      cfg.entry()->append_argument(argument::create("_varg_", type));
    }
    JLM_ASSERT(n < node->fcttype().NumArguments());

    auto & iotype = node->fcttype().Arguments()[n++];
    auto iostate = cfg.entry()->append_argument(argument::create("_io_", iotype));

    auto & memtype = node->fcttype().Arguments()[n++];
    auto memstate = cfg.entry()->append_argument(argument::create("_s_", memtype));

    JLM_ASSERT(n == node->fcttype().NumArguments());
    ctx.set_iostate(iostate);
    ctx.set_memory_state(memstate);
  };

  auto cfg = cfg::create(ctx.module());

  add_arguments(f, *cfg, ctx);
  auto bbmap = convert_basic_blocks(f, *cfg);

  /* create entry block */
  auto entry_block = basic_block::create(*cfg);
  cfg->exit()->divert_inedges(entry_block);
  entry_block->add_outedge(bbmap[&f.getEntryBlock()]);

  /* add results */
  const tacvariable * result = nullptr;
  if (!f.getReturnType()->isVoidTy())
  {
    auto type = ctx.GetTypeConverter().ConvertLlvmType(*f.getReturnType());
    entry_block->append_last(UndefValueOperation::Create(type, "_r_"));
    result = entry_block->last()->result(0);

    JLM_ASSERT(node->fcttype().NumResults() == 3);
    JLM_ASSERT(result->type() == node->fcttype().ResultType(0));
    cfg->exit()->append_result(result);
  }
  cfg->exit()->append_result(ctx.iostate());
  cfg->exit()->append_result(ctx.memory_state());

  /* convert instructions */
  ctx.set_basic_block_map(bbmap);
  ctx.set_result(result);
  auto phis = convert_instructions(f, ctx);
  patch_phi_operands(phis, ctx);

  EnsureSingleInEdgeToExitNode(*cfg);

  straighten(*cfg);
  prune(*cfg);
  return cfg;
}

static void
convert_function(::llvm::Function & function, context & ctx)
{
  if (function.isDeclaration())
    return;

  auto fv = static_cast<const fctvariable *>(ctx.lookup_value(&function));

  ctx.set_node(fv->function());
  fv->function()->add_cfg(create_cfg(function, ctx));
  ctx.set_node(nullptr);
}

static const llvm::linkage &
convert_linkage(const ::llvm::GlobalValue::LinkageTypes & linkage)
{
  static std::unordered_map<::llvm::GlobalValue::LinkageTypes, llvm::linkage> map(
      { { ::llvm::GlobalValue::ExternalLinkage, llvm::linkage::external_linkage },
        { ::llvm::GlobalValue::AvailableExternallyLinkage,
          llvm::linkage::available_externally_linkage },
        { ::llvm::GlobalValue::LinkOnceAnyLinkage, llvm::linkage::link_once_any_linkage },
        { ::llvm::GlobalValue::LinkOnceODRLinkage, llvm::linkage::link_once_odr_linkage },
        { ::llvm::GlobalValue::WeakAnyLinkage, llvm::linkage::weak_any_linkage },
        { ::llvm::GlobalValue::WeakODRLinkage, llvm::linkage::weak_odr_linkage },
        { ::llvm::GlobalValue::AppendingLinkage, llvm::linkage::appending_linkage },
        { ::llvm::GlobalValue::InternalLinkage, llvm::linkage::internal_linkage },
        { ::llvm::GlobalValue::PrivateLinkage, llvm::linkage::private_linkage },
        { ::llvm::GlobalValue::ExternalWeakLinkage, llvm::linkage::external_weak_linkage },
        { ::llvm::GlobalValue::CommonLinkage, llvm::linkage::common_linkage } });

  JLM_ASSERT(map.find(linkage) != map.end());
  return map[linkage];
}

static void
declare_globals(::llvm::Module & lm, context & ctx)
{
  auto create_data_node = [](const ::llvm::GlobalVariable & gv, context & ctx)
  {
    auto name = gv.getName().str();
    auto constant = gv.isConstant();
    auto type = ctx.GetTypeConverter().ConvertLlvmType(*gv.getValueType());
    auto linkage = convert_linkage(gv.getLinkage());
    auto section = gv.getSection().str();

    return data_node::Create(
        ctx.module().ipgraph(),
        name,
        type,
        linkage,
        std::move(section),
        constant);
  };

  auto create_function_node = [](const ::llvm::Function & f, context & ctx)
  {
    auto name = f.getName().str();
    auto linkage = convert_linkage(f.getLinkage());
    auto type = ctx.GetTypeConverter().ConvertFunctionType(*f.getFunctionType());
    auto attributes = convert_attributes(f.getAttributes().getFnAttrs(), ctx);

    return function_node::create(ctx.module().ipgraph(), name, type, linkage, attributes);
  };

  for (auto & gv : lm.globals())
  {
    auto node = create_data_node(gv, ctx);
    ctx.insert_value(&gv, ctx.module().create_global_value(node));
  }

  for (auto & f : lm.getFunctionList())
  {
    auto node = create_function_node(f, ctx);
    ctx.insert_value(&f, ctx.module().create_variable(node));
  }
}

static std::unique_ptr<data_node_init>
create_initialization(::llvm::GlobalVariable & gv, context & ctx)
{
  if (!gv.hasInitializer())
    return nullptr;

  auto init = gv.getInitializer();
  auto tacs = ConvertConstant(init, ctx);
  if (tacs.empty())
    return std::make_unique<data_node_init>(ctx.lookup_value(init));

  return std::make_unique<data_node_init>(std::move(tacs));
}

static void
convert_global_value(::llvm::GlobalVariable & gv, context & ctx)
{
  auto v = static_cast<const gblvalue *>(ctx.lookup_value(&gv));

  ctx.set_node(v->node());
  v->node()->set_initialization(create_initialization(gv, ctx));
  ctx.set_node(nullptr);
}

static void
convert_globals(::llvm::Module & lm, context & ctx)
{
  for (auto & gv : lm.globals())
    convert_global_value(gv, ctx);

  for (auto & f : lm.getFunctionList())
    convert_function(f, ctx);
}

std::unique_ptr<ipgraph_module>
ConvertLlvmModule(::llvm::Module & m)
{
  util::filepath fp(m.getSourceFileName());
  auto im = ipgraph_module::create(fp, m.getTargetTriple(), m.getDataLayoutStr());

  context ctx(*im);
  declare_globals(m, ctx);
  convert_globals(m, ctx);

  im->SetStructTypeDeclarations(ctx.GetTypeConverter().ReleaseStructTypeDeclarations());

  return im;
}

}
