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
      // list. Once all basic blocks have been converted, all SsaPhiOperations are revisited and
      // given operands.
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

/**
 * During conversion of LLVM instructions, phi instructions were created without operands.
 * Once all instructions have been converted, this function goes over all phi instructions
 * and assigns proper operands.
 */
static void
PatchPhiOperands(const std::vector<::llvm::PHINode *> & phis, context & ctx)
{
  for (const auto & phi : phis)
  {
    std::vector<ControlFlowGraphNode *> incomingNodes;
    std::vector<const Variable *> operands;
    for (size_t n = 0; n < phi->getNumOperands(); n++)
    {
      // In LLVM, phi instructions may have incoming basic blocks that are unreachable.
      // These are not visited during convert_basic_blocks, and thus do not have corresponding
      // jlm::llvm::basic_blocks. The SsaPhiOperation can safely ignore these, as they are dead.
      if (!ctx.has(phi->getIncomingBlock(n)))
        continue;

      // The LLVM phi instruction may have multiple operands with the same incoming cfg node.
      // When this happens in valid LLVM IR, all operands from the same basic block are identical.
      // We therefore skip any operands that reference already handled basic blocks.
      auto predecessor = ctx.get(phi->getIncomingBlock(n));
      if (std::find(incomingNodes.begin(), incomingNodes.end(), predecessor) != incomingNodes.end())
        continue;

      // Convert the operand value in the predecessor basic block, as that is where it is "used".
      tacsvector_t tacs;
      operands.push_back(ConvertValue(phi->getIncomingValue(n), tacs, ctx));
      predecessor->insert_before_branch(tacs);
      incomingNodes.push_back(predecessor);
    }

    JLM_ASSERT(operands.size() >= 1);

    auto phi_tac = util::AssertedCast<const ThreeAddressCodeVariable>(ctx.lookup_value(phi))->tac();
    phi_tac->replace(
        SsaPhiOperation(std::move(incomingNodes), phi_tac->result(0)->Type()),
        operands);
  }
}

static basic_block_map
convert_basic_blocks(::llvm::Function & f, ControlFlowGraph & cfg)
{
  basic_block_map bbmap;
  ::llvm::ReversePostOrderTraversal<::llvm::Function *> rpotraverser(&f);
  for (auto & bb : rpotraverser)
    bbmap.insert(bb, BasicBlock::create(cfg));

  return bbmap;
}

Attribute::kind
ConvertAttributeKind(const ::llvm::Attribute::AttrKind & kind)
{
  typedef ::llvm::Attribute::AttrKind ak;

  static std::unordered_map<::llvm::Attribute::AttrKind, Attribute::kind> map(
      { { ak::None, Attribute::kind::None },
        { ak::FirstEnumAttr, Attribute::kind::FirstEnumAttr },
        { ak::AllocAlign, Attribute::kind::AllocAlign },
        { ak::AllocatedPointer, Attribute::kind::AllocatedPointer },
        { ak::AlwaysInline, Attribute::kind::AlwaysInline },
        { ak::Builtin, Attribute::kind::Builtin },
        { ak::Cold, Attribute::kind::Cold },
        { ak::Convergent, Attribute::kind::Convergent },
        { ak::CoroDestroyOnlyWhenComplete, Attribute::kind::CoroDestroyOnlyWhenComplete },
        { ak::DeadOnUnwind, Attribute::kind::DeadOnUnwind },
        { ak::DisableSanitizerInstrumentation, Attribute::kind::DisableSanitizerInstrumentation },
        { ak::FnRetThunkExtern, Attribute::kind::FnRetThunkExtern },
        { ak::Hot, Attribute::kind::Hot },
        { ak::ImmArg, Attribute::kind::ImmArg },
        { ak::InReg, Attribute::kind::InReg },
        { ak::InlineHint, Attribute::kind::InlineHint },
        { ak::JumpTable, Attribute::kind::JumpTable },
        { ak::Memory, Attribute::kind::Memory },
        { ak::MinSize, Attribute::kind::MinSize },
        { ak::MustProgress, Attribute::kind::MustProgress },
        { ak::Naked, Attribute::kind::Naked },
        { ak::Nest, Attribute::kind::Nest },
        { ak::NoAlias, Attribute::kind::NoAlias },
        { ak::NoBuiltin, Attribute::kind::NoBuiltin },
        { ak::NoCallback, Attribute::kind::NoCallback },
        { ak::NoCapture, Attribute::kind::NoCapture },
        { ak::NoCfCheck, Attribute::kind::NoCfCheck },
        { ak::NoDuplicate, Attribute::kind::NoDuplicate },
        { ak::NoFree, Attribute::kind::NoFree },
        { ak::NoImplicitFloat, Attribute::kind::NoImplicitFloat },
        { ak::NoInline, Attribute::kind::NoInline },
        { ak::NoMerge, Attribute::kind::NoMerge },
        { ak::NoProfile, Attribute::kind::NoProfile },
        { ak::NoRecurse, Attribute::kind::NoRecurse },
        { ak::NoRedZone, Attribute::kind::NoRedZone },
        { ak::NoReturn, Attribute::kind::NoReturn },
        { ak::NoSanitizeBounds, Attribute::kind::NoSanitizeBounds },
        { ak::NoSanitizeCoverage, Attribute::kind::NoSanitizeCoverage },
        { ak::NoSync, Attribute::kind::NoSync },
        { ak::NoUndef, Attribute::kind::NoUndef },
        { ak::NoUnwind, Attribute::kind::NoUnwind },
        { ak::NonLazyBind, Attribute::kind::NonLazyBind },
        { ak::NonNull, Attribute::kind::NonNull },
        { ak::NullPointerIsValid, Attribute::kind::NullPointerIsValid },
        { ak::OptForFuzzing, Attribute::kind::OptForFuzzing },
        { ak::OptimizeForDebugging, Attribute::kind::OptimizeForDebugging },
        { ak::OptimizeForSize, Attribute::kind::OptimizeForSize },
        { ak::OptimizeNone, Attribute::kind::OptimizeNone },
        { ak::PresplitCoroutine, Attribute::kind::PresplitCoroutine },
        { ak::ReadNone, Attribute::kind::ReadNone },
        { ak::ReadOnly, Attribute::kind::ReadOnly },
        { ak::Returned, Attribute::kind::Returned },
        { ak::ReturnsTwice, Attribute::kind::ReturnsTwice },
        { ak::SExt, Attribute::kind::SExt },
        { ak::SafeStack, Attribute::kind::SafeStack },
        { ak::SanitizeAddress, Attribute::kind::SanitizeAddress },
        { ak::SanitizeHWAddress, Attribute::kind::SanitizeHWAddress },
        { ak::SanitizeMemTag, Attribute::kind::SanitizeMemTag },
        { ak::SanitizeMemory, Attribute::kind::SanitizeMemory },
        { ak::SanitizeThread, Attribute::kind::SanitizeThread },
        { ak::ShadowCallStack, Attribute::kind::ShadowCallStack },
        { ak::SkipProfile, Attribute::kind::SkipProfile },
        { ak::Speculatable, Attribute::kind::Speculatable },
        { ak::SpeculativeLoadHardening, Attribute::kind::SpeculativeLoadHardening },
        { ak::StackProtect, Attribute::kind::StackProtect },
        { ak::StackProtectReq, Attribute::kind::StackProtectReq },
        { ak::StackProtectStrong, Attribute::kind::StackProtectStrong },
        { ak::StrictFP, Attribute::kind::StrictFP },
        { ak::SwiftAsync, Attribute::kind::SwiftAsync },
        { ak::SwiftError, Attribute::kind::SwiftError },
        { ak::SwiftSelf, Attribute::kind::SwiftSelf },
        { ak::WillReturn, Attribute::kind::WillReturn },
        { ak::Writable, Attribute::kind::Writable },
        { ak::WriteOnly, Attribute::kind::WriteOnly },
        { ak::ZExt, Attribute::kind::ZExt },
        { ak::LastEnumAttr, Attribute::kind::LastEnumAttr },
        { ak::FirstTypeAttr, Attribute::kind::FirstTypeAttr },
        { ak::ByRef, Attribute::kind::ByRef },
        { ak::ByVal, Attribute::kind::ByVal },
        { ak::ElementType, Attribute::kind::ElementType },
        { ak::InAlloca, Attribute::kind::InAlloca },
        { ak::Preallocated, Attribute::kind::Preallocated },
        { ak::StructRet, Attribute::kind::StructRet },
        { ak::LastTypeAttr, Attribute::kind::LastTypeAttr },
        { ak::FirstIntAttr, Attribute::kind::FirstIntAttr },
        { ak::Alignment, Attribute::kind::Alignment },
        { ak::AllocKind, Attribute::kind::AllocKind },
        { ak::AllocSize, Attribute::kind::AllocSize },
        { ak::Dereferenceable, Attribute::kind::Dereferenceable },
        { ak::DereferenceableOrNull, Attribute::kind::DereferenceableOrNull },
        { ak::NoFPClass, Attribute::kind::NoFPClass },
        { ak::StackAlignment, Attribute::kind::StackAlignment },
        { ak::UWTable, Attribute::kind::UWTable },
        { ak::VScaleRange, Attribute::kind::VScaleRange },
        { ak::LastIntAttr, Attribute::kind::LastIntAttr },
        { ak::EndAttrKinds, Attribute::kind::EndAttrKinds } });

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
    return { Attribute::kind::ByVal, std::move(type) };
  }

  if (attribute.getKindAsEnum() == ::llvm::Attribute::AttrKind::StructRet)
  {
    auto type = ctx.GetTypeConverter().ConvertLlvmType(*attribute.getValueAsType());
    return { Attribute::kind::StructRet, std::move(type) };
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
EnsureSingleInEdgeToExitNode(ControlFlowGraph & cfg)
{
  auto exitNode = cfg.exit();

  if (exitNode->NumInEdges() == 0)
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

        auto basicBlock = BasicBlock::create(cfg);

        rvsdg::ctlconstant_op op(rvsdg::ctlvalue_repr(1, 2));
        auto operand = basicBlock->append_last(ThreeAddressCode::create(op, {}))->result(0);
        basicBlock->append_last(BranchOperation::create(2, operand));

        basicBlock->add_outedge(exitNode);
        basicBlock->add_outedge(repetitionEdge->sink());

        repetitionEdge->divert(basicBlock);
        break;
      }
    }
  }

  if (exitNode->NumInEdges() == 1)
    return;

  /*
    We have multiple incoming edges to the exit node. Insert an empty basic block, divert all
    incoming edges to this block, and add an outgoing edge from this block to the exit node.
  */
  auto basicBlock = BasicBlock::create(cfg);
  exitNode->divert_inedges(basicBlock);
  basicBlock->add_outedge(exitNode);
}

static std::unique_ptr<ControlFlowGraph>
create_cfg(::llvm::Function & f, context & ctx)
{
  auto node = static_cast<const fctvariable *>(ctx.lookup_value(&f))->function();

  auto add_arguments = [](const ::llvm::Function & f, ControlFlowGraph & cfg, context & ctx)
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

  auto cfg = ControlFlowGraph::create(ctx.module());

  add_arguments(f, *cfg, ctx);
  auto bbmap = convert_basic_blocks(f, *cfg);

  /* create entry block */
  auto entry_block = BasicBlock::create(*cfg);
  cfg->exit()->divert_inedges(entry_block);
  entry_block->add_outedge(bbmap[&f.getEntryBlock()]);

  /* add results */
  const ThreeAddressCodeVariable * result = nullptr;
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
  PatchPhiOperands(phis, ctx);

  EnsureSingleInEdgeToExitNode(*cfg);

  // Merge basic blocks A -> B when possible
  straighten(*cfg);
  // Remove unreachable nodes
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

    return DataNode::Create(
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
  auto v = static_cast<const GlobalValue *>(ctx.lookup_value(&gv));

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

std::unique_ptr<InterProceduralGraphModule>
ConvertLlvmModule(::llvm::Module & m)
{
  util::FilePath fp(m.getSourceFileName());
  auto im = InterProceduralGraphModule::create(fp, m.getTargetTriple(), m.getDataLayoutStr());

  context ctx(*im);
  declare_globals(m, ctx);
  convert_globals(m, ctx);

  im->SetStructTypeDeclarations(ctx.GetTypeConverter().ReleaseStructTypeDeclarations());

  return im;
}

}
