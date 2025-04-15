/*
 * Copyright 2025 Håvard Krogstie <krogstie.havard@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/opt/alias-analyses/AliasAnalysis.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/lambda.hpp>

#include <numeric>

namespace jlm::llvm::aa
{

AliasAnalysis::AliasAnalysis() = default;

AliasAnalysis::~AliasAnalysis() = default;

PointsToGraphAliasAnalysis::PointsToGraphAliasAnalysis(PointsToGraph & pointsToGraph)
    : PointsToGraph_(pointsToGraph)
{}

PointsToGraphAliasAnalysis::~PointsToGraphAliasAnalysis() = default;

AliasAnalysis::AliasQueryResponse
PointsToGraphAliasAnalysis::Query(
    const rvsdg::output & p1,
    [[maybe_unused]] size_t s1,
    const rvsdg::output & p2,
    [[maybe_unused]] size_t s2)
{
  // If the two pointers are the same value, they must alias
  // This is the only situation where this analysis gives MustAlias,
  // as no offset information is stored in the PointsToGraph.
  if (&p1 == &p2)
    return MustAlias;

  // Assume that all pointers actually exist in the PointsToGraph
  auto & p1RegisterNode = PointsToGraph_.GetRegisterNode(p1);
  auto & p2RegisterNode = PointsToGraph_.GetRegisterNode(p2);

  // TODO: Check for MustAlias using the following
  // - Both pointers have only one possible target
  // - the target is a global or an import
  // - both accesses have a size equal to the size of the target

  // If the registers are represented by the same node, they may alias
  if (&p1RegisterNode == &p2RegisterNode)
    return MayAlias;

  // Check if both pointers may target the external node, to avoid iterating over large sets
  const auto & externalNode = PointsToGraph_.GetExternalMemoryNode();
  if (p1RegisterNode.HasTarget(externalNode) && p2RegisterNode.HasTarget(externalNode))
    return MayAlias;

  // Check if p1 and p2 share any target memory nodes
  for (auto & target : p1RegisterNode.Targets())
  {
    // TODO: Ignore targets that are smaller than the access size
    if (p2RegisterNode.HasTarget(target))
      return MayAlias;
  }

  return NoAlias;
}

std::string
PointsToGraphAliasAnalysis::ToString() const
{
  return "PointsToGraphAA";
}

ChainedAliasAnalysis::ChainedAliasAnalysis(AliasAnalysis & first, AliasAnalysis & second)
    : First_(first),
      Second_(second)
{}

ChainedAliasAnalysis::~ChainedAliasAnalysis() = default;

AliasAnalysis::AliasQueryResponse
ChainedAliasAnalysis::Query(
    const rvsdg::output & p1,
    size_t s1,
    const rvsdg::output & p2,
    size_t s2)
{
  const auto firstResponse = First_.Query(p1, s1, p2, s2);

  // Anything other than MayAlias is precise, and can be returned right away
  if (firstResponse != MayAlias)
    return firstResponse;

  return Second_.Query(p1, s1, p2, s2);
}

std::string
ChainedAliasAnalysis::ToString() const
{
  return util::strfmt("ChainedAA(", First_.ToString(), ",", Second_.ToString(), ")");
}

BasicAliasAnalysis::BasicAliasAnalysis()
{}

BasicAliasAnalysis::~BasicAliasAnalysis() = default;

AliasAnalysis::AliasQueryResponse
BasicAliasAnalysis::Query(
    const rvsdg::output & p1,
    [[maybe_unused]] size_t s1,
    const rvsdg::output & p2,
    [[maybe_unused]] size_t s2)
{
  // If the two pointers are the same value, they must alias
  if (&p1 == &p2)
    return MustAlias;

  // Trace the origin of the pointers as long as possible
  util::HashSet<const rvsdg::output *> p1Trace, p2Trace;
  const auto p1SingleTarget = GetSingleTarget(p1, p1Trace);
  const auto p2SingleTarget = GetSingleTarget(p2, p2Trace);

  // If it was possible to track both p1 and p2 to a single target object
  // they are either identical, or point to completely different objects
  if (p1SingleTarget && p2SingleTarget)
  {
    if (*p1SingleTarget == *p2SingleTarget)
      return MustAlias;
    return NoAlias;
  }

  // Check if the trace from tracking the origin of p1 and p2 reach the same output at any point
  p1Trace.IntersectWithAndClear(p2Trace);
  if (!p1Trace.IsEmpty())
    return MustAlias;

  return MayAlias;
}

std::string
BasicAliasAnalysis::ToString() const
{
  return "BasicAA";
}

std::optional<const rvsdg::output *>
BasicAliasAnalysis::GetSingleTarget(
    const rvsdg::output & pointer,
    util::HashSet<const rvsdg::output *> & trace)
{
  // The head of the trace
  const rvsdg::output * p = &pointer;

  while (true)
  {
    // Insert p in the trace, quit if it was already there
    if (!trace.Insert(p))
      return std::nullopt;

    // Try either tracing the origin of p further, or stopping if a single target is found

    if (const auto import = dynamic_cast<const GraphImport *>(p))
    {
      // Each GraphImport represents a unique object
      return p;
    }

    if (const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*p))
    {
      if (is<IOBarrierOperation>(node))
      {
        p = node->input(0)->origin();
      }
      else if (is<alloca_op>(node))
      {
        return p;
      }
      else if (is<delta::operation>(node))
      {
        return p;
      }
      else if (is<malloc_op>(node))
      {
        return p;
      }
      else if (is<LlvmLambdaOperation>(node))
      {
        return p;
      }
    }
    else if (
        [[maybe_unused]] const auto structNode = rvsdg::TryGetOwnerNode<rvsdg::StructuralNode>(*p))
    {
      // If the output is a phi recursion variable, continue tracing inside the phi
      if (const auto phiResult = dynamic_cast<const llvm::phi::rvoutput *>(p))
      {
        p = phiResult->result()->origin();
      }
      else
      {
        // TODO: Outputs of structural nodes may be invariant, so do not give up already
        return std::nullopt;
      }
    }
    else if (const auto outerGamma = rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*p))
    {
      // Follow the gamma input
      p = outerGamma->GetEntryVar(p->index()).input->origin();
    }
    else if (const auto outerTheta = rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*p))
    {
      const auto loopVar = outerTheta->GetLoopVars()[p->index()];

      // If the loop variable is not invariant, stop tracing now
      if (!ThetaLoopVarIsInvariant(loopVar))
        return std::nullopt;

      p = loopVar.input->origin();
    }
    else if (const auto outerLambda = rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(*p))
    {
      const auto ctxVar = outerLambda->MapBinderContextVar(*p);

      // If it was not a contex variable, stop tracing
      if (!ctxVar)
        return std::nullopt;

      p = ctxVar->input->origin();
    }
    else if (rvsdg::TryGetRegionParentNode<llvm::phi::node>(*p) != nullptr)
    {
      if (const auto cvArg = dynamic_cast<const llvm::phi::cvargument*>(p))
      {
        // Follow the context variable to outside the phi
        p = cvArg->input()->origin();
      }
      else if (const auto rvArg = dynamic_cast<const llvm::phi::rvargument*>(p))
      {
        // Follow to the recursion variable's definition
        p = rvArg->result()->origin();
      }
      else
        JLM_UNREACHABLE("Unknown phi argument");
    }
  }
}

/**
 * @return the given \p size, rounded up to be a multiple of the given \p alignment
 */
size_t
RoundUpToMultipleOf(size_t size, size_t alignment)
{
  const auto miss = size % alignment;
  if (miss == 0)
    return size;
  return size + alignment - miss;
}

size_t
GetLlvmTypeSize(const rvsdg::ValueType & type)
{
  if (auto bits = dynamic_cast<const rvsdg::bittype *>(&type))
  {
    // Assume 8 bits per byte, rounding up to a whole byte
    return (bits->nbits() + 7) / 8;
  }
  if (is<PointerType>(type))
  {
    // Assume 64-bit pointers
    return 8;
  }
  if (auto arrayType = dynamic_cast<const ArrayType *>(&type))
  {
    return arrayType->nelements() * GetLlvmTypeSize(*arrayType->GetElementType());
  }
  if (auto floatType = dynamic_cast<const FloatingPointType *>(&type))
  {
    switch (floatType->size())
    {
    case fpsize::half:
      return 2;
    case fpsize::flt:
      return 4;
    case fpsize::dbl:
      return 8;
    case fpsize::fp128:
      return 16;
    case fpsize::x86fp80:
      return 16; // Will never actually be written to memory, but we round up
    default:
      JLM_UNREACHABLE("Unknown float size");
    }
  }
  if (auto structType = dynamic_cast<const StructType *>(&type))
  {
    // TODO: This function may over-estimate the size of a struct due to over-estimating field
    // alignment. For the purposes of alias analysis, that is sound, but it could be improved.

    size_t totalSize = 0;
    size_t alignment = 1;

    const auto & decl = structType->GetDeclaration();
    // A packed struct has alignment 1, and all fields are tightly packed
    const auto isPacked = structType->IsPacked();

    for (size_t i = 0; i < decl.NumElements(); i++)
    {
      auto & field = decl.GetElement(i);
      auto fieldSize = GetLlvmTypeSize(field);

      // Since we do not have proper alignment information, we assume field size
      auto fieldAlignment = isPacked ? 1 : fieldSize;

      // Add the size of the field, including any needed padding
      totalSize = RoundUpToMultipleOf(totalSize, fieldAlignment);
      totalSize += fieldSize;

      // The struct as a whole must be at least as aligned as each field
      alignment = std::lcm(alignment, fieldSize);
    }

    // Round size up to a multiple of alignment
    totalSize = RoundUpToMultipleOf(totalSize, alignment);

    // TODO: In C++, but not C, the sizeof() an empty struct is 1

    return totalSize;
  }
  if (auto vectorType = dynamic_cast<const VectorType *>(&type))
  {
    return vectorType->size() * GetLlvmTypeSize(*vectorType->Type());
  }

  std::cerr << "Unknown type: " << typeid(type).name() << std::endl;
  JLM_UNREACHABLE("Unknown type");
}

}
