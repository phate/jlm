/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/aggregation.hpp>
#include <jlm/llvm/ir/Annotation.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <algorithm>
#include <typeindex>

namespace jlm::llvm
{

std::string
VariableSet::DebugString() const noexcept
{
  std::string debugString("{");

  bool isFirst = true;
  for (auto & variable : Variables())
  {
    debugString += isFirst ? "" : ", ";
    debugString += variable.name();
    isFirst = false;
  }

  debugString += "}";

  return debugString;
}

AnnotationSet::~AnnotationSet() noexcept = default;

EntryAnnotationSet::~EntryAnnotationSet() noexcept = default;

std::string
EntryAnnotationSet::DebugString() const noexcept
{
  return util::strfmt(
      "ReadSet:",
      ReadSet().DebugString(),
      " ",
      "AllWriteSet:",
      AllWriteSet().DebugString(),
      " ",
      "FullWriteSet:",
      FullWriteSet().DebugString(),
      " ",
      "TopSet:",
      TopSet_.DebugString(),
      " ");
}

bool
EntryAnnotationSet::operator==(const AnnotationSet & other)
{
  auto otherEntryDemandSet = dynamic_cast<const EntryAnnotationSet *>(&other);
  return otherEntryDemandSet && AnnotationSet::operator==(other)
      && TopSet_ == otherEntryDemandSet->TopSet_;
}

ExitAnnotationSet::~ExitAnnotationSet() noexcept = default;

std::string
ExitAnnotationSet::DebugString() const noexcept
{
  return util::strfmt(
      "ReadSet:",
      ReadSet().DebugString(),
      " ",
      "AllWriteSet:",
      AllWriteSet().DebugString(),
      " ",
      "FullWriteSet:",
      FullWriteSet().DebugString(),
      " ");
}

bool
ExitAnnotationSet::operator==(const AnnotationSet & other)
{
  auto otherExitDemandSet = dynamic_cast<const ExitAnnotationSet *>(&other);
  return otherExitDemandSet && AnnotationSet::operator==(other);
}

BasicBlockAnnotationSet::~BasicBlockAnnotationSet() noexcept = default;

std::string
BasicBlockAnnotationSet::DebugString() const noexcept
{
  return util::strfmt(
      "ReadSet:",
      ReadSet().DebugString(),
      " ",
      "AllWriteSet:",
      AllWriteSet().DebugString(),
      " ",
      "FullWriteSet:",
      FullWriteSet().DebugString(),
      " ");
}

bool
BasicBlockAnnotationSet::operator==(const AnnotationSet & other)
{
  auto otherBasicBlockDemandSet = dynamic_cast<const BasicBlockAnnotationSet *>(&other);
  return otherBasicBlockDemandSet && AnnotationSet::operator==(other);
}

LinearAnnotationSet::~LinearAnnotationSet() noexcept = default;

std::string
LinearAnnotationSet::DebugString() const noexcept
{
  return util::strfmt(
      "ReadSet:",
      ReadSet().DebugString(),
      " ",
      "AllWriteSet:",
      AllWriteSet().DebugString(),
      " ",
      "FullWriteSet:",
      FullWriteSet().DebugString(),
      " ");
}

bool
LinearAnnotationSet::operator==(const AnnotationSet & other)
{
  auto otherLinearDemandSet = dynamic_cast<const LinearAnnotationSet *>(&other);
  return otherLinearDemandSet && AnnotationSet::operator==(other);
}

BranchAnnotationSet::~BranchAnnotationSet() noexcept = default;

std::string
BranchAnnotationSet::DebugString() const noexcept
{
  return util::strfmt(
      "ReadSet:",
      ReadSet().DebugString(),
      " ",
      "AllWriteSet:",
      AllWriteSet().DebugString(),
      " ",
      "FullWriteSet:",
      FullWriteSet().DebugString(),
      " ",
      "TopSet:",
      InputVariables().DebugString(),
      " ",
      "BottomSet:",
      OutputVariables().DebugString(),
      " ");
}

bool
BranchAnnotationSet::operator==(const AnnotationSet & other)
{
  auto otherBranchDemandSet = dynamic_cast<const BranchAnnotationSet *>(&other);
  return otherBranchDemandSet && AnnotationSet::operator==(other)
      && InputVariables_ == otherBranchDemandSet->InputVariables_
      && OutputVariables_ == otherBranchDemandSet->OutputVariables_;
}

LoopAnnotationSet::~LoopAnnotationSet() noexcept = default;

std::string
LoopAnnotationSet::DebugString() const noexcept
{
  return util::strfmt(
      "ReadSet:",
      ReadSet().DebugString(),
      " ",
      "AllWriteSet:",
      AllWriteSet().DebugString(),
      " ",
      "FullWriteSet:",
      FullWriteSet().DebugString(),
      " ",
      "LoopVariables:",
      LoopVariables_.DebugString(),
      " ");
}

bool
LoopAnnotationSet::operator==(const AnnotationSet & other)
{
  auto otherLoopDemandSet = dynamic_cast<const LoopAnnotationSet *>(&other);
  return otherLoopDemandSet && AnnotationSet::operator==(other)
      && LoopVariables_ == otherLoopDemandSet->LoopVariables_;
}

static void
AnnotateReadWrite(const AggregationNode &, AnnotationMap &);

static void
AnnotateReadWrite(const EntryAggregationNode & entryAggregationNode, AnnotationMap & demandMap)
{
  VariableSet allWriteSet;
  VariableSet fullWriteSet;
  for (auto & argument : entryAggregationNode)
  {
    allWriteSet.Insert(argument);
    fullWriteSet.Insert(argument);
  }

  auto demandSet =
      EntryAnnotationSet::Create(VariableSet(), std::move(allWriteSet), std::move(fullWriteSet));
  demandMap.Insert(entryAggregationNode, std::move(demandSet));
}

static void
AnnotateReadWrite(const ExitAggregationNode & exitAggregationNode, AnnotationMap & demandMap)
{
  VariableSet readSet;
  for (auto & result : exitAggregationNode)
    readSet.Insert(*result);

  auto demandSet = ExitAnnotationSet::Create(std::move(readSet), VariableSet(), VariableSet());
  demandMap.Insert(exitAggregationNode, std::move(demandSet));
}

static void
AnnotateReadWrite(
    const BasicBlockAggregationNode & basicBlockAggregationNode,
    AnnotationMap & demandMap)
{
  auto & threeAddressCodeList = basicBlockAggregationNode.tacs();

  VariableSet readSet;
  VariableSet allWriteSet;
  VariableSet fullWriteSet;
  for (auto it = threeAddressCodeList.rbegin(); it != threeAddressCodeList.rend(); it++)
  {
    auto & tac = *it;
    if (is<AssignmentOperation>(tac->operation()))
    {
      /*
          We need special treatment for assignment operation, since the variable
          they assign the value to is modeled as an argument of the three address code.
      */
      JLM_ASSERT(tac->noperands() == 2 && tac->nresults() == 0);
      readSet.Remove(*tac->operand(0));
      allWriteSet.Insert(*tac->operand(0));
      fullWriteSet.Insert(*tac->operand(0));
      readSet.Insert(*tac->operand(1));
    }
    else
    {
      for (size_t n = 0; n < tac->nresults(); n++)
      {
        readSet.Remove(*tac->result(n));
        allWriteSet.Insert(*tac->result(n));
        fullWriteSet.Insert(*tac->result(n));
      }
      for (size_t n = 0; n < tac->noperands(); n++)
        readSet.Insert(*tac->operand(n));
    }
  }

  auto demandSet = BasicBlockAnnotationSet::Create(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  demandMap.Insert(basicBlockAggregationNode, std::move(demandSet));
}

static void
AnnotateReadWrite(const LinearAggregationNode & linearAggregationNode, AnnotationMap & demandMap)
{
  VariableSet readSet;
  VariableSet allWriteSet;
  VariableSet fullWriteSet;
  for (size_t n = linearAggregationNode.nchildren() - 1; n != static_cast<size_t>(-1); n--)
  {
    auto & childDemandSet = demandMap.Lookup<AnnotationSet>(*linearAggregationNode.child(n));

    readSet.Remove(childDemandSet.FullWriteSet());
    readSet.Insert(childDemandSet.ReadSet());
    allWriteSet.Insert(childDemandSet.AllWriteSet());
    fullWriteSet.Insert(childDemandSet.FullWriteSet());
  }

  auto demandSet = LinearAnnotationSet::Create(
      std::move(readSet),
      std::move(allWriteSet),
      std::move(fullWriteSet));
  demandMap.Insert(linearAggregationNode, std::move(demandSet));
}

static void
AnnotateReadWrite(const BranchAggregationNode & branchAggregationNode, AnnotationMap & demandMap)
{
  auto & case0DemandSet = demandMap.Lookup<AnnotationSet>(*branchAggregationNode.child(0));
  auto & case0ReadSet = case0DemandSet.ReadSet();
  auto & case0AllWriteSet = case0DemandSet.AllWriteSet();
  auto & case0FullWriteSet = case0DemandSet.FullWriteSet();

  VariableSet branchReadSet(case0ReadSet);
  VariableSet branchAllWriteSet(case0AllWriteSet);
  VariableSet branchFullWriteSet(case0FullWriteSet);
  for (size_t n = 1; n < branchAggregationNode.nchildren(); n++)
  {
    auto & caseDemandSet = demandMap.Lookup<AnnotationSet>(*branchAggregationNode.child(n));

    branchAllWriteSet.Insert(caseDemandSet.AllWriteSet());
    branchFullWriteSet.Intersect(caseDemandSet.FullWriteSet());
    branchReadSet.Insert(caseDemandSet.ReadSet());
  }

  auto branchDemandSet = BranchAnnotationSet::Create(
      std::move(branchReadSet),
      std::move(branchAllWriteSet),
      std::move(branchFullWriteSet));
  demandMap.Insert(branchAggregationNode, std::move(branchDemandSet));
}

static void
AnnotateReadWrite(const loopaggnode & loopAggregationNode, AnnotationMap & demandMap)
{
  auto & loopBody = *loopAggregationNode.child(0);
  auto & bodyDemandSet = demandMap.Lookup<AnnotationSet>(loopBody);

  auto demandSet = LoopAnnotationSet::Create(
      bodyDemandSet.ReadSet(),
      bodyDemandSet.AllWriteSet(),
      bodyDemandSet.FullWriteSet());
  demandMap.Insert(loopAggregationNode, std::move(demandSet));
}

static void
AnnotateReadWrite(const AggregationNode & aggregationNode, AnnotationMap & demandMap)
{
  for (size_t n = 0; n < aggregationNode.nchildren(); n++)
    AnnotateReadWrite(*aggregationNode.child(n), demandMap);

  if (auto entryNode = dynamic_cast<const EntryAggregationNode *>(&aggregationNode))
  {
    AnnotateReadWrite(*entryNode, demandMap);
  }
  else if (auto exitNode = dynamic_cast<const ExitAggregationNode *>(&aggregationNode))
  {
    AnnotateReadWrite(*exitNode, demandMap);
  }
  else if (auto blockNode = dynamic_cast<const BasicBlockAggregationNode *>(&aggregationNode))
  {
    AnnotateReadWrite(*blockNode, demandMap);
  }
  else if (const auto linearNode = dynamic_cast<const LinearAggregationNode *>(&aggregationNode))
  {
    AnnotateReadWrite(*linearNode, demandMap);
  }
  else if (auto branchNode = dynamic_cast<const BranchAggregationNode *>(&aggregationNode))
  {
    AnnotateReadWrite(*branchNode, demandMap);
  }
  else if (auto loopNode = dynamic_cast<const loopaggnode *>(&aggregationNode))
  {
    AnnotateReadWrite(*loopNode, demandMap);
  }
  else
  {
    JLM_UNREACHABLE("Unhandled aggregation node type");
  }
}

static void
AnnotateDemandSet(const AggregationNode &, VariableSet &, AnnotationMap &);

static void
AnnotateDemandSet(
    const EntryAggregationNode & entryAggregationNode,
    VariableSet & workingSet,
    AnnotationMap & demandMap)
{
  auto & demandSet = demandMap.Lookup<EntryAnnotationSet>(entryAggregationNode);

  workingSet.Remove(demandSet.FullWriteSet());

  demandSet.TopSet_ = workingSet;
}

static void
AnnotateDemandSet(
    const ExitAggregationNode & exitAggregationNode,
    VariableSet & workingSet,
    AnnotationMap & demandMap)
{
  auto & demandSet = demandMap.Lookup<ExitAnnotationSet>(exitAggregationNode);
  workingSet.Insert(demandSet.ReadSet());
}

static void
AnnotateDemandSet(
    const BasicBlockAggregationNode & basicBlockAggregationNode,
    VariableSet & workingSet,
    AnnotationMap & demandMap)
{
  auto & demandSet = demandMap.Lookup<BasicBlockAnnotationSet>(basicBlockAggregationNode);
  workingSet.Remove(demandSet.FullWriteSet());
  workingSet.Insert(demandSet.ReadSet());
}

static void
AnnotateDemandSet(
    const LinearAggregationNode & linearAggregationNode,
    VariableSet & workingSet,
    AnnotationMap & demandMap)
{
  auto & demandSet = demandMap.Lookup<LinearAnnotationSet>(linearAggregationNode);

  for (size_t n = linearAggregationNode.nchildren() - 1; n != static_cast<size_t>(-1); n--)
    AnnotateDemandSet(*linearAggregationNode.child(n), workingSet, demandMap);

  workingSet.Remove(demandSet.FullWriteSet());
  workingSet.Insert(demandSet.ReadSet());
}

static void
AnnotateDemandSet(
    const BranchAggregationNode & branchAggregationNode,
    VariableSet & workingSet,
    AnnotationMap & demandMap)
{
  auto & demandSet = demandMap.Lookup<BranchAnnotationSet>(branchAggregationNode);

  VariableSet branchWorkingSet = workingSet;
  branchWorkingSet.Intersect(demandSet.AllWriteSet());
  demandSet.SetOutputVariables(branchWorkingSet);

  for (size_t n = 0; n < branchAggregationNode.nchildren(); n++)
  {
    auto caseWorkingSet = branchWorkingSet;
    AnnotateDemandSet(*branchAggregationNode.child(n), caseWorkingSet, demandMap);
  }

  branchWorkingSet.Remove(demandSet.FullWriteSet());
  branchWorkingSet.Insert(demandSet.ReadSet());
  demandSet.SetInputVariables(branchWorkingSet);

  workingSet.Insert(branchWorkingSet);
}

static void
AnnotateDemandSet(
    const loopaggnode & loopAggregationNode,
    VariableSet & workingSet,
    AnnotationMap & demandMap)
{
  auto & demandSet = demandMap.Lookup<LoopAnnotationSet>(loopAggregationNode);

  workingSet.Insert(demandSet.ReadSet());
  demandSet.SetLoopVariables(workingSet);
  AnnotateDemandSet(*loopAggregationNode.child(0), workingSet, demandMap);
}

template<class T>
static void
AnnotateDemandSet(
    const AggregationNode * aggregationNode,
    VariableSet & workingSet,
    AnnotationMap & demandMap)
{
  JLM_ASSERT(is<T>(aggregationNode));
  AnnotateDemandSet(*static_cast<const T *>(aggregationNode), workingSet, demandMap);
}

static void
AnnotateDemandSet(
    const AggregationNode & aggregationNode,
    VariableSet & workingSet,
    AnnotationMap & demandMap)
{
  static std::unordered_map<
      std::type_index,
      void (*)(const AggregationNode *, VariableSet &, AnnotationMap &)>
      map({ { typeid(EntryAggregationNode), AnnotateDemandSet<EntryAggregationNode> },
            { typeid(ExitAggregationNode), AnnotateDemandSet<ExitAggregationNode> },
            { typeid(BasicBlockAggregationNode), AnnotateDemandSet<BasicBlockAggregationNode> },
            { typeid(LinearAggregationNode), AnnotateDemandSet<LinearAggregationNode> },
            { typeid(BranchAggregationNode), AnnotateDemandSet<BranchAggregationNode> },
            { typeid(loopaggnode), AnnotateDemandSet<loopaggnode> } });

  JLM_ASSERT(map.find(typeid(aggregationNode)) != map.end());
  return map[typeid(aggregationNode)](&aggregationNode, workingSet, demandMap);
}

std::unique_ptr<AnnotationMap>
Annotate(const AggregationNode & aggregationTreeRoot)
{
  auto demandMap = AnnotationMap::Create();
  AnnotateReadWrite(aggregationTreeRoot, *demandMap);

  VariableSet workingSet;
  AnnotateDemandSet(aggregationTreeRoot, workingSet, *demandMap);

  return demandMap;
}

}
