/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/aggregation.hpp>
#include <jlm/ir/Annotation.hpp>
#include <jlm/ir/basic-block.hpp>
#include <jlm/ir/operators/operators.hpp>

#include <algorithm>
#include <typeindex>

namespace jlm {

std::string VariableSet::DebugString() const noexcept
{
  std::string debugString("{");

  bool isFirst = true;
  for (auto & variable : Variables()) {
    debugString += isFirst ? "" : ", ";
    debugString += variable.name();
    isFirst = false;
  }

  debugString += "}";

  return debugString;
}

DemandSet::~DemandSet() noexcept
= default;

EntryDemandSet::~EntryDemandSet() noexcept
= default;

std::string
EntryDemandSet::DebugString() const noexcept
{
  return strfmt("ReadSet:", ReadSet().DebugString(), " ",
                "AllWriteSet:", AllWriteSet().DebugString(), " ",
                "FullWriteSet:", FullWriteSet().DebugString(), " ",
                "TopSet:", TopSet_.DebugString(), " ",
                "BottomSet:", BottomSet_.DebugString(), " ");
}

bool
EntryDemandSet::operator==(const DemandSet & other)
{
  auto otherEntryDemandSet = dynamic_cast<const EntryDemandSet*>(&other);
  return otherEntryDemandSet
      && DemandSet::operator==(other)
      && TopSet_ == otherEntryDemandSet->TopSet_
      && BottomSet_ == otherEntryDemandSet->BottomSet_;
}

ExitDemandSet::~ExitDemandSet() noexcept
= default;

std::string
ExitDemandSet::DebugString() const noexcept
{
  return strfmt("ReadSet:", ReadSet().DebugString(), " ",
                "AllWriteSet:", AllWriteSet().DebugString(), " ",
                "FullWriteSet:", FullWriteSet().DebugString(), " ",
                "TopSet:", TopSet_.DebugString(), " ",
                "BottomSet:", BottomSet_.DebugString(), " ");
}

bool
ExitDemandSet::operator==(const DemandSet & other)
{
  auto otherExitDemandSet = dynamic_cast<const ExitDemandSet*>(&other);
  return otherExitDemandSet
         && DemandSet::operator==(other)
         && TopSet_ == otherExitDemandSet->TopSet_
         && BottomSet_ == otherExitDemandSet->BottomSet_;
}

BasicBlockDemandSet::~BasicBlockDemandSet() noexcept
= default;

std::string
BasicBlockDemandSet::DebugString() const noexcept
{
  return strfmt("ReadSet:", ReadSet().DebugString(), " ",
                "AllWriteSet:", AllWriteSet().DebugString(), " ",
                "FullWriteSet:", FullWriteSet().DebugString(), " ",
                "TopSet:", TopSet_.DebugString(), " ",
                "BottomSet:", BottomSet_.DebugString(), " ");
}

bool
BasicBlockDemandSet::operator==(const DemandSet & other)
{
  auto otherBasicBlockDemandSet = dynamic_cast<const BasicBlockDemandSet*>(&other);
  return otherBasicBlockDemandSet
         && DemandSet::operator==(other)
         && TopSet_ == otherBasicBlockDemandSet->TopSet_
         && BottomSet_ == otherBasicBlockDemandSet->BottomSet_;
}

LinearDemandSet::~LinearDemandSet() noexcept
= default;

std::string
LinearDemandSet::DebugString() const noexcept
{
  return strfmt("ReadSet:", ReadSet().DebugString(), " ",
                "AllWriteSet:", AllWriteSet().DebugString(), " ",
                "FullWriteSet:", FullWriteSet().DebugString(), " ",
                "TopSet:", TopSet_.DebugString(), " ",
                "BottomSet:", BottomSet_.DebugString(), " ");
}

bool
LinearDemandSet::operator==(const DemandSet & other)
{
  auto otherLinearDemandSet = dynamic_cast<const LinearDemandSet*>(&other);
  return otherLinearDemandSet
         && DemandSet::operator==(other)
         && TopSet_ == otherLinearDemandSet->TopSet_
         && BottomSet_ == otherLinearDemandSet->BottomSet_;
}

BranchDemandSet::~BranchDemandSet() noexcept
= default;

std::string
BranchDemandSet::DebugString() const noexcept
{
  return strfmt("ReadSet:", ReadSet().DebugString(), " ",
                "AllWriteSet:", AllWriteSet().DebugString(), " ",
                "FullWriteSet:", FullWriteSet().DebugString(), " ",
                "TopSet:", TopSet_.DebugString(), " ",
                "BottomSet:", BottomSet_.DebugString(), " ");
}

bool
BranchDemandSet::operator==(const DemandSet & other)
{
  auto otherBranchDemandSet = dynamic_cast<const BranchDemandSet*>(&other);
  return otherBranchDemandSet
         && DemandSet::operator==(other)
         && TopSet_ == otherBranchDemandSet->TopSet_
         && BottomSet_ == otherBranchDemandSet->BottomSet_;
}

LoopDemandSet::~LoopDemandSet() noexcept
= default;

std::string
LoopDemandSet::DebugString() const noexcept
{
  return strfmt("ReadSet:", ReadSet().DebugString(), " ",
                "AllWriteSet:", AllWriteSet().DebugString(), " ",
                "FullWriteSet:", FullWriteSet().DebugString(), " ",
                "TopSet:", TopSet_.DebugString(), " ",
                "BottomSet:", BottomSet_.DebugString(), " ");
}

bool
LoopDemandSet::operator==(const DemandSet & other)
{
  auto otherLoopDemandSet = dynamic_cast<const LoopDemandSet*>(&other);
  return otherLoopDemandSet
         && DemandSet::operator==(other)
         && TopSet_ == otherLoopDemandSet->TopSet_
         && BottomSet_ == otherLoopDemandSet->BottomSet_;
}

static void
AnnotateReadWrite(
  const aggnode&,
  DemandMap&);

static void
AnnotateReadWrite(
  const entryaggnode & entryAggregationNode,
  DemandMap & demandMap)
{
  VariableSet allWriteSet;
  VariableSet fullWriteSet;
	for (auto & argument : entryAggregationNode) {
		allWriteSet.Insert(argument);
		fullWriteSet.Insert(argument);
	}

  auto demandSet = EntryDemandSet::Create(
    VariableSet(),
    std::move(allWriteSet),
    std::move(fullWriteSet));
  demandMap.Insert(entryAggregationNode, std::move(demandSet));
}

static void
AnnotateReadWrite(
  const exitaggnode & exitAggregationNode,
  DemandMap & demandMap)
{
  VariableSet readSet;
	for (auto & result : exitAggregationNode)
		readSet.Insert(*result);

  auto demandSet = ExitDemandSet::Create(
    std::move(readSet),
    VariableSet(),
    VariableSet());
  demandMap.Insert(exitAggregationNode, std::move(demandSet));
}

static void
AnnotateReadWrite(
  const blockaggnode & basicBlockAggregationNode,
  DemandMap & demandMap)
{
	auto & threeAddressCodeList = basicBlockAggregationNode.tacs();

  VariableSet readSet;
  VariableSet allWriteSet;
  VariableSet fullWriteSet;
	for (auto it = threeAddressCodeList.rbegin(); it != threeAddressCodeList.rend(); it++) {
		auto & tac = *it;
		if (is<assignment_op>(tac->operation())) {
			/*
					We need special treatment for assignment operation, since the variable
					they assign the value to is modeled as an argument of the tac.
			*/
			JLM_ASSERT(tac->noperands() == 2 && tac->nresults() == 0);
			readSet.Remove(*tac->operand(0));
			allWriteSet.Insert(*tac->operand(0));
			fullWriteSet.Insert(*tac->operand(0));
			readSet.Insert(*tac->operand(1));
		} else {
			for (size_t n = 0; n < tac->nresults(); n++) {
				readSet.Remove(*tac->result(n));
				allWriteSet.Insert(*tac->result(n));
				fullWriteSet.Insert(*tac->result(n));
			}
			for (size_t n = 0; n < tac->noperands(); n++)
				readSet.Insert(*tac->operand(n));
		}
	}

  auto demandSet = BasicBlockDemandSet::Create(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet));
  demandMap.Insert(basicBlockAggregationNode, std::move(demandSet));
}

static void
AnnotateReadWrite(
  const linearaggnode & linearAggregationNode,
  DemandMap & demandMap)
{
  VariableSet readSet;
  VariableSet allWriteSet;
  VariableSet fullWriteSet;
	for (size_t n = linearAggregationNode.nchildren() - 1; n != static_cast<size_t>(-1); n--) {
		auto & childDemandSet = demandMap.Lookup<DemandSet>(*linearAggregationNode.child(n));

    readSet.Remove(childDemandSet.FullWriteSet());
    readSet.Insert(childDemandSet.ReadSet());
    allWriteSet.Insert(childDemandSet.AllWriteSet());
    fullWriteSet.Insert(childDemandSet.FullWriteSet());
	}

  auto demandSet = LinearDemandSet::Create(
    std::move(readSet),
    std::move(allWriteSet),
    std::move(fullWriteSet));
  demandMap.Insert(linearAggregationNode, std::move(demandSet));
}

static void
AnnotateReadWrite(
  const branchaggnode & branchAggregationNode,
  DemandMap & demandMap)
{
  auto & case0DemandSet = demandMap.Lookup<DemandSet>(*branchAggregationNode.child(0));
  auto & case0ReadSet = case0DemandSet.ReadSet();
  auto & case0AllWriteSet = case0DemandSet.AllWriteSet();
  auto & case0FullWriteSet = case0DemandSet.FullWriteSet();

  VariableSet branchReadSet(case0ReadSet);
  VariableSet branchAllWriteSet(case0AllWriteSet);
  VariableSet branchFullWriteSet(case0FullWriteSet);
	for (size_t n = 1; n < branchAggregationNode.nchildren(); n++) {
		auto & caseDemandSet = demandMap.Lookup<DemandSet>(*branchAggregationNode.child(n));

		branchAllWriteSet.Insert(caseDemandSet.AllWriteSet());
		branchFullWriteSet.Intersect(caseDemandSet.FullWriteSet());
		branchReadSet.Insert(caseDemandSet.ReadSet());
	}

  auto branchDemandSet = BranchDemandSet::Create(
    std::move(branchReadSet),
    std::move(branchAllWriteSet),
    std::move(branchFullWriteSet));
  demandMap.Insert(branchAggregationNode, std::move(branchDemandSet));
}

static void
AnnotateReadWrite(
  const loopaggnode & loopAggregationNode,
  DemandMap & demandMap)
{
  auto & loopBody = *loopAggregationNode.child(0);
  auto & bodyDemandSet = demandMap.Lookup<DemandSet>(loopBody);

	auto demandSet = LoopDemandSet::Create(
    bodyDemandSet.ReadSet(),
    bodyDemandSet.AllWriteSet(),
    bodyDemandSet.FullWriteSet());
  demandMap.Insert(loopAggregationNode, std::move(demandSet));
}

template<class T> static void
AnnotateReadWrite(
  const aggnode & aggregationNode,
  DemandMap & DemandMap)
{
	JLM_ASSERT(is<T>(&aggregationNode));
  AnnotateReadWrite(*static_cast<const T*>(&aggregationNode), DemandMap);
}

static void
AnnotateReadWrite(
  const aggnode & aggregationNode,
  DemandMap & demandMap)
{
	static std::unordered_map<
		std::type_index,
		void(*)(const aggnode&, DemandMap&)
	> map({
	  {typeid(entryaggnode),  AnnotateReadWrite<entryaggnode>}
	, {typeid(exitaggnode),   AnnotateReadWrite<exitaggnode>}
	, {typeid(blockaggnode),  AnnotateReadWrite<blockaggnode>}
	, {typeid(linearaggnode), AnnotateReadWrite<linearaggnode>}
	, {typeid(branchaggnode), AnnotateReadWrite<branchaggnode>}
	, {typeid(loopaggnode),   AnnotateReadWrite<loopaggnode>}
	});

	for (size_t n = 0; n < aggregationNode.nchildren(); n++)
    AnnotateReadWrite(*aggregationNode.child(n), demandMap);

	JLM_ASSERT(map.find(typeid(aggregationNode)) != map.end());
	return map[typeid(aggregationNode)](aggregationNode, demandMap);
}

static void
AnnotateDemandSet(
  const aggnode&,
  VariableSet&,
  DemandMap&);

static void
AnnotateDemandSet(
  const entryaggnode & entryAggregationNode,
  VariableSet & workingSet,
  DemandMap & demandMap)
{
	auto & demandSet = demandMap.Lookup<EntryDemandSet>(entryAggregationNode);
  demandSet.BottomSet_ = workingSet;

	workingSet.Remove(demandSet.FullWriteSet());

  demandSet.TopSet_ = workingSet;
}

static void
AnnotateDemandSet(
  const exitaggnode & exitAggregationNode,
  VariableSet & workingSet,
  DemandMap & demandMap)
{
	auto & demandSet = demandMap.Lookup<ExitDemandSet>(exitAggregationNode);
  demandSet.BottomSet_ = workingSet;

  workingSet.Insert(demandSet.ReadSet());

  demandSet.TopSet_ = workingSet;
}

static void
AnnotateDemandSet(
  const blockaggnode & basicBlockAggregationNode,
  VariableSet & workingSet,
  DemandMap & demandMap)
{
	auto & demandSet = demandMap.Lookup<BasicBlockDemandSet>(basicBlockAggregationNode);
  demandSet.BottomSet_ = workingSet;

	workingSet.Remove(demandSet.FullWriteSet());
	workingSet.Insert(demandSet.ReadSet());

  demandSet.TopSet_ = workingSet;
}

static void
AnnotateDemandSet(
  const linearaggnode & linearAggregationNode,
  VariableSet & workingSet,
  DemandMap & demandMap)
{
	auto & demandSet = demandMap.Lookup<LinearDemandSet>(linearAggregationNode);
  demandSet.BottomSet_ = workingSet;

	for (size_t n = linearAggregationNode.nchildren() - 1; n != static_cast<size_t>(-1); n--)
    AnnotateDemandSet(*linearAggregationNode.child(n), workingSet, demandMap);

	workingSet.Remove(demandSet.FullWriteSet());
	workingSet.Insert(demandSet.ReadSet());

  demandSet.TopSet_ = workingSet;
}

static void
AnnotateDemandSet(
  const branchaggnode & branchAggregationNode,
  VariableSet & workingSet,
  DemandMap & demandMap)
{
	auto & demandSet = demandMap.Lookup<BranchDemandSet>(branchAggregationNode);

	VariableSet passby = workingSet;
	passby.Subtract(demandSet.AllWriteSet());

	VariableSet bottom = workingSet;
	bottom.Intersect(demandSet.AllWriteSet());
  demandSet.BottomSet_ = bottom;

	for (size_t n = 0; n < branchAggregationNode.nchildren(); n++) {
		auto tmp = bottom;
    AnnotateDemandSet(*branchAggregationNode.child(n), tmp, demandMap);
	}


	workingSet.Remove(demandSet.FullWriteSet());
	workingSet.Insert(demandSet.ReadSet());
  demandSet.TopSet_ = workingSet;

	workingSet.Insert(passby);
}

static void
AnnotateDemandSet(
  const loopaggnode & loopAggregationNode,
  VariableSet & workingSet,
  DemandMap & demandMap)
{
	auto & demandSet = demandMap.Lookup<LoopDemandSet>(loopAggregationNode);

	workingSet.Insert(demandSet.ReadSet());
  demandSet.BottomSet_ = demandSet.TopSet_ = workingSet;
  AnnotateDemandSet(*loopAggregationNode.child(0), workingSet, demandMap);

	for (auto & v : demandSet.ReadSet().Variables())
		JLM_ASSERT(workingSet.Contains(v));

	for (auto & v : demandSet.FullWriteSet().Variables())
		if (!demandSet.ReadSet().Contains(v))
			JLM_ASSERT(!workingSet.Contains(v));
}

template<class T> static void
AnnotateDemandSet(
  const aggnode * aggregationNode,
  VariableSet & workingSet,
  DemandMap & demandMap)
{
	JLM_ASSERT(is<T>(aggregationNode));
  AnnotateDemandSet(*static_cast<const T *>(aggregationNode), workingSet, demandMap);
}

static void
AnnotateDemandSet(
  const aggnode & aggregationNode,
  VariableSet & workingSet,
  DemandMap & demandMap)
{
	static std::unordered_map<
		std::type_index,
		void(*)(const aggnode*, VariableSet&, DemandMap&)
	> map({
	  {typeid(entryaggnode),  AnnotateDemandSet<entryaggnode>}
	, {typeid(exitaggnode),   AnnotateDemandSet<exitaggnode>}
	, {typeid(blockaggnode),  AnnotateDemandSet<blockaggnode>}
	, {typeid(linearaggnode), AnnotateDemandSet<linearaggnode>}
	, {typeid(branchaggnode), AnnotateDemandSet<branchaggnode>}
	, {typeid(loopaggnode),   AnnotateDemandSet<loopaggnode>}
	});

	JLM_ASSERT(map.find(typeid(aggregationNode)) != map.end());
	return map[typeid(aggregationNode)](&aggregationNode, workingSet, demandMap);
}

std::unique_ptr<DemandMap>
Annotate(const aggnode & aggregationTreeRoot)
{
	auto demandMap = DemandMap::Create();
  AnnotateReadWrite(aggregationTreeRoot, *demandMap);

  VariableSet workingSet;
  AnnotateDemandSet(aggregationTreeRoot, workingSet, *demandMap);

	return demandMap;
}

}
