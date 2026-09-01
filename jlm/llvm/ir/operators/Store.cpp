/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/Trace.hpp>
#include <jlm/llvm/ir/types.hpp>
#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/util/HashSet.hpp>

namespace jlm::llvm
{

StoreNonVolatileOperation::~StoreNonVolatileOperation() noexcept = default;

bool
StoreNonVolatileOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const StoreNonVolatileOperation *>(&other);
  return operation && operation->narguments() == narguments()
      && operation->GetStoredType() == GetStoredType()
      && operation->GetAlignment() == GetAlignment();
}

std::string
StoreNonVolatileOperation::debug_string() const
{
  return util::strfmt("Store[", GetStoredType().debug_string(), "]");
}

std::unique_ptr<rvsdg::Operation>
StoreNonVolatileOperation::copy() const
{
  return std::make_unique<StoreNonVolatileOperation>(*this);
}

static bool
is_store_mux_reducible(const std::vector<jlm::rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() > 2);

  const auto memStateMergeNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operands[2]);
  if (!is<MemoryStateMergeOperation>(memStateMergeNode))
    return false;

  for (size_t n = 2; n < operands.size(); n++)
  {
    if (rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operands[n]) != memStateMergeNode)
      return false;
  }

  return true;
}

static bool
is_store_alloca_reducible(const std::vector<jlm::rvsdg::Output *> & operands)
{
  if (operands.size() == 3)
    return false;

  const auto allocaNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operands[0]);
  if (!is<AllocaOperation>(allocaNode))
    return false;

  std::unordered_set states(std::next(std::next(operands.begin())), operands.end());
  if (states.find(allocaNode->output(1)) == states.end())
    return false;

  if (allocaNode->output(1)->nusers() != 1)
    return false;

  return true;
}

static bool
is_multiple_origin_reducible(const std::vector<jlm::rvsdg::Output *> & operands)
{
  const util::HashSet<rvsdg::Output *> states(std::next(operands.begin(), 2), operands.end());
  return states.Size() != operands.size() - 2;
}

static std::vector<jlm::rvsdg::Output *>
perform_store_mux_reduction(
    const StoreNonVolatileOperation & op,
    const std::vector<jlm::rvsdg::Output *> & operands)
{
  const auto memStateMergeNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operands[2]);
  auto memStateMergeOperands = jlm::rvsdg::operands(memStateMergeNode);

  auto states = StoreNonVolatileOperation::Create(
      operands[0],
      operands[1],
      memStateMergeOperands,
      op.GetAlignment());
  return { MemoryStateMergeOperation::Create(states) };
}

static std::vector<jlm::rvsdg::Output *>
perform_store_alloca_reduction(
    const StoreNonVolatileOperation & op,
    const std::vector<jlm::rvsdg::Output *> & operands)
{
  auto value = operands[1];
  auto address = operands[0];
  auto alloca_state = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*address)->output(1);
  std::unordered_set<jlm::rvsdg::Output *> states(
      std::next(std::next(operands.begin())),
      operands.end());

  auto outputs =
      StoreNonVolatileOperation::Create(address, value, { alloca_state }, op.GetAlignment());
  states.erase(alloca_state);
  states.insert(outputs[0]);
  return { states.begin(), states.end() };
}

static std::vector<jlm::rvsdg::Output *>
perform_multiple_origin_reduction(
    const StoreNonVolatileOperation & operation,
    const std::vector<jlm::rvsdg::Output *> & operands)
{
  // FIXME: Unify with the duplicate state removal reduction of the LoadNonVolatile operation

  JLM_ASSERT(operands.size() > 2);
  const auto address = operands[0];
  const auto value = operands[1];

  std::vector<rvsdg::Output *> newInputStates;
  std::unordered_map<rvsdg::Output *, size_t> stateIndexMap;
  for (size_t n = 2; n < operands.size(); n++)
  {
    auto state = operands[n];
    if (stateIndexMap.find(state) == stateIndexMap.end())
    {
      const size_t resultIndex = newInputStates.size();
      newInputStates.push_back(state);
      stateIndexMap[state] = resultIndex;
    }
  }

  const auto storeResults =
      StoreNonVolatileOperation::Create(address, value, newInputStates, operation.GetAlignment());

  std::vector<rvsdg::Output *> results(operation.nresults(), nullptr);
  for (size_t n = 2; n < operands.size(); n++)
  {
    auto state = operands[n];
    JLM_ASSERT(stateIndexMap.find(state) != stateIndexMap.end());
    results[n - 2] = storeResults[stateIndexMap[state]];
  }

  return results;
}

std::optional<std::vector<rvsdg::Output *>>
StoreNonVolatileOperation::NormalizeStoreMux(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  if (is_store_mux_reducible(operands))
    return perform_store_mux_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::Output *>>
StoreNonVolatileOperation::normalizeStoreStore(
    const StoreNonVolatileOperation & store2Op,
    const std::vector<rvsdg::Output *> & operands)
{
  if (store2Op.NumMemoryStates() == 0)
  {
    // We have a store node without memory state edges. This can happen if the compiler can
    // statically prove that the store node's address is a null pointer.
    return std::nullopt;
  }

  JLM_ASSERT(operands.size() > 2);
  auto & store2Address = *operands[0];
  auto & store2Value = *operands[1];
  const auto & store2FirstMemoryState = *operands[2];

  // Try tracing a memory state edge to a previous store
  const auto [store1Node, store1Op] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<StoreNonVolatileOperation>(store2FirstMemoryState);
  if (!store1Op)
    return std::nullopt;

  // Store1 and store2 must have the same address
  auto & store1Address = *AddressInput(*store1Node).origin();
  if (&llvm::traceOutput(store1Address) != &llvm::traceOutput(store2Address))
    return std::nullopt;

  // Check that all memory state inputs originate from store1 AND have no other users
  std::vector<rvsdg::Output *> newMemoryStates;
  for (size_t n = 2; n < operands.size(); n++)
  {
    auto & memoryState = *operands[n];
    JLM_ASSERT(is<MemoryStateType>(memoryState.Type()));

    if (rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(memoryState) == store1Node
        && memoryState.nusers() == 1)
    {
      auto & memoryStateInput = MapMemoryStateOutputToInput(memoryState);
      newMemoryStates.push_back(memoryStateInput.origin());
    }
    else
    {
      return std::nullopt;
    }
  }

  // Check that store2 fully overwrites store1
  const auto & store1Type = store1Op->GetStoredType();
  const auto & store2Type = store2Op.GetStoredType();
  if (GetTypeStoreSize(store2Type) < GetTypeStoreSize(store1Type))
    return std::nullopt;

  return Create(&store2Address, &store2Value, newMemoryStates, store2Op.GetAlignment());
}

std::optional<std::vector<rvsdg::Output *>>
StoreNonVolatileOperation::NormalizeStoreAlloca(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  if (is_store_alloca_reducible(operands))
    return perform_store_alloca_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::Output *>>
StoreNonVolatileOperation::NormalizeDuplicateStates(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  if (is_multiple_origin_reducible(operands))
    return perform_multiple_origin_reduction(operation, operands);

  return std::nullopt;
}

// FIXME: We have exactly the same function for the
// LoadNonVolatileOperation::normalizeIOBarrierAddress
static std::optional<size_t>
getAllocationSizeInBytes(const rvsdg::Output & output)
{
  auto [allocaNode, allocaOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<AllocaOperation>(output);
  if (allocaOperation)
  {
    return GetTypeAllocSize(*allocaOperation->allocatedType());
  }

  if (const auto deltaNode = rvsdg::TryGetOwnerNode<rvsdg::DeltaNode>(output))
  {
    const auto deltaOperation =
        util::assertedCast<const LlvmDeltaOperation>(&deltaNode->GetOperation());
    return GetTypeAllocSize(*deltaOperation->Type());
  }

  if (const auto llvmImport = dynamic_cast<const LlvmGraphImport *>(&output))
  {
    return GetTypeAllocSize(*llvmImport->ValueType());
  }

  return std::nullopt;
}

std::optional<std::vector<rvsdg::Output *>>
StoreNonVolatileOperation::normalizeIOBarrierAddress(
    const StoreNonVolatileOperation & storeOperation,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() >= 2);
  const auto address = operands[0];
  const auto value = operands[1];

  auto [ioBarrierNode, ioBarrierOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(*address);
  if (!ioBarrierOperation)
    return std::nullopt;

  auto & barredAddress = *IOBarrierOperation::BarredInput(*ioBarrierNode).origin();
  const auto & pointerOrigin = TracePointerOriginPrecise(barredAddress);
  const auto allocationSizeInBytes = getAllocationSizeInBytes(*pointerOrigin.BasePointer);
  if (!allocationSizeInBytes.has_value())
    return std::nullopt;

  size_t offsetInBytes = 0;
  if (const auto offsetInBytesOpt = pointerOrigin.getOffsetInBytes(); offsetInBytesOpt.has_value())
  {
    offsetInBytes = offsetInBytesOpt.value();
  }

  // This transformation is only valid if the affected bytes by the store operation are within the
  // size of the allocation site.
  if (offsetInBytes + GetTypeStoreSize(storeOperation.GetStoredType())
      > allocationSizeInBytes.value())
    return std::nullopt;

  auto & storeNode = CreateNode(
      barredAddress,
      *value,
      { std::next(operands.begin(), 2), operands.end() },
      storeOperation.GetAlignment());

  return { outputs(&storeNode) };
}

std::optional<std::vector<rvsdg::Output *>>
StoreNonVolatileOperation::normalizeStoreAllocaSingleUser(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() >= 2);
  const auto & address = *operands[0];

  // We cannot(!) use the traced address in this normalization as it might result in the wrong
  // number of users. The address can be routed through a structural node where it has multiple
  // users, but the traced address would still just have a single user.
  if (!rvsdg::IsOwnerNodeOperation<AllocaOperation>(address))
    return std::nullopt;

  if (address.nusers() != 1)
    return std::nullopt;

  std::vector newMemoryStateResults(operands.begin() + 2, operands.end());
  JLM_ASSERT(newMemoryStateResults.size() == operation.NumMemoryStates());

  return newMemoryStateResults;
}

StoreVolatileOperation::~StoreVolatileOperation() noexcept = default;

bool
StoreVolatileOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const StoreVolatileOperation *>(&other);
  return operation && operation->NumMemoryStates() == NumMemoryStates()
      && operation->GetStoredType() == GetStoredType()
      && operation->GetAlignment() == GetAlignment();
}

std::string
StoreVolatileOperation::debug_string() const
{
  return "StoreVolatile";
}

std::unique_ptr<rvsdg::Operation>
StoreVolatileOperation::copy() const
{
  return std::make_unique<StoreVolatileOperation>(*this);
}

}
