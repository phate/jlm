/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/util/HashSet.hpp>

namespace jlm::llvm
{

LoadNonVolatileOperation::~LoadNonVolatileOperation() noexcept = default;

bool
LoadNonVolatileOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const LoadNonVolatileOperation *>(&other);
  return operation && operation->narguments() == narguments()
      && operation->GetLoadedType() == GetLoadedType()
      && operation->GetAlignment() == GetAlignment();
}

std::string
LoadNonVolatileOperation::debug_string() const
{
  return "Load";
}

std::unique_ptr<rvsdg::Operation>
LoadNonVolatileOperation::copy() const
{
  return std::make_unique<LoadNonVolatileOperation>(*this);
}

/*
  If the producer of a load's address is an alloca, then we can remove
  all state edges originating from other allocas.

  a1 s1 = AllocaOperation ...
  a2 s2 = AllocaOperation ...
  s3 = mux_op s1
  v sl1 sl2 sl3 = load_op a1 s1 s2 s3
  =>
  ...
  v sl1 sl3 = load_op a1 s1 s3
*/
static bool
is_load_alloca_reducible(const std::vector<rvsdg::Output *> & operands)
{
  auto address = operands[0];

  auto [allocaNode, allocaOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<AllocaOperation>(*address);
  if (!allocaOperation)
    return false;

  for (size_t n = 1; n < operands.size(); n++)
  {
    const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operands[n]);
    if (is<AllocaOperation>(node) && node != allocaNode)
      return true;
  }

  return false;
}

static bool
is_reducible_state(const rvsdg::Output * state, const rvsdg::Node * loadalloca)
{
  auto [storeNode, storeOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<StoreNonVolatileOperation>(*state);
  if (storeOperation)
  {
    auto address = StoreNonVolatileOperation::AddressInput(*storeNode).origin();
    auto [allocaNode, allocaOperation] =
        jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<AllocaOperation>(*address);
    if (allocaOperation && allocaNode != loadalloca)
      return true;
  }

  return false;
}

/*
  a1 sa1 = AllocaOperation ...
  a2 sa2 = AllocaOperation ...
  ss1 = store_op a1 ... sa1
  ss2 = store_op a2 ... sa2
  ... = load_op a1 ss1 ss2
  =>
  ...
  ... = load_op a1 ss1
*/
static bool
is_load_store_state_reducible(
    const LoadNonVolatileOperation & op,
    const std::vector<rvsdg::Output *> & operands)
{
  auto address = operands[0];

  if (operands.size() == 2)
    return false;

  auto [allocaNode, allocaOperation] =
      jlm::rvsdg::TryGetSimpleNodeAndOptionalOp<AllocaOperation>(*address);
  if (!allocaOperation)
  {
    return false;
  }

  size_t redstates = 0;
  for (size_t n = 1; n < operands.size(); n++)
  {
    auto state = operands[n];
    if (is_reducible_state(state, allocaNode))
      redstates++;
  }

  return redstates == op.NumMemoryStates() || redstates == 0 ? false : true;
}

/*
  v so1 so2 so3 = load_op a si1 si1 si1
  =>
  v so1 = load_op a si1
*/
static bool
is_multiple_origin_reducible(const std::vector<rvsdg::Output *> & operands)
{
  const util::HashSet<rvsdg::Output *> states(std::next(operands.begin()), operands.end());
  return states.Size() != operands.size() - 1;
}

// s2 = store_op a v1 s1
// v2 s3 = load_op a s2
// ... = any_op v2
// =>
// s2 = store_op a v1 s1
// ... = any_op v1
static bool
is_load_store_reducible(
    const LoadNonVolatileOperation & loadOperation,
    const std::vector<rvsdg::Output *> & operands)
{
  // We do not need to check further if no state edge is provided to the load
  if (operands.size() < 2)
  {
    return false;
  }

  // Check that the first state edge originates from a store
  auto firstState = operands[1];
  auto [storeNode, storeOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<StoreNonVolatileOperation>(*firstState);
  if (!storeOperation)
  {
    return false;
  }

  // Check that all state edges to the load originate from the same store
  if (storeOperation->NumMemoryStates() != loadOperation.NumMemoryStates())
  {
    return false;
  }
  for (size_t n = 1; n < operands.size(); n++)
  {
    auto state = operands[n];
    const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*state);
    if (node != storeNode)
    {
      return false;
    }
  }

  // Check that the address to the load and store originate from the same value
  auto loadAddress = operands[0];
  auto storeAddress = StoreNonVolatileOperation::AddressInput(*storeNode).origin();
  if (loadAddress != storeAddress)
  {
    return false;
  }

  // Check that the loaded and stored value type are the same
  //
  // FIXME: This is too restrictive and can be improved upon by inserting truncation or narrowing
  // operations instead. For example, a store of a 32 bit integer followed by a load of a 8 bit
  // integer can be converted to a trunc operation.
  auto loadedValueType = loadOperation.GetLoadedType();
  auto & storedValueType = *StoreNonVolatileOperation::StoredValueInput(*storeNode).Type();
  if (*loadedValueType != storedValueType)
  {
    return false;
  }

  JLM_ASSERT(loadOperation.GetAlignment() == storeOperation->GetAlignment());
  return true;
}

static std::vector<rvsdg::Output *>
perform_load_store_reduction(
    const LoadNonVolatileOperation &,
    const std::vector<rvsdg::Output *> & operands)
{
  const auto storeNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operands[1]);

  std::vector results(1, storeNode->input(1)->origin());
  results.insert(results.end(), std::next(operands.begin()), operands.end());

  return results;
}

static std::vector<rvsdg::Output *>
perform_load_alloca_reduction(
    const LoadNonVolatileOperation & op,
    const std::vector<rvsdg::Output *> & operands)
{
  const auto allocaNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operands[0]);

  std::vector<rvsdg::Output *> loadstates;
  std::vector<rvsdg::Output *> otherstates;
  for (size_t n = 1; n < operands.size(); n++)
  {
    const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operands[n]);
    if (!is<AllocaOperation>(node) || node == allocaNode)
      loadstates.push_back(operands[n]);
    else
      otherstates.push_back(operands[n]);
  }

  auto ld = LoadNonVolatileOperation::Create(
      operands[0],
      loadstates,
      op.GetLoadedType(),
      op.GetAlignment());

  std::vector<rvsdg::Output *> results(1, ld[0]);
  results.insert(results.end(), std::next(ld.begin()), ld.end());
  results.insert(results.end(), otherstates.begin(), otherstates.end());
  return results;
}

static std::vector<rvsdg::Output *>
perform_load_store_state_reduction(
    const LoadNonVolatileOperation & op,
    const std::vector<rvsdg::Output *> & operands)
{
  auto address = operands[0];
  const auto allocaNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*address);

  std::vector<rvsdg::Output *> new_loadstates;
  std::vector<rvsdg::Output *> results(operands.size(), nullptr);
  for (size_t n = 1; n < operands.size(); n++)
  {
    auto state = operands[n];
    if (is_reducible_state(state, allocaNode))
      results[n] = state;
    else
      new_loadstates.push_back(state);
  }

  auto ld = LoadNonVolatileOperation::Create(
      operands[0],
      new_loadstates,
      op.GetLoadedType(),
      op.GetAlignment());

  results[0] = ld[0];
  for (size_t n = 1, s = 1; n < results.size(); n++)
  {
    if (results[n] == nullptr)
      results[n] = ld[s++];
  }

  return results;
}

static std::vector<rvsdg::Output *>
perform_multiple_origin_reduction(
    const LoadNonVolatileOperation & op,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() > 1);
  const auto address = operands[0];

  std::vector<rvsdg::Output *> newInputStates;
  std::unordered_map<rvsdg::Output *, size_t> stateIndexMap;
  for (size_t n = 1; n < operands.size(); n++)
  {
    auto state = operands[n];
    if (stateIndexMap.find(state) == stateIndexMap.end())
    {
      const size_t resultIndex = 1 + newInputStates.size(); // loaded value + states seen so far
      newInputStates.push_back(state);
      stateIndexMap[state] = resultIndex;
    }
  }

  const auto loadResults = LoadNonVolatileOperation::Create(
      address,
      newInputStates,
      op.GetLoadedType(),
      op.GetAlignment());

  std::vector<rvsdg::Output *> results(operands.size(), nullptr);
  results[0] = loadResults[0];
  for (size_t n = 1; n < operands.size(); n++)
  {
    auto state = operands[n];
    JLM_ASSERT(stateIndexMap.find(state) != stateIndexMap.end());
    results[n] = loadResults[stateIndexMap[state]];
  }

  return results;
}

std::optional<std::vector<rvsdg::Output *>>
LoadNonVolatileOperation::NormalizeLoadMemoryStateMerge(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  auto & address = *operands[0];
  const auto oldLoadMemoryStates = std::vector(std::next(operands.begin()), operands.end());

  bool foundMemoryStateMergeOperation = false;
  std::vector<rvsdg::Output *> newLoadMemoryStates;
  for (const auto memoryState : oldLoadMemoryStates)
  {
    auto [memoryStateMergeNode, memoryStateMergeOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateMergeOperation>(*memoryState);
    if (memoryStateMergeOperation)
    {
      foundMemoryStateMergeOperation = true;
      auto memoryStateMergeOperands = rvsdg::operands(memoryStateMergeNode);
      newLoadMemoryStates.insert(
          newLoadMemoryStates.end(),
          memoryStateMergeOperands.begin(),
          memoryStateMergeOperands.end());
    }
    else
    {
      newLoadMemoryStates.push_back(memoryState);
    }
  }
  if (!foundMemoryStateMergeOperation)
    return std::nullopt;

  auto & newLoadNode =
      CreateNode(address, newLoadMemoryStates, operation.GetLoadedType(), operation.GetAlignment());

  size_t newMemoryStateResultIndex = 1;
  std::vector<rvsdg::Output *> results;
  results.push_back(&LoadedValueOutput(newLoadNode));
  for (auto & oldMemoryStateOperand : oldLoadMemoryStates)
  {
    auto [memoryStateMergeNode, memoryStateMergeOperation] =
        rvsdg::TryGetSimpleNodeAndOptionalOp<MemoryStateMergeOperation>(*oldMemoryStateOperand);
    if (memoryStateMergeOperation)
    {
      size_t numMemoryStates = memoryStateMergeNode->ninputs();
      auto memoryStateMergeOperands =
          rvsdg::Outputs(newLoadNode, newMemoryStateResultIndex, numMemoryStates);
      const auto result = MemoryStateMergeOperation::CreateNode(memoryStateMergeOperands).output(0);
      results.push_back(result);
      newMemoryStateResultIndex += numMemoryStates;
    }
    else
    {
      results.push_back(newLoadNode.output(newMemoryStateResultIndex));
      newMemoryStateResultIndex++;
    }

    JLM_ASSERT(newMemoryStateResultIndex <= newLoadNode.noutputs());
  }

  return results;
}

std::optional<std::vector<rvsdg::Output *>>
LoadNonVolatileOperation::NormalizeLoadStore(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  if (is_load_store_reducible(operation, operands))
    return perform_load_store_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::Output *>>
LoadNonVolatileOperation::NormalizeLoadAlloca(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  if (is_load_alloca_reducible(operands))
    return perform_load_alloca_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::Output *>>
LoadNonVolatileOperation::NormalizeLoadStoreState(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  if (is_load_store_state_reducible(operation, operands))
    return perform_load_store_state_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::Output *>>
LoadNonVolatileOperation::NormalizeDuplicateStates(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  if (is_multiple_origin_reducible(operands))
    return perform_multiple_origin_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::Output *>>
LoadNonVolatileOperation::NormalizeLoadLoadState(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  if (operation.NumMemoryStates() == 0)
  {
    return std::nullopt;
  }

  bool shouldPerformNormalization = false;
  for (size_t n = 1; n < operands.size(); n++)
  {
    const auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operands[n]);
    shouldPerformNormalization |= is<LoadNonVolatileOperation>(simpleNode);
  }
  if (!shouldPerformNormalization)
    return std::nullopt;

  std::function<
      rvsdg::Output *(size_t, rvsdg::Output *, std::vector<std::vector<rvsdg::Output *>> &)>
      traceLoadState = [&](size_t index, rvsdg::Output * operand, auto & joinOperands)
  {
    JLM_ASSERT(rvsdg::is<rvsdg::StateType>(operand->Type()));

    if (!is<LoadNonVolatileOperation>(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*operand)))
      return operand;

    joinOperands[index].push_back(operand);
    return traceLoadState(index, MapMemoryStateOutputToInput(*operand).origin(), joinOperands);
  };

  std::vector<rvsdg::Output *> newLoadMemoryStates;
  std::vector<std::vector<rvsdg::Output *>> joinOperands(operation.NumMemoryStates());
  for (size_t n = 1; n < operands.size(); n++)
  {
    newLoadMemoryStates.push_back(traceLoadState(n - 1, operands[n], joinOperands));
  }

  const auto loadAddress = operands[0];
  auto loadResults = rvsdg::outputs(&CreateNode(
      *loadAddress,
      newLoadMemoryStates,
      operation.GetLoadedType(),
      operation.GetAlignment()));

  for (size_t n = 0; n < joinOperands.size(); n++)
  {
    auto & states = joinOperands[n];
    if (!states.empty())
    {
      states.push_back(loadResults[n + 1]);
      loadResults[n + 1] = MemoryStateJoinOperation::CreateNode(states).output(0);
    }
  }

  return loadResults;
}

std::optional<std::vector<rvsdg::Output *>>
LoadNonVolatileOperation::NormalizeIOBarrierAllocaAddress(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  JLM_ASSERT(operands.size() >= 1);
  const auto address = operands[0];

  auto [ioBarrierNode, ioBarrierOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<IOBarrierOperation>(*address);
  if (!ioBarrierOperation)
    return std::nullopt;

  const auto barredAddress = IOBarrierOperation::BarredInput(*ioBarrierNode).origin();
  if (!rvsdg::IsOwnerNodeOperation<AllocaOperation>(*barredAddress))
    return std::nullopt;

  auto & loadNode = CreateNode(
      *barredAddress,
      { std::next(operands.begin()), operands.end() },
      operation.GetLoadedType(),
      operation.GetAlignment());

  return { outputs(&loadNode) };
}

LoadVolatileOperation::~LoadVolatileOperation() noexcept = default;

bool
LoadVolatileOperation::operator==(const Operation & other) const noexcept
{
  auto operation = dynamic_cast<const LoadVolatileOperation *>(&other);
  return operation && operation->narguments() == narguments()
      && operation->GetLoadedType() == GetLoadedType()
      && operation->GetAlignment() == GetAlignment();
}

std::string
LoadVolatileOperation::debug_string() const
{
  return "LoadVolatile";
}

std::unique_ptr<rvsdg::Operation>
LoadVolatileOperation::copy() const
{
  return std::make_unique<LoadVolatileOperation>(*this);
}

rvsdg::SimpleNode &
LoadVolatileOperation::CreateNode(
    rvsdg::Region & region,
    std::unique_ptr<LoadVolatileOperation> loadOperation,
    const std::vector<rvsdg::Output *> & operands)
{
  return rvsdg::SimpleNode::Create(region, std::move(loadOperation), operands);
}

}
