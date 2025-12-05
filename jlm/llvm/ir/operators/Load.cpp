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
LoadNonVolatileOperation::NormalizeLoadStore(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::Output *> & operands)
{
  // We do not need to check further if no state edge is provided to the load
  if (operands.size() < 2)
  {
    return std::nullopt;
  }
  const auto loadAddressOperand = operands[0];

  // Check that the first state edge originates from a store
  auto firstState = operands[1];
  auto [storeNode, storeOperation] =
      rvsdg::TryGetSimpleNodeAndOptionalOp<StoreNonVolatileOperation>(*firstState);
  if (!storeOperation)
  {
    return std::nullopt;
  }
  const auto storeAddressOperand = StoreNonVolatileOperation::AddressInput(*storeNode).origin();
  const auto storeValueOperand = StoreNonVolatileOperation::StoredValueInput(*storeNode).origin();

  if (loadAddressOperand != storeAddressOperand)
  {
    return std::nullopt;
  }

  // Check that all state edges to the load originate from the same store
  if (storeOperation->NumMemoryStates() != operation.NumMemoryStates())
  {
    return std::nullopt;
  }
  for (size_t n = 1; n < operands.size(); n++)
  {
    auto state = operands[n];
    const auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*state);
    if (node != storeNode)
    {
      return std::nullopt;
    }
  }

  // Check that the loaded and stored value type are the same
  //
  // FIXME: This is too restrictive and can be improved upon by inserting truncation or narrowing
  // operations instead. For example, a store of a 32 bit integer followed by a load of a 8 bit
  // integer can be converted to a trunc operation.
  auto loadedValueType = operation.GetLoadedType();
  auto & storedValueType = *storeValueOperand->Type();
  if (*loadedValueType != storedValueType)
  {
    return std::nullopt;
  }

  JLM_ASSERT(operation.GetAlignment() == storeOperation->GetAlignment());

  std::vector results(1, storeValueOperand);
  results.insert(results.end(), std::next(operands.begin()), operands.end());

  return results;
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

  auto & barredAddress = *IOBarrierOperation::BarredInput(*ioBarrierNode).origin();
  const auto & tracedAddress = rvsdg::traceOutputIntraProcedurally(barredAddress);
  if (!rvsdg::IsOwnerNodeOperation<AllocaOperation>(tracedAddress))
    return std::nullopt;

  auto & loadNode = CreateNode(
      barredAddress,
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
