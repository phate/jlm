/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/util/HashSet.hpp>

namespace jlm::llvm
{

const StoreOperation &
StoreNode::GetOperation() const noexcept
{
  return *util::AssertedCast<const StoreOperation>(&SimpleNode::GetOperation());
}

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
  return "Store";
}

std::unique_ptr<rvsdg::Operation>
StoreNonVolatileOperation::copy() const
{
  return std::make_unique<StoreNonVolatileOperation>(*this);
}

[[nodiscard]] size_t
StoreNonVolatileOperation::NumMemoryStates() const noexcept
{
  return nresults();
}

[[nodiscard]] const StoreNonVolatileOperation &
StoreNonVolatileNode::GetOperation() const noexcept
{
  return *util::AssertedCast<const StoreNonVolatileOperation>(&StoreNode::GetOperation());
}

[[nodiscard]] StoreNode::MemoryStateInputRange
StoreNonVolatileNode::MemoryStateInputs() const noexcept
{
  if (NumMemoryStates() == 0)
  {
    return { MemoryStateInputIterator(nullptr), MemoryStateInputIterator(nullptr) };
  }

  return { MemoryStateInputIterator(input(2)), MemoryStateInputIterator(nullptr) };
}

[[nodiscard]] StoreNode::MemoryStateOutputRange
StoreNonVolatileNode::MemoryStateOutputs() const noexcept
{
  if (NumMemoryStates() == 0)
  {
    return { MemoryStateOutputIterator(nullptr), MemoryStateOutputIterator(nullptr) };
  }

  return { MemoryStateOutputIterator(output(0)), MemoryStateOutputIterator(nullptr) };
}

StoreNonVolatileNode &
StoreNonVolatileNode::CopyWithNewMemoryStates(
    const std::vector<rvsdg::output *> & memoryStates) const
{
  return CreateNode(
      *GetAddressInput().origin(),
      *GetStoredValueInput().origin(),
      memoryStates,
      GetAlignment());
}

rvsdg::Node *
StoreNonVolatileNode::copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands)
    const
{
  return &CreateNode(*region, GetOperation(), operands);
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

[[nodiscard]] size_t
StoreVolatileOperation::NumMemoryStates() const noexcept
{
  // Subtracting I/O state
  return nresults() - 1;
}

[[nodiscard]] const StoreVolatileOperation &
StoreVolatileNode::GetOperation() const noexcept
{
  return *util::AssertedCast<const StoreVolatileOperation>(&StoreNode::GetOperation());
}

[[nodiscard]] StoreNode::MemoryStateInputRange
StoreVolatileNode::MemoryStateInputs() const noexcept
{
  if (NumMemoryStates() == 0)
  {
    return { MemoryStateInputIterator(nullptr), MemoryStateInputIterator(nullptr) };
  }

  return { MemoryStateInputIterator(input(3)), MemoryStateInputIterator(nullptr) };
}

[[nodiscard]] StoreNode::MemoryStateOutputRange
StoreVolatileNode::MemoryStateOutputs() const noexcept
{
  if (NumMemoryStates() == 0)
  {
    return { MemoryStateOutputIterator(nullptr), MemoryStateOutputIterator(nullptr) };
  }

  return { MemoryStateOutputIterator(output(1)), MemoryStateOutputIterator(nullptr) };
}

StoreVolatileNode &
StoreVolatileNode::CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const
{
  return CreateNode(
      *GetAddressInput().origin(),
      *GetStoredValueInput().origin(),
      *GetIoStateInput().origin(),
      memoryStates,
      GetAlignment());
}

rvsdg::Node *
StoreVolatileNode::copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands) const
{
  return &CreateNode(*region, GetOperation(), operands);
}

/* store normal form */

static bool
is_store_mux_reducible(const std::vector<jlm::rvsdg::output *> & operands)
{
  JLM_ASSERT(operands.size() > 2);

  auto memStateMergeNode = jlm::rvsdg::output::GetNode(*operands[2]);
  if (!is<MemoryStateMergeOperation>(memStateMergeNode))
    return false;

  for (size_t n = 2; n < operands.size(); n++)
  {
    auto node = jlm::rvsdg::output::GetNode(*operands[n]);
    if (node != memStateMergeNode)
      return false;
  }

  return true;
}

static bool
is_store_store_reducible(
    const StoreNonVolatileOperation & op,
    const std::vector<jlm::rvsdg::output *> & operands)
{
  JLM_ASSERT(operands.size() > 2);

  auto storenode = jlm::rvsdg::output::GetNode(*operands[2]);
  if (!is<StoreNonVolatileOperation>(storenode))
    return false;

  if (op.NumMemoryStates() != storenode->noutputs())
    return false;

  /* check for same address */
  if (operands[0] != storenode->input(0)->origin())
    return false;

  for (size_t n = 2; n < operands.size(); n++)
  {
    if (jlm::rvsdg::output::GetNode(*operands[n]) != storenode || operands[n]->nusers() != 1)
      return false;
  }

  auto other = static_cast<const StoreNonVolatileOperation *>(&storenode->GetOperation());
  JLM_ASSERT(op.GetAlignment() == other->GetAlignment());
  return true;
}

static bool
is_store_alloca_reducible(const std::vector<jlm::rvsdg::output *> & operands)
{
  if (operands.size() == 3)
    return false;

  auto alloca = jlm::rvsdg::output::GetNode(*operands[0]);
  if (!alloca || !is<alloca_op>(alloca->GetOperation()))
    return false;

  std::unordered_set<jlm::rvsdg::output *> states(
      std::next(std::next(operands.begin())),
      operands.end());
  if (states.find(alloca->output(1)) == states.end())
    return false;

  if (alloca->output(1)->nusers() != 1)
    return false;

  return true;
}

static bool
is_multiple_origin_reducible(const std::vector<jlm::rvsdg::output *> & operands)
{
  const util::HashSet<rvsdg::output *> states(std::next(operands.begin(), 2), operands.end());
  return states.Size() != operands.size() - 2;
}

static std::vector<jlm::rvsdg::output *>
perform_store_mux_reduction(
    const StoreNonVolatileOperation & op,
    const std::vector<jlm::rvsdg::output *> & operands)
{
  auto memStateMergeNode = jlm::rvsdg::output::GetNode(*operands[2]);
  auto memStateMergeOperands = jlm::rvsdg::operands(memStateMergeNode);

  auto states = StoreNonVolatileNode::Create(
      operands[0],
      operands[1],
      memStateMergeOperands,
      op.GetAlignment());
  return { MemoryStateMergeOperation::Create(states) };
}

static std::vector<jlm::rvsdg::output *>
perform_store_store_reduction(
    const StoreNonVolatileOperation & op,
    const std::vector<jlm::rvsdg::output *> & operands)
{
  JLM_ASSERT(is_store_store_reducible(op, operands));
  auto storenode = jlm::rvsdg::output::GetNode(*operands[2]);

  auto storeops = jlm::rvsdg::operands(storenode);
  std::vector<jlm::rvsdg::output *> states(std::next(std::next(storeops.begin())), storeops.end());
  return StoreNonVolatileNode::Create(operands[0], operands[1], states, op.GetAlignment());
}

static std::vector<jlm::rvsdg::output *>
perform_store_alloca_reduction(
    const StoreNonVolatileOperation & op,
    const std::vector<jlm::rvsdg::output *> & operands)
{
  auto value = operands[1];
  auto address = operands[0];
  auto alloca_state = jlm::rvsdg::output::GetNode(*address)->output(1);
  std::unordered_set<jlm::rvsdg::output *> states(
      std::next(std::next(operands.begin())),
      operands.end());

  auto outputs = StoreNonVolatileNode::Create(address, value, { alloca_state }, op.GetAlignment());
  states.erase(alloca_state);
  states.insert(outputs[0]);
  return { states.begin(), states.end() };
}

static std::vector<jlm::rvsdg::output *>
perform_multiple_origin_reduction(
    const StoreNonVolatileOperation & operation,
    const std::vector<jlm::rvsdg::output *> & operands)
{
  // FIXME: Unify with the duplicate state removal reduction of the LoadNonVolatile operation

  JLM_ASSERT(operands.size() > 2);
  const auto address = operands[0];
  const auto value = operands[1];

  std::vector<rvsdg::output *> newInputStates;
  std::unordered_map<rvsdg::output *, size_t> stateIndexMap;
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
      StoreNonVolatileNode::Create(address, value, newInputStates, operation.GetAlignment());

  std::vector<rvsdg::output *> results(operation.nresults(), nullptr);
  for (size_t n = 2; n < operands.size(); n++)
  {
    auto state = operands[n];
    JLM_ASSERT(stateIndexMap.find(state) != stateIndexMap.end());
    results[n - 2] = storeResults[stateIndexMap[state]];
  }

  return results;
}

std::optional<std::vector<rvsdg::output *>>
NormalizeStoreMux(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  if (is_store_mux_reducible(operands))
    return perform_store_mux_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::output *>>
NormalizeStoreStore(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  if (is_store_store_reducible(operation, operands))
    return perform_store_store_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::output *>>
NormalizeStoreAlloca(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  if (is_store_alloca_reducible(operands))
    return perform_store_alloca_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::output *>>
NormalizeStoreDuplicateState(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  if (is_multiple_origin_reducible(operands))
    return perform_multiple_origin_reduction(operation, operands);

  return std::nullopt;
}

}
