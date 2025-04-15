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

  auto states = StoreNonVolatileOperation::Create(
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
  return StoreNonVolatileOperation::Create(operands[0], operands[1], states, op.GetAlignment());
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

  auto outputs =
      StoreNonVolatileOperation::Create(address, value, { alloca_state }, op.GetAlignment());
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
      StoreNonVolatileOperation::Create(address, value, newInputStates, operation.GetAlignment());

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
