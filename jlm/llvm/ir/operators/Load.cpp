/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/util/HashSet.hpp>

namespace jlm::llvm
{

const LoadOperation &
LoadNode::GetOperation() const noexcept
{
  return *util::AssertedCast<const LoadOperation>(&SimpleNode::GetOperation());
}

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

size_t
LoadNonVolatileOperation::NumMemoryStates() const noexcept
{
  // Subtracting address
  return narguments() - 1;
}

const LoadNonVolatileOperation &
LoadNonVolatileNode::GetOperation() const noexcept
{
  return *util::AssertedCast<const LoadNonVolatileOperation>(&SimpleNode::GetOperation());
}

[[nodiscard]] LoadNode::MemoryStateInputRange
LoadNonVolatileNode::MemoryStateInputs() const noexcept
{
  if (NumMemoryStates() == 0)
  {
    return { MemoryStateInputIterator(nullptr), MemoryStateInputIterator(nullptr) };
  }

  return { MemoryStateInputIterator(input(1)), MemoryStateInputIterator(nullptr) };
}

[[nodiscard]] LoadNode::MemoryStateOutputRange
LoadNonVolatileNode::MemoryStateOutputs() const noexcept
{
  if (NumMemoryStates() == 0)
  {
    return { MemoryStateOutputIterator(nullptr), MemoryStateOutputIterator(nullptr) };
  }

  return { MemoryStateOutputIterator(output(1)), MemoryStateOutputIterator(nullptr) };
}

LoadNonVolatileNode &
LoadNonVolatileNode::CopyWithNewMemoryStates(
    const std::vector<rvsdg::output *> & memoryStates) const
{
  return CreateNode(
      *GetAddressInput().origin(),
      memoryStates,
      GetOperation().GetLoadedType(),
      GetAlignment());
}

rvsdg::Node *
LoadNonVolatileNode::copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands)
    const
{
  return &CreateNode(*region, GetOperation(), operands);
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

size_t
LoadVolatileOperation::NumMemoryStates() const noexcept
{
  // Subtracting address and I/O state
  return narguments() - 2;
}

[[nodiscard]] const LoadVolatileOperation &
LoadVolatileNode::GetOperation() const noexcept
{
  return *util::AssertedCast<const LoadVolatileOperation>(&LoadNode::GetOperation());
}

[[nodiscard]] LoadNode::MemoryStateInputRange
LoadVolatileNode::MemoryStateInputs() const noexcept
{
  if (NumMemoryStates() == 0)
  {
    return { MemoryStateInputIterator(nullptr), MemoryStateInputIterator(nullptr) };
  }

  return { MemoryStateInputIterator(input(2)), MemoryStateInputIterator(nullptr) };
}

[[nodiscard]] LoadNode::MemoryStateOutputRange
LoadVolatileNode::MemoryStateOutputs() const noexcept
{
  if (NumMemoryStates() == 0)
  {
    return { MemoryStateOutputIterator(nullptr), MemoryStateOutputIterator(nullptr) };
  }

  return { MemoryStateOutputIterator(output(2)), MemoryStateOutputIterator(nullptr) };
}

LoadVolatileNode &
LoadVolatileNode::CopyWithNewMemoryStates(const std::vector<rvsdg::output *> & memoryStates) const
{
  return CreateNode(
      *GetAddressInput().origin(),
      *GetIoStateInput().origin(),
      memoryStates,
      GetOperation().GetLoadedType(),
      GetAlignment());
}

rvsdg::Node *
LoadVolatileNode::copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands) const
{
  return &CreateNode(*region, GetOperation(), operands);
}

/* load normal form */

/*
  sx1 = MemStateMerge si1 ... siM
  v sl1 = load_op a sx1
  =>
  v sl1 ... slM = load_op a si1 ... siM
  sx1 = MemStateMerge sl1 ... slM

  FIXME: The reduction can be generalized: A load node can have multiple operands from different
  merge nodes.
*/
static bool
is_load_mux_reducible(const std::vector<rvsdg::output *> & operands)
{
  // Ignore loads that have no state edge.
  // This can happen when the compiler can statically show that the address of a load is NULL.
  if (operands.size() == 1)
    return false;

  if (operands.size() != 2)
    return false;

  auto memStateMergeNode = rvsdg::output::GetNode(*operands[1]);
  if (!is<MemoryStateMergeOperation>(memStateMergeNode))
    return false;

  return true;
}

/*
  If the producer of a load's address is an alloca, then we can remove
  all state edges originating from other allocas.

  a1 s1 = alloca_op ...
  a2 s2 = alloca_op ...
  s3 = mux_op s1
  v sl1 sl2 sl3 = load_op a1 s1 s2 s3
  =>
  ...
  v sl1 sl3 = load_op a1 s1 s3
*/
static bool
is_load_alloca_reducible(const std::vector<rvsdg::output *> & operands)
{
  auto address = operands[0];

  auto allocanode = rvsdg::output::GetNode(*address);
  if (!is<alloca_op>(allocanode))
    return false;

  for (size_t n = 1; n < operands.size(); n++)
  {
    auto node = rvsdg::output::GetNode(*operands[n]);
    if (is<alloca_op>(node) && node != allocanode)
      return true;
  }

  return false;
}

static bool
is_reducible_state(const rvsdg::output * state, const rvsdg::Node * loadalloca)
{
  if (is<StoreNonVolatileOperation>(rvsdg::output::GetNode(*state)))
  {
    auto storenode = rvsdg::output::GetNode(*state);
    auto addressnode = rvsdg::output::GetNode(*storenode->input(0)->origin());
    if (is<alloca_op>(addressnode) && addressnode != loadalloca)
      return true;
  }

  return false;
}

/*
  a1 sa1 = alloca_op ...
  a2 sa2 = alloca_op ...
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
    const std::vector<rvsdg::output *> & operands)
{
  auto address = operands[0];

  if (operands.size() == 2)
    return false;

  auto allocanode = rvsdg::output::GetNode(*address);
  if (!is<alloca_op>(allocanode))
    return false;

  size_t redstates = 0;
  for (size_t n = 1; n < operands.size(); n++)
  {
    auto state = operands[n];
    if (is_reducible_state(state, allocanode))
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
is_multiple_origin_reducible(const std::vector<rvsdg::output *> & operands)
{
  const util::HashSet<rvsdg::output *> states(std::next(operands.begin()), operands.end());
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
    const std::vector<rvsdg::output *> & operands)
{
  // We do not need to check further if no state edge is provided to the load
  if (operands.size() < 2)
  {
    return false;
  }

  // Check that the first state edge originates from a store
  auto firstState = operands[1];
  auto storeNode = dynamic_cast<const StoreNonVolatileNode *>(rvsdg::output::GetNode(*firstState));
  if (!storeNode)
  {
    return false;
  }

  // Check that all state edges to the load originate from the same store
  if (storeNode->NumMemoryStates() != loadOperation.NumMemoryStates())
  {
    return false;
  }
  for (size_t n = 1; n < operands.size(); n++)
  {
    auto state = operands[n];
    auto node = rvsdg::output::GetNode(*state);
    if (node != storeNode)
    {
      return false;
    }
  }

  // Check that the address to the load and store originate from the same value
  auto loadAddress = operands[0];
  auto storeAddress = storeNode->GetAddressInput().origin();
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
  auto & storedValueType = storeNode->GetStoredValueInput().type();
  if (*loadedValueType != storedValueType)
  {
    return false;
  }

  JLM_ASSERT(loadOperation.GetAlignment() == storeNode->GetAlignment());
  return true;
}

static std::vector<rvsdg::output *>
perform_load_store_reduction(
    const LoadNonVolatileOperation &,
    const std::vector<rvsdg::output *> & operands)
{
  auto storenode = rvsdg::output::GetNode(*operands[1]);

  std::vector<rvsdg::output *> results(1, storenode->input(1)->origin());
  results.insert(results.end(), std::next(operands.begin()), operands.end());

  return results;
}

static std::vector<rvsdg::output *>
perform_load_mux_reduction(
    const LoadNonVolatileOperation & op,
    const std::vector<rvsdg::output *> & operands)
{
  auto memStateMergeNode = rvsdg::output::GetNode(*operands[1]);

  auto ld = LoadNonVolatileNode::Create(
      operands[0],
      rvsdg::operands(memStateMergeNode),
      op.GetLoadedType(),
      op.GetAlignment());

  std::vector<rvsdg::output *> states = { std::next(ld.begin()), ld.end() };
  auto mx = MemoryStateMergeOperation::Create(states);

  return { ld[0], mx };
}

static std::vector<rvsdg::output *>
perform_load_alloca_reduction(
    const LoadNonVolatileOperation & op,
    const std::vector<rvsdg::output *> & operands)
{
  auto allocanode = rvsdg::output::GetNode(*operands[0]);

  std::vector<rvsdg::output *> loadstates;
  std::vector<rvsdg::output *> otherstates;
  for (size_t n = 1; n < operands.size(); n++)
  {
    auto node = rvsdg::output::GetNode(*operands[n]);
    if (!is<alloca_op>(node) || node == allocanode)
      loadstates.push_back(operands[n]);
    else
      otherstates.push_back(operands[n]);
  }

  auto ld =
      LoadNonVolatileNode::Create(operands[0], loadstates, op.GetLoadedType(), op.GetAlignment());

  std::vector<rvsdg::output *> results(1, ld[0]);
  results.insert(results.end(), std::next(ld.begin()), ld.end());
  results.insert(results.end(), otherstates.begin(), otherstates.end());
  return results;
}

static std::vector<rvsdg::output *>
perform_load_store_state_reduction(
    const LoadNonVolatileOperation & op,
    const std::vector<rvsdg::output *> & operands)
{
  auto address = operands[0];
  auto allocanode = rvsdg::output::GetNode(*address);

  std::vector<rvsdg::output *> new_loadstates;
  std::vector<rvsdg::output *> results(operands.size(), nullptr);
  for (size_t n = 1; n < operands.size(); n++)
  {
    auto state = operands[n];
    if (is_reducible_state(state, allocanode))
      results[n] = state;
    else
      new_loadstates.push_back(state);
  }

  auto ld = LoadNonVolatileNode::Create(
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

static std::vector<rvsdg::output *>
perform_multiple_origin_reduction(
    const LoadNonVolatileOperation & op,
    const std::vector<rvsdg::output *> & operands)
{
  JLM_ASSERT(operands.size() > 1);
  const auto address = operands[0];

  std::vector<rvsdg::output *> newInputStates;
  std::unordered_map<rvsdg::output *, size_t> stateIndexMap;
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

  const auto loadResults =
      LoadNonVolatileNode::Create(address, newInputStates, op.GetLoadedType(), op.GetAlignment());

  std::vector<rvsdg::output *> results(operands.size(), nullptr);
  results[0] = loadResults[0];
  for (size_t n = 1; n < operands.size(); n++)
  {
    auto state = operands[n];
    JLM_ASSERT(stateIndexMap.find(state) != stateIndexMap.end());
    results[n] = loadResults[stateIndexMap[state]];
  }

  return results;
}

/*
  _ so1 = load_op _ si1
  _ so2 = load_op _ so1
  _ so3 = load_op _ so2
  =>
  _ so1 = load_op _ si1
  _ so2 = load_op _ si1
  _ so3 = load_op _ si1
*/
static bool
is_load_load_state_reducible(const std::vector<rvsdg::output *> & operands)
{
  JLM_ASSERT(operands.size() >= 2);

  for (size_t n = 1; n < operands.size(); n++)
  {
    if (is<LoadNonVolatileOperation>(rvsdg::output::GetNode(*operands[n])))
      return true;
  }

  return false;
}

static std::vector<rvsdg::output *>
perform_load_load_state_reduction(
    const LoadNonVolatileOperation & op,
    const std::vector<rvsdg::output *> & operands)
{
  size_t nstates = operands.size() - 1;

  auto load_state_input = [](rvsdg::output * result)
  {
    auto ld = rvsdg::output::GetNode(*result);
    JLM_ASSERT(is<LoadNonVolatileOperation>(ld));

    /*
      FIXME: This function returns the corresponding state input for a state output of a load
      node. It should be part of a load node class.
    */
    for (size_t n = 1; n < ld->noutputs(); n++)
    {
      if (result == ld->output(n))
        return ld->input(n);
    }

    JLM_UNREACHABLE("This should have never happened!");
  };

  std::function<
      rvsdg::output *(size_t, rvsdg::output *, std::vector<std::vector<rvsdg::output *>> &)>
      reduce_state = [&](size_t index, rvsdg::output * operand, auto & mxstates)
  {
    JLM_ASSERT(rvsdg::is<rvsdg::StateType>(operand->type()));

    if (!is<LoadNonVolatileOperation>(rvsdg::output::GetNode(*operand)))
      return operand;

    mxstates[index].push_back(operand);
    return reduce_state(index, load_state_input(operand)->origin(), mxstates);
  };

  std::vector<rvsdg::output *> ldstates;
  std::vector<std::vector<rvsdg::output *>> mxstates(nstates);
  for (size_t n = 1; n < operands.size(); n++)
    ldstates.push_back(reduce_state(n - 1, operands[n], mxstates));

  auto ld =
      LoadNonVolatileNode::Create(operands[0], ldstates, op.GetLoadedType(), op.GetAlignment());
  for (size_t n = 0; n < mxstates.size(); n++)
  {
    auto & states = mxstates[n];
    if (!states.empty())
    {
      states.push_back(ld[n + 1]);
      ld[n + 1] = MemoryStateMergeOperation::Create(states);
    }
  }

  return ld;
}

std::optional<std::vector<rvsdg::output *>>
NormalizeLoadMux(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  if (is_load_mux_reducible(operands))
    return perform_load_mux_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::output *>>
NormalizeLoadStore(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  if (is_load_store_reducible(operation, operands))
    return perform_load_store_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::output *>>
NormalizeLoadAlloca(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  if (is_load_alloca_reducible(operands))
    return perform_load_alloca_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::output *>>
NormalizeLoadStoreState(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  if (is_load_store_state_reducible(operation, operands))
    return perform_load_store_state_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::output *>>
NormalizeLoadDuplicateState(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  if (is_multiple_origin_reducible(operands))
    return perform_multiple_origin_reduction(operation, operands);

  return std::nullopt;
}

std::optional<std::vector<rvsdg::output *>>
NormalizeLoadLoadState(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  if (is_load_load_state_reducible(operands))
    return perform_load_load_state_reduction(operation, operands);

  return std::nullopt;
}

}
