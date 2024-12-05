/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>

namespace jlm::llvm
{

const LoadOperation &
LoadNode::GetOperation() const noexcept
{
  return *util::AssertedCast<const LoadOperation>(&simple_node::GetOperation());
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
  return *util::AssertedCast<const LoadNonVolatileOperation>(&simple_node::GetOperation());
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

LoadMuxReduction::~LoadMuxReduction() noexcept = default;

LoadMuxReduction::LoadMuxReduction() noexcept = default;

bool
LoadMuxReduction::IsApplicable(
    const LoadNonVolatileOperation &,
    const std::vector<rvsdg::output *> & operands)
{
  // Ignore loads that have no state edge.
  // This can happen when the compiler can statically show that the address of a load is NULL.
  if (operands.size() == 1)
    return false;

  if (operands.size() != 2)
    return false;

  if (const auto memStateMergeNode = rvsdg::output::GetNode(*operands[1]);
      !is<MemoryStateMergeOperation>(memStateMergeNode))
    return false;

  return true;
}

std::vector<rvsdg::output *>
LoadMuxReduction::ApplyNormalization(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  const auto memStateMergeNode = rvsdg::output::GetNode(*operands[1]);

  auto ld = LoadNonVolatileNode::Create(
      operands[0],
      rvsdg::operands(memStateMergeNode),
      operation.GetLoadedType(),
      operation.GetAlignment());

  const std::vector<rvsdg::output *> states = { std::next(ld.begin()), ld.end() };
  auto mx = MemoryStateMergeOperation::Create(states);

  return { ld[0], mx };
}

LoadMuxReduction &
LoadMuxReduction::GetInstance() noexcept
{
  static LoadMuxReduction loadMuxReduction;
  return loadMuxReduction;
}

LoadAllocaReduction::~LoadAllocaReduction() noexcept = default;

LoadAllocaReduction::LoadAllocaReduction() noexcept = default;

bool
LoadAllocaReduction::IsApplicable(
    const LoadNonVolatileOperation &,
    const std::vector<rvsdg::output *> & operands)
{
  const auto address = operands[0];

  const auto allocaNode = rvsdg::output::GetNode(*address);
  if (!is<alloca_op>(allocaNode))
    return false;

  for (size_t n = 1; n < operands.size(); n++)
  {
    if (const auto node = rvsdg::output::GetNode(*operands[n]);
        is<alloca_op>(node) && node != allocaNode)
      return true;
  }

  return false;
}

std::vector<rvsdg::output *>
LoadAllocaReduction::ApplyNormalization(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  const auto allocaNode = rvsdg::output::GetNode(*operands[0]);

  std::vector<rvsdg::output *> loadStates;
  std::vector<rvsdg::output *> otherStates;
  for (size_t n = 1; n < operands.size(); n++)
  {
    if (const auto node = rvsdg::output::GetNode(*operands[n]);
        !is<alloca_op>(node) || node == allocaNode)
      loadStates.push_back(operands[n]);
    else
      otherStates.push_back(operands[n]);
  }

  auto ld = LoadNonVolatileNode::Create(
      operands[0],
      loadStates,
      operation.GetLoadedType(),
      operation.GetAlignment());

  std::vector<rvsdg::output *> results(1, ld[0]);
  results.insert(results.end(), std::next(ld.begin()), ld.end());
  results.insert(results.end(), otherStates.begin(), otherStates.end());
  return results;
}

LoadAllocaReduction &
LoadAllocaReduction::GetInstance() noexcept
{
  static LoadAllocaReduction loadAllocaReduction;
  return loadAllocaReduction;
}

LoadStoreReduction::~LoadStoreReduction() noexcept = default;

LoadStoreReduction::LoadStoreReduction() noexcept = default;

bool
LoadStoreReduction::IsApplicable(
    const LoadNonVolatileOperation & loadOperation,
    const std::vector<rvsdg::output *> & operands)
{
  // We do not need to check further if no state edge is provided to the load
  if (operands.size() < 2)
  {
    return false;
  }

  // Check that the first state edge originates from a store
  const auto firstState = operands[1];
  const auto storeNode =
      dynamic_cast<const StoreNonVolatileNode *>(rvsdg::output::GetNode(*firstState));
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
    const auto state = operands[n];
    if (const auto node = rvsdg::output::GetNode(*state); node != storeNode)
    {
      return false;
    }
  }

  // Check that the address to the load and store originate from the same value
  const auto loadAddress = operands[0];
  if (const auto storeAddress = storeNode->GetAddressInput().origin(); loadAddress != storeAddress)
  {
    return false;
  }

  // Check that the loaded and stored value type are the same
  //
  // FIXME: This is too restrictive and can be improved upon by inserting truncation or narrowing
  // operations instead. For example, a store of a 32 bit integer followed by a load of a 8 bit
  // integer can be converted to a trunc operation.
  const auto loadedValueType = loadOperation.GetLoadedType();
  if (auto & storedValueType = storeNode->GetStoredValueInput().type();
      *loadedValueType != storedValueType)
  {
    return false;
  }

  JLM_ASSERT(loadOperation.GetAlignment() == storeNode->GetAlignment());
  return true;
}

std::vector<rvsdg::output *>
LoadStoreReduction::ApplyNormalization(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  auto storeNode = rvsdg::output::GetNode(*operands[1]);

  std::vector<rvsdg::output *> results(1, storeNode->input(1)->origin());
  results.insert(results.end(), std::next(operands.begin()), operands.end());

  return results;
}

LoadStoreReduction &
LoadStoreReduction::GetInstance() noexcept
{
  static LoadStoreReduction loadStoreReduction;
  return loadStoreReduction;
}

LoadStoreStateReduction::~LoadStoreStateReduction() noexcept = default;

LoadStoreStateReduction::LoadStoreStateReduction() noexcept = default;

bool
LoadStoreStateReduction::IsReducibleState(
    const rvsdg::output * state,
    const rvsdg::node * loadAlloca)
{
  if (is<StoreNonVolatileOperation>(rvsdg::output::GetNode(*state)))
  {
    const auto storeNode = rvsdg::output::GetNode(*state);
    const auto addressNode = rvsdg::output::GetNode(*storeNode->input(0)->origin());
    if (is<alloca_op>(addressNode) && addressNode != loadAlloca)
      return true;
  }

  return false;
}

bool
LoadStoreStateReduction::IsApplicable(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  const auto address = operands[0];

  if (operands.size() == 2)
    return false;

  const auto allocaNode = rvsdg::output::GetNode(*address);
  if (!is<alloca_op>(allocaNode))
    return false;

  size_t numReducibleStates = 0;
  for (size_t n = 1; n < operands.size(); n++)
  {
    const auto state = operands[n];
    if (IsReducibleState(state, allocaNode))
      numReducibleStates++;
  }

  return numReducibleStates != operation.NumMemoryStates() && numReducibleStates != 0;
}

std::vector<rvsdg::output *>
LoadStoreStateReduction::ApplyNormalization(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  const auto address = operands[0];
  const auto allocaNode = rvsdg::output::GetNode(*address);

  std::vector<rvsdg::output *> newLoadStates;
  std::vector<rvsdg::output *> results(operands.size(), nullptr);
  for (size_t n = 1; n < operands.size(); n++)
  {
    auto state = operands[n];
    if (IsReducibleState(state, allocaNode))
      results[n] = state;
    else
      newLoadStates.push_back(state);
  }

  const auto ld = LoadNonVolatileNode::Create(
      operands[0],
      newLoadStates,
      operation.GetLoadedType(),
      operation.GetAlignment());

  results[0] = ld[0];
  for (size_t n = 1, s = 1; n < results.size(); n++)
  {
    if (results[n] == nullptr)
      results[n] = ld[s++];
  }

  return results;
}

LoadStoreStateReduction &
LoadStoreStateReduction::GetInstance() noexcept
{
  static LoadStoreStateReduction loadStoreStateReduction;
  return loadStoreStateReduction;
}

LoadDuplicateStateReduction::~LoadDuplicateStateReduction() = default;

LoadDuplicateStateReduction::LoadDuplicateStateReduction() noexcept = default;

bool
LoadDuplicateStateReduction::IsApplicable(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  const std::unordered_set<rvsdg::output *> states(std::next(operands.begin()), operands.end());
  return states.size() != operands.size() - 1;
}

std::vector<rvsdg::output *>
LoadDuplicateStateReduction::ApplyNormalization(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  std::vector<rvsdg::output *> newLoadStates;
  std::unordered_set<rvsdg::output *> seenStates;
  std::vector<rvsdg::output *> results(operands.size(), nullptr);
  for (size_t n = 1; n < operands.size(); n++)
  {
    auto state = operands[n];
    if (seenStates.find(state) != seenStates.end())
      results[n] = state;
    else
      newLoadStates.push_back(state);

    seenStates.insert(state);
  }

  const auto ld = LoadNonVolatileNode::Create(
      operands[0],
      newLoadStates,
      operation.GetLoadedType(),
      operation.GetAlignment());

  results[0] = ld[0];
  for (size_t n = 1, s = 1; n < results.size(); n++)
  {
    if (results[n] == nullptr)
      results[n] = ld[s++];
  }

  return results;
}

LoadDuplicateStateReduction &
LoadDuplicateStateReduction::GetInstance() noexcept
{
  static LoadDuplicateStateReduction loadDuplicateStateReduction;
  return loadDuplicateStateReduction;
}

LoadLoadStateReduction::~LoadLoadStateReduction() noexcept = default;

LoadLoadStateReduction::LoadLoadStateReduction() noexcept = default;

bool
LoadLoadStateReduction::IsApplicable(
    const LoadNonVolatileOperation &,
    const std::vector<rvsdg::output *> & operands)
{
  JLM_ASSERT(operands.size() >= 2);

  for (size_t n = 1; n < operands.size(); n++)
  {
    if (is<LoadNonVolatileOperation>(rvsdg::output::GetNode(*operands[n])))
      return true;
  }

  return false;
}

rvsdg::input &
LoadLoadStateReduction::GetLoadStateInput(const rvsdg::output & output)
{
  const auto loadNode = rvsdg::output::GetNode(output);
  JLM_ASSERT(is<LoadNonVolatileOperation>(loadNode));

  for (size_t n = 1; n < loadNode->noutputs(); n++)
  {
    if (loadNode->output(n) == &output)
      return *loadNode->input(n);
  }

  JLM_UNREACHABLE("This should have never happened!");
}

rvsdg::output &
LoadLoadStateReduction::ReduceState(
    const size_t index,
    rvsdg::output & operand,
    std::vector<std::vector<rvsdg::output *>> & mxStates)
{
  JLM_ASSERT(rvsdg::is<rvsdg::StateType>(operand.type()));

  if (!is<LoadNonVolatileOperation>(rvsdg::output::GetNode(operand)))
    return operand;

  mxStates[index].push_back(&operand);
  return ReduceState(index, *GetLoadStateInput(operand).origin(), mxStates);
}

std::vector<rvsdg::output *>
LoadLoadStateReduction::ApplyNormalization(
    const LoadNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  const size_t numStates = operands.size() - 1;

  std::vector<rvsdg::output *> ldstates;
  std::vector<std::vector<rvsdg::output *>> mxstates(numStates);
  for (size_t n = 1; n < operands.size(); n++)
    ldstates.push_back(&ReduceState(n - 1, *operands[n], mxstates));

  auto ld = LoadNonVolatileNode::Create(
      operands[0],
      ldstates,
      operation.GetLoadedType(),
      operation.GetAlignment());
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

LoadLoadStateReduction &
LoadLoadStateReduction::GetInstance() noexcept
{
  static LoadLoadStateReduction loadLoadStateReduction;
  return loadLoadStateReduction;
}

load_normal_form::~load_normal_form()
{}

load_normal_form::load_normal_form(
    const std::type_info & opclass,
    rvsdg::node_normal_form * parent,
    rvsdg::Graph * graph) noexcept
    : simple_normal_form(opclass, parent, graph),
      enable_load_mux_(false),
      enable_load_store_(false),
      enable_load_alloca_(false),
      enable_load_load_state_(false),
      enable_multiple_origin_(false),
      enable_load_store_state_(false)
{}

bool
load_normal_form::normalize_node(rvsdg::Node * node) const
{
  JLM_ASSERT(is<LoadNonVolatileOperation>(node->GetOperation()));
  auto & operation = *util::AssertedCast<const LoadNonVolatileOperation>(&node->GetOperation());
  const auto operands = rvsdg::operands(node);

  if (!get_mutable())
    return true;

  auto & loadMuxReduction = LoadMuxReduction::GetInstance();
  if (get_load_mux_reducible() && loadMuxReduction.IsApplicable(operation, operands))
  {
    divert_users(node, loadMuxReduction.ApplyNormalization(operation, operands));
    remove(node);
    return false;
  }

  auto & loadStoreReduction = LoadStoreReduction::GetInstance();
  if (get_load_store_reducible() && loadStoreReduction.IsApplicable(operation, operands))
  {
    divert_users(node, loadStoreReduction.ApplyNormalization(operation, operands));
    remove(node);
    return false;
  }

  auto & loadAllocaReduction = LoadAllocaReduction::GetInstance();
  if (get_load_alloca_reducible() && loadAllocaReduction.IsApplicable(operation, operands))
  {
    divert_users(node, loadAllocaReduction.ApplyNormalization(operation, operands));
    remove(node);
    return false;
  }

  auto & loadStoreStateReduction = LoadStoreStateReduction::GetInstance();
  if (get_load_store_state_reducible() && loadStoreStateReduction.IsApplicable(operation, operands))
  {
    divert_users(node, loadStoreStateReduction.ApplyNormalization(operation, operands));
    remove(node);
    return false;
  }

  auto & loadDuplicateStateReduction = LoadDuplicateStateReduction::GetInstance();
  if (get_multiple_origin_reducible()
      && loadDuplicateStateReduction.IsApplicable(operation, operands))
  {
    divert_users(node, loadDuplicateStateReduction.ApplyNormalization(operation, operands));
    remove(node);
    return false;
  }

  auto & loadLoadStateReduction = LoadLoadStateReduction::GetInstance();
  if (get_load_load_state_reducible() && loadLoadStateReduction.IsApplicable(operation, operands))
  {
    divert_users(node, loadLoadStateReduction.ApplyNormalization(operation, operands));
    remove(node);
    return false;
  }

  return simple_normal_form::normalize_node(node);
}

std::vector<rvsdg::output *>
load_normal_form::normalized_create(
    rvsdg::Region * region,
    const rvsdg::SimpleOperation & op,
    const std::vector<rvsdg::output *> & operands) const
{
  JLM_ASSERT(is<LoadNonVolatileOperation>(op));
  auto & operation = *util::AssertedCast<const LoadNonVolatileOperation>(&op);

  if (!get_mutable())
    return simple_normal_form::normalized_create(region, op, operands);

  auto & loadMuxReduction = LoadMuxReduction::GetInstance();
  if (get_load_mux_reducible() && loadMuxReduction.IsApplicable(operation, operands))
    return loadMuxReduction.ApplyNormalization(operation, operands);

  auto & loadStoreReduction = LoadStoreReduction::GetInstance();
  if (get_load_store_reducible() && loadStoreReduction.IsApplicable(operation, operands))
    return loadStoreReduction.ApplyNormalization(operation, operands);

  auto & loadAllocaReduction = LoadAllocaReduction::GetInstance();
  if (get_load_alloca_reducible() && loadAllocaReduction.IsApplicable(operation, operands))
    return loadAllocaReduction.ApplyNormalization(operation, operands);

  auto & loadStoreStateReduction = LoadStoreReduction::GetInstance();
  if (get_load_store_state_reducible() && loadStoreStateReduction.IsApplicable(operation, operands))
    return loadStoreStateReduction.ApplyNormalization(operation, operands);

  auto & loadDuplicateStateReduction = LoadDuplicateStateReduction::GetInstance();
  if (get_multiple_origin_reducible()
      && loadDuplicateStateReduction.IsApplicable(operation, operands))
    return loadDuplicateStateReduction.ApplyNormalization(operation, operands);

  auto & loadLoadStateReduction = LoadLoadStateReduction::GetInstance();
  if (get_load_load_state_reducible() && loadLoadStateReduction.IsApplicable(operation, operands))
    return loadLoadStateReduction.ApplyNormalization(operation, operands);

  return simple_normal_form::normalized_create(region, op, operands);
}

}

namespace
{

static jlm::rvsdg::node_normal_form *
create_load_normal_form(
    const std::type_info & opclass,
    jlm::rvsdg::node_normal_form * parent,
    jlm::rvsdg::Graph * graph)
{
  return new jlm::llvm::load_normal_form(opclass, parent, graph);
}

static void __attribute__((constructor))
register_normal_form()
{
  jlm::rvsdg::node_normal_form::register_factory(
      typeid(jlm::llvm::LoadNonVolatileOperation),
      create_load_normal_form);
}

}
