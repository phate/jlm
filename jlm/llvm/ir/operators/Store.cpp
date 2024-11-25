/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/MemoryStateOperations.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>

namespace jlm::llvm
{

StoreNonVolatileOperation::~StoreNonVolatileOperation() noexcept = default;

bool
StoreNonVolatileOperation::operator==(const operation & other) const noexcept
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

std::unique_ptr<jlm::rvsdg::operation>
StoreNonVolatileOperation::copy() const
{
  return std::unique_ptr<jlm::rvsdg::operation>(new StoreNonVolatileOperation(*this));
}

[[nodiscard]] size_t
StoreNonVolatileOperation::NumMemoryStates() const noexcept
{
  return nresults();
}

[[nodiscard]] const StoreNonVolatileOperation &
StoreNonVolatileNode::GetOperation() const noexcept
{
  return *util::AssertedCast<const StoreNonVolatileOperation>(&operation());
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

rvsdg::node *
StoreNonVolatileNode::copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands)
    const
{
  return &CreateNode(*region, GetOperation(), operands);
}

StoreVolatileOperation::~StoreVolatileOperation() noexcept = default;

bool
StoreVolatileOperation::operator==(const operation & other) const noexcept
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

std::unique_ptr<rvsdg::operation>
StoreVolatileOperation::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new StoreVolatileOperation(*this));
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
  return *util::AssertedCast<const StoreVolatileOperation>(&operation());
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

rvsdg::node *
StoreVolatileNode::copy(rvsdg::Region * region, const std::vector<rvsdg::output *> & operands) const
{
  return &CreateNode(*region, GetOperation(), operands);
}

/* store normal form */

StoreMuxReduction::~StoreMuxReduction() = default;

bool
StoreMuxReduction::IsApplicable(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  JLM_ASSERT(operands.size() > 2);

  const auto memStateMergeNode = rvsdg::output::GetNode(*operands[2]);
  if (!is<MemoryStateMergeOperation>(memStateMergeNode))
    return false;

  for (size_t n = 2; n < operands.size(); n++)
  {
    if (const auto node = rvsdg::output::GetNode(*operands[n]); node != memStateMergeNode)
      return false;
  }

  return true;
}

std::vector<rvsdg::output *>
StoreMuxReduction::ApplyNormalization(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  const auto memStateMergeNode = jlm::rvsdg::output::GetNode(*operands[2]);
  const auto memStateMergeOperands = jlm::rvsdg::operands(memStateMergeNode);

  const auto states = StoreNonVolatileNode::Create(
      operands[0],
      operands[1],
      memStateMergeOperands,
      operation.GetAlignment());
  return { MemoryStateMergeOperation::Create(states) };
}

StoreStoreReduction::~StoreStoreReduction() = default;

bool
StoreStoreReduction::IsApplicable(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  JLM_ASSERT(operands.size() > 2);

  const auto storeNode = rvsdg::output::GetNode(*operands[2]);
  if (!is<StoreNonVolatileOperation>(storeNode))
    return false;

  if (operation.NumMemoryStates() != storeNode->noutputs())
    return false;

  // check for same address
  if (operands[0] != storeNode->input(0)->origin())
    return false;

  for (size_t n = 2; n < operands.size(); n++)
  {
    if (rvsdg::output::GetNode(*operands[n]) != storeNode || operands[n]->nusers() != 1)
      return false;
  }

  const auto other = static_cast<const StoreNonVolatileOperation *>(&storeNode->operation());
  JLM_ASSERT(operation.GetAlignment() == other->GetAlignment());
  return true;
}

std::vector<rvsdg::output *>
StoreStoreReduction::ApplyNormalization(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  JLM_ASSERT(IsApplicable(operation, operands));
  const auto storeNode = rvsdg::output::GetNode(*operands[2]);

  auto storeOperands = rvsdg::operands(storeNode);
  const std::vector states(std::next(std::next(storeOperands.begin())), storeOperands.end());
  return StoreNonVolatileNode::Create(operands[0], operands[1], states, operation.GetAlignment());
}

StoreAllocaReduction::~StoreAllocaReduction() = default;

bool
StoreAllocaReduction::IsApplicable(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  if (operands.size() == 3)
    return false;

  const auto alloca = rvsdg::output::GetNode(*operands[0]);
  if (!alloca || !is<alloca_op>(alloca->operation()))
    return false;

  if (std::unordered_set states(std::next(std::next(operands.begin())), operands.end());
      states.find(alloca->output(1)) == states.end())
    return false;

  if (alloca->output(1)->nusers() != 1)
    return false;

  return true;
}

std::vector<rvsdg::output *>
StoreAllocaReduction::ApplyNormalization(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  const auto value = operands[1];
  const auto address = operands[0];
  auto alloca_state = rvsdg::output::GetNode(*address)->output(1);
  std::unordered_set states(std::next(std::next(operands.begin())), operands.end());

  const auto outputs =
      StoreNonVolatileNode::Create(address, value, { alloca_state }, operation.GetAlignment());
  states.erase(alloca_state);
  states.insert(outputs[0]);
  return { states.begin(), states.end() };
}

StoreDuplicateStateReduction::~StoreDuplicateStateReduction() = default;

bool
StoreDuplicateStateReduction::IsApplicable(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  const std::unordered_set states(std::next(std::next(operands.begin())), operands.end());
  return states.size() != operands.size() - 2;
}

std::vector<rvsdg::output *>
StoreDuplicateStateReduction::ApplyNormalization(
    const StoreNonVolatileOperation & operation,
    const std::vector<rvsdg::output *> & operands)
{
  std::unordered_set states(std::next(std::next(operands.begin())), operands.end());
  return StoreNonVolatileNode::Create(
      operands[0],
      operands[1],
      { states.begin(), states.end() },
      operation.GetAlignment());
}

static StoreMuxReduction storeMuxReduction;
static StoreStoreReduction storeStoreReduction;
static StoreAllocaReduction storeAllocaReduction;
static StoreDuplicateStateReduction storeDuplicateStateReduction;

store_normal_form::~store_normal_form()
{}

store_normal_form::store_normal_form(
    const std::type_info & opclass,
    jlm::rvsdg::node_normal_form * parent,
    jlm::rvsdg::graph * graph) noexcept
    : simple_normal_form(opclass, parent, graph),
      enable_store_mux_(false),
      enable_store_store_(false),
      enable_store_alloca_(false),
      enable_multiple_origin_(false)
{
  if (auto p = dynamic_cast<const store_normal_form *>(parent))
  {
    enable_multiple_origin_ = p->enable_multiple_origin_;
    enable_store_store_ = p->enable_store_store_;
    enable_store_mux_ = p->enable_store_mux_;
  }
}

bool
store_normal_form::normalize_node(jlm::rvsdg::node * node) const
{
  JLM_ASSERT(is<StoreNonVolatileOperation>(node->operation()));
  auto & operation = *util::AssertedCast<const StoreNonVolatileOperation>(&node->operation());
  auto operands = rvsdg::operands(node);

  if (!get_mutable())
    return true;

  if (get_store_mux_reducible() && storeMuxReduction.IsApplicable(operation, operands))
  {
    divert_users(node, storeMuxReduction.ApplyNormalization(operation, operands));
    node->region()->remove_node(node);
    return false;
  }

  if (get_store_store_reducible() && storeStoreReduction.IsApplicable(operation, operands))
  {
    divert_users(node, storeStoreReduction.ApplyNormalization(operation, operands));
    remove(node);
    return false;
  }

  if (get_store_alloca_reducible() && storeAllocaReduction.IsApplicable(operation, operands))
  {
    divert_users(node, storeAllocaReduction.ApplyNormalization(operation, operands));
    node->region()->remove_node(node);
    return false;
  }

  if (get_multiple_origin_reducible()
      && storeDuplicateStateReduction.IsApplicable(operation, operands))
  {
    auto outputs = storeDuplicateStateReduction.ApplyNormalization(operation, operands);
    auto new_node = jlm::rvsdg::output::GetNode(*outputs[0]);

    std::unordered_map<jlm::rvsdg::output *, jlm::rvsdg::output *> origin2output;
    for (size_t n = 0; n < outputs.size(); n++)
    {
      auto origin = new_node->input(n + 2)->origin();
      JLM_ASSERT(origin2output.find(origin) == origin2output.end());
      origin2output[origin] = outputs[n];
    }

    for (size_t n = 2; n < node->ninputs(); n++)
    {
      auto origin = node->input(n)->origin();
      JLM_ASSERT(origin2output.find(origin) != origin2output.end());
      node->output(n - 2)->divert_users(origin2output[origin]);
    }
    remove(node);
    return false;
  }

  return simple_normal_form::normalize_node(node);
}

std::vector<jlm::rvsdg::output *>
store_normal_form::normalized_create(
    rvsdg::Region * region,
    const jlm::rvsdg::simple_op & op,
    const std::vector<jlm::rvsdg::output *> & operands) const
{
  auto & operation = *util::AssertedCast<const StoreNonVolatileOperation>(&op);

  if (!get_mutable())
    return simple_normal_form::normalized_create(region, op, operands);

  if (get_store_mux_reducible() && storeMuxReduction.IsApplicable(operation, operands))
    return storeMuxReduction.ApplyNormalization(operation, operands);

  if (get_store_alloca_reducible() && storeAllocaReduction.IsApplicable(operation, operands))
    return storeAllocaReduction.ApplyNormalization(operation, operands);

  if (get_multiple_origin_reducible()
      && storeDuplicateStateReduction.IsApplicable(operation, operands))
    return storeDuplicateStateReduction.ApplyNormalization(operation, operands);

  return simple_normal_form::normalized_create(region, op, operands);
}

void
store_normal_form::set_store_mux_reducible(bool enable)
{
  if (get_store_mux_reducible() == enable)
    return;

  children_set<store_normal_form, &store_normal_form::set_store_mux_reducible>(enable);

  enable_store_mux_ = enable;
  if (get_mutable() && enable)
    graph()->mark_denormalized();
}

void
store_normal_form::set_store_store_reducible(bool enable)
{
  if (get_store_store_reducible() == enable)
    return;

  children_set<store_normal_form, &store_normal_form::set_store_store_reducible>(enable);

  enable_store_store_ = enable;
  if (get_mutable() && enable)
    graph()->mark_denormalized();
}

void
store_normal_form::set_store_alloca_reducible(bool enable)
{
  if (get_store_alloca_reducible() == enable)
    return;

  children_set<store_normal_form, &store_normal_form::set_store_alloca_reducible>(enable);

  enable_store_alloca_ = enable;
  if (get_mutable() && enable)
    graph()->mark_denormalized();
}

void
store_normal_form::set_multiple_origin_reducible(bool enable)
{
  if (get_multiple_origin_reducible() == enable)
    return;

  children_set<store_normal_form, &store_normal_form::set_multiple_origin_reducible>(enable);

  enable_multiple_origin_ = enable;
  if (get_mutable() && enable)
    graph()->mark_denormalized();
}

}

namespace
{

static jlm::rvsdg::node_normal_form *
create_store_normal_form(
    const std::type_info & opclass,
    jlm::rvsdg::node_normal_form * parent,
    jlm::rvsdg::graph * graph)
{
  return new jlm::llvm::store_normal_form(opclass, parent, graph);
}

static void __attribute__((constructor))
register_normal_form()
{
  jlm::rvsdg::node_normal_form::register_factory(
      typeid(jlm::llvm::StoreNonVolatileOperation),
      create_store_normal_form);
}

}
