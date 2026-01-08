/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <cmath>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/util/Hash.hpp>

namespace jlm::hls
{

BranchOperation::~BranchOperation() noexcept = default;

ForkOperation::~ForkOperation() noexcept = default;

MuxOperation::~MuxOperation() noexcept = default;

SinkOperation::~SinkOperation() noexcept = default;

PredicateBufferOperation::~PredicateBufferOperation() noexcept = default;

LoopConstantBufferOperation::~LoopConstantBufferOperation() noexcept = default;

BundleType::~BundleType() noexcept = default;

LoopOperation::~LoopOperation() noexcept = default;

PrintOperation::~PrintOperation() noexcept = default;

BufferOperation::~BufferOperation() noexcept = default;

TriggerOperation::~TriggerOperation() noexcept = default;

TriggerType::~TriggerType() noexcept = default;

StateGateOperation::~StateGateOperation() noexcept = default;

LoadOperation::~LoadOperation() noexcept = default;

DecoupledLoadOperation::~DecoupledLoadOperation() noexcept = default;

AddressQueueOperation::~AddressQueueOperation() noexcept = default;

MemoryResponseOperation::~MemoryResponseOperation() noexcept = default;

LocalLoadOperation::~LocalLoadOperation() noexcept = default;

LocalMemoryOperation::~LocalMemoryOperation() noexcept = default;

LocalMemoryRequestOperation::~LocalMemoryRequestOperation() noexcept = default;

LocalMemoryResponseOperation::~LocalMemoryResponseOperation() noexcept = default;

LocalStoreOperation::~LocalStoreOperation() noexcept = default;

StoreOperation::~StoreOperation() noexcept = default;

std::size_t
TriggerType::ComputeHash() const noexcept
{
  return typeid(TriggerType).hash_code();
}

rvsdg::TypeKind
TriggerType::Kind() const noexcept
{
  return rvsdg::TypeKind::State;
}

std::shared_ptr<const TriggerType>
TriggerType::Create()
{
  static const TriggerType instance;
  return std::shared_ptr<const TriggerType>(std::shared_ptr<void>(), &instance);
}

std::size_t
BundleType::ComputeHash() const noexcept
{
  std::size_t seed = typeid(BundleType).hash_code();
  for (auto & element : elements_)
  {
    auto firstHash = std::hash<std::string>()(element.first);
    util::combineHashesWithSeed(seed, firstHash, element.second->ComputeHash());
  }

  return seed;
}

rvsdg::TypeKind
BundleType::Kind() const noexcept
{
  return rvsdg::TypeKind::Value;
}

EntryArgument::~EntryArgument() noexcept = default;

EntryArgument &
EntryArgument::Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) const
{
  return Create(region, *input, Type());
}

BackEdgeArgument &
BackEdgeArgument::Copy(rvsdg::Region & region, rvsdg::StructuralInput * input) const
{
  JLM_ASSERT(input == nullptr);
  return create(&region, Type());
}

BackEdgeResult &
BackEdgeResult::Copy(rvsdg::Output & origin, rvsdg::StructuralOutput * output) const
{
  JLM_ASSERT(output == nullptr);
  return create(&origin);
}

ExitResult::~ExitResult() noexcept = default;

ExitResult::ExitResult(rvsdg::Output & origin, rvsdg::StructuralOutput & output)
    : rvsdg::RegionResult(origin.region(), &origin, &output, origin.Type())
{
  JLM_ASSERT(dynamic_cast<const LoopNode *>(origin.region()->node()));
}

ExitResult &
ExitResult::Copy(rvsdg::Output & origin, rvsdg::StructuralOutput * output) const
{
  return Create(origin, *output);
}

rvsdg::StructuralOutput *
LoopNode::AddLoopVar(jlm::rvsdg::Output * origin, jlm::rvsdg::Output ** buffer)
{
  // Create StructuralInput and EntryArgument
  const auto input =
      addInput(std::make_unique<rvsdg::StructuralInput>(this, origin, origin->Type()), true);
  auto & argument_in = EntryArgument::Create(*subregion(), *input, origin->Type());

  // Create back-edge
  auto backedge_argument = add_backedge(origin->Type());
  auto backedge_result = backedge_argument->result();

  // Create Mux to pick between EntryArgument and BackEdgeArgument
  auto mux = MuxOperation::create(
      GetPredicateBuffer(),
      { &argument_in, backedge_argument },
      false,
      true)[0];
  // Give the caller a
  if (buffer != nullptr)
    *buffer = mux;

  // Create Branch to send the result to either an ExitResult or a BackEdgeResult
  // We need to give it a value, so use the output of the mux as the result for now
  auto branch = BranchOperation::create(*predicate()->origin(), *mux, true);

  // Create an ExitResult + StructuralOutput for when the loop is finished
  const auto output = addOutput(std::make_unique<rvsdg::StructuralOutput>(this, origin->Type()));
  ExitResult::Create(*branch[0], *output);

  // If the loop is not done, send the value to the BackEdgeResult, with a small buffer in between.
  auto buf = BufferOperation::create(*branch[1], 2)[0];
  backedge_result->divert_to(buf);
  return output;
}

jlm::rvsdg::Output *
LoopNode::addLoopConstant(jlm::rvsdg::Output * origin)
{
  auto input =
      addInput(std::make_unique<rvsdg::StructuralInput>(this, origin, origin->Type()), true);

  auto & argument_in = EntryArgument::Create(*subregion(), *input, origin->Type());
  auto buffer = LoopConstantBufferOperation::create(GetPredicateBuffer(), argument_in)[0];
  return buffer;
}

rvsdg::Output *
LoopNode::addResponseInput(rvsdg::Output * origin)
{
  const auto input =
      addInput(std::make_unique<rvsdg::StructuralInput>(this, origin, origin->Type()), true);
  return &EntryArgument::Create(*subregion(), *input, origin->Type());
}

rvsdg::Output *
LoopNode::addRequestOutput(rvsdg::Output * origin)
{
  const auto output = addOutput(std::make_unique<rvsdg::StructuralOutput>(this, origin->Type()));
  ExitResult::Create(*origin, *output);
  return output;
}

void
LoopNode::removeLoopOutput(rvsdg::StructuralOutput * output)
{
  JLM_ASSERT(output->node() == this);
  JLM_ASSERT(output->IsDead());
  JLM_ASSERT(output->results.size() == 1);
  auto result = output->results.begin();

  subregion()->RemoveResults({ result->index() });
  RemoveOutputs({ output->index() });
}

void
LoopNode::removeLoopInput(rvsdg::StructuralInput * input)
{
  JLM_ASSERT(input->node() == this);
  JLM_ASSERT(input->arguments.size() == 1);
  auto argument = input->arguments.begin();
  JLM_ASSERT(argument->IsDead());

  subregion()->RemoveArguments({ argument->index() });
  RemoveInputs({ input->index() });
}

[[nodiscard]] const rvsdg::Operation &
LoopNode::GetOperation() const noexcept
{
  static const LoopOperation singleton;
  return singleton;
}

LoopNode *
LoopNode::copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const
{
  auto loop = create(region, false);

  for (size_t i = 0; i < ninputs(); ++i)
  {
    auto in_origin = smap.lookup(input(i)->origin());
    auto inp = loop->addInput(
        std::make_unique<rvsdg::StructuralInput>(loop, in_origin, in_origin->Type()),
        true);

    smap.insert(input(i), loop->input(i));
    auto oarg = input(i)->arguments.begin().ptr();
    auto & narg = EntryArgument::Create(*loop->subregion(), *inp, oarg->Type());
    smap.insert(oarg, &narg);
  }
  for (size_t i = 0; i < noutputs(); ++i)
  {
    auto out = loop->addOutput(std::make_unique<rvsdg::StructuralOutput>(loop, output(i)->Type()));

    smap.insert(output(i), out);
    smap.insert(output(i), out);
  }
  for (size_t i = 0; i < subregion()->narguments(); ++i)
  {
    auto arg = subregion()->argument(i);
    if (auto ba = dynamic_cast<BackEdgeArgument *>(arg))
    {
      auto na = loop->add_backedge(arg->Type());
      smap.insert(ba, na);
    }
  }

  subregion()->copy(loop->subregion(), smap, false, false);
  loop->PredicateBuffer_ = smap.lookup(PredicateBuffer_);
  // redirect backedges
  for (size_t i = 0; i < subregion()->narguments(); ++i)
  {
    auto arg = subregion()->argument(i);
    if (auto ba = dynamic_cast<BackEdgeArgument *>(arg))
    {
      auto na = dynamic_cast<BackEdgeArgument *>(smap.lookup(ba));
      na->result()->divert_to(smap.lookup(ba->result()->origin()));
    }
  }
  for (size_t i = 0; i < noutputs(); ++i)
  {
    auto outp = output(i);
    auto res = outp->results.begin().ptr();
    auto origin = smap.lookup(res->origin());
    ExitResult::Create(*origin, *loop->output(i));
  }

  return loop;
}

BackEdgeArgument *
LoopNode::add_backedge(std::shared_ptr<const jlm::rvsdg::Type> type)
{
  auto & argument_loop = BackEdgeArgument::create(subregion(), std::move(type));
  auto & result_loop = BackEdgeResult::create(&argument_loop);
  argument_loop.result_ = &result_loop;
  result_loop.argument_ = &argument_loop;
  return &argument_loop;
}

LoopNode *
LoopNode::create(rvsdg::Region * parent, bool init)
{
  auto ln = new LoopNode(parent);
  if (init)
  {
    auto predicate = &rvsdg::ControlConstantOperation::createFalse(*ln->subregion());
    auto pred_arg = ln->add_backedge(rvsdg::ControlType::Create(2));
    pred_arg->result()->divert_to(predicate);
    // we need a buffer without pass-through behavior to avoid a combinatorial cycle of ready
    // signals
    auto pre_buffer = BufferOperation::create(*pred_arg, 2)[0];
    ln->PredicateBuffer_ = PredicateBufferOperation::create(*pre_buffer)[0];
  }
  return ln;
}

void
LoopNode::set_predicate(jlm::rvsdg::Output * p)
{
  auto node = rvsdg::TryGetOwnerNode<Node>(*predicate()->origin());
  predicate()->origin()->divert_users(p);
  if (node && node->IsDead())
    remove(node);
}

std::shared_ptr<const BundleType>
get_mem_req_type(std::shared_ptr<const rvsdg::Type> elementType, bool write)
{
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> elements;
  elements.emplace_back("addr", llvm::PointerType::Create());
  elements.emplace_back("size", jlm::rvsdg::BitType::Create(4));
  elements.emplace_back("id", jlm::rvsdg::BitType::Create(8));
  if (write)
  {
    elements.emplace_back("data", std::move(elementType));
    elements.emplace_back("write", jlm::rvsdg::BitType::Create(1));
  }
  return std::make_shared<BundleType>(std::move(elements));
}

std::shared_ptr<const BundleType>
get_mem_res_type(std::shared_ptr<const jlm::rvsdg::Type> dataType)
{
  std::vector<std::pair<std::string, std::shared_ptr<const jlm::rvsdg::Type>>> elements;
  elements.emplace_back("data", std::move(dataType));
  elements.emplace_back("id", jlm::rvsdg::BitType::Create(8));
  return std::make_shared<BundleType>(std::move(elements));
}

int
JlmSize(const jlm::rvsdg::Type * type)
{
  if (auto bt = dynamic_cast<const jlm::rvsdg::BitType *>(type))
  {
    return bt->nbits();
  }
  else if (auto at = dynamic_cast<const llvm::ArrayType *>(type))
  {
    return JlmSize(&at->element_type()) * at->nelements();
  }
  else if (auto vt = dynamic_cast<const llvm::VectorType *>(type))
  {
    return JlmSize(&vt->type()) * vt->size();
  }
  else if (dynamic_cast<const llvm::PointerType *>(type))
  {
    return GetPointerSizeInBits();
  }
  else if (auto ct = dynamic_cast<const rvsdg::ControlType *>(type))
  {
    return ceil(log2(ct->nalternatives()));
  }
  else if (type->Kind() == rvsdg::TypeKind::State)
  {
    return 1;
  }
  else if (rvsdg::is<BundleType>(*type))
  {
    // TODO: fix this ugly hack needed for get_node_name
    return 0;
  }
  else if (auto ft = dynamic_cast<const llvm::FloatingPointType *>(type))
  {
    switch (ft->size())
    {
    case llvm::fpsize::half:
      return 16;
    case llvm::fpsize::flt:
      return 32;
    case llvm::fpsize::dbl:
      return 64;
    default:
      throw std::logic_error("Size of '" + type->debug_string() + "' is not implemented!");
    }
  }
  else
  {
    throw std::logic_error("Size of '" + type->debug_string() + "' is not implemented!");
  }
}

size_t
GetPointerSizeInBits()
{
  return 64;
}
}
