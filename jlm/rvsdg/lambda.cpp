/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/lambda.hpp>
#include <jlm/util/strfmt.hpp>

namespace jlm::rvsdg
{

LambdaOperation::~LambdaOperation() = default;

LambdaOperation::LambdaOperation(std::shared_ptr<const FunctionType> type)
    : type_(std::move(type))
{}

std::string
LambdaOperation::debug_string() const
{
  return util::strfmt("Lambda[", Type()->debug_string(), "]");
}

bool
LambdaOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const LambdaOperation *>(&other);
  return op && op->type() == type();
}

std::unique_ptr<rvsdg::Operation>
LambdaOperation::copy() const
{
  return std::make_unique<LambdaOperation>(*this);
}

LambdaNode::~LambdaNode() = default;

LambdaNode::LambdaNode(rvsdg::Region & parent, std::unique_ptr<LambdaOperation> op)
    : StructuralNode(&parent, 1),
      Operation_(std::move(op))
{
  for (auto & argumentType : GetOperation().Type()->Arguments())
  {
    rvsdg::RegionArgument::Create(*subregion(), nullptr, argumentType);
  }
}

LambdaOperation &
LambdaNode::GetOperation() const noexcept
{
  return *Operation_;
}

[[nodiscard]] std::vector<rvsdg::Output *>
LambdaNode::GetFunctionArguments() const
{
  std::vector<rvsdg::Output *> arguments;
  const auto & type = GetOperation().Type();
  for (std::size_t n = 0; n < type->Arguments().size(); ++n)
  {
    arguments.push_back(subregion()->argument(n));
  }
  return arguments;
}

[[nodiscard]] std::vector<rvsdg::Input *>
LambdaNode::GetFunctionResults() const
{
  std::vector<rvsdg::Input *> results;
  for (std::size_t n = 0; n < subregion()->nresults(); ++n)
  {
    results.push_back(subregion()->result(n));
  }
  return results;
}

[[nodiscard]] LambdaNode::ContextVar
LambdaNode::MapInputContextVar(const rvsdg::Input & input) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<LambdaNode>(input) == this);
  return ContextVar{ const_cast<rvsdg::Input *>(&input),
                     subregion()->argument(GetOperation().Type()->NumArguments() + input.index()) };
}

[[nodiscard]] std::optional<LambdaNode::ContextVar>
LambdaNode::MapBinderContextVar(const rvsdg::Output & output) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetOwnerRegion(output) == subregion());
  auto numArguments = GetOperation().Type()->NumArguments();
  if (output.index() >= numArguments)
  {
    return ContextVar{ input(output.index() - GetOperation().Type()->NumArguments()),
                       const_cast<rvsdg::Output *>(&output) };
  }
  else
  {
    return std::nullopt;
  }
}

std::variant<LambdaNode::ArgumentVar, LambdaNode::ContextVar>
LambdaNode::MapArgument(const rvsdg::Output & output) const
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<LambdaNode>(output) == this);
  std::size_t nargs = GetOperation().Type()->NumArguments();
  if (output.index() < nargs)
  {
    return ArgumentVar{ subregion()->argument(output.index()) };
  }
  else
  {
    return ContextVar{ input(output.index() - nargs), subregion()->argument(output.index()) };
  }
}

[[nodiscard]] std::vector<LambdaNode::ContextVar>
LambdaNode::GetContextVars() const noexcept
{
  std::vector<ContextVar> vars;
  for (size_t n = 0; n < ninputs(); ++n)
  {
    vars.push_back(
        ContextVar{ input(n), subregion()->argument(n + GetOperation().Type()->NumArguments()) });
  }
  return vars;
}

LambdaNode::ContextVar
LambdaNode::AddContextVar(jlm::rvsdg::Output & origin)
{
  const auto input =
      addInput(std::make_unique<StructuralInput>(this, &origin, origin.Type()), true);
  const auto argument = &RegionArgument::Create(*subregion(), input, origin.Type());
  return ContextVar{ input, argument };
}

LambdaNode *
LambdaNode::Create(rvsdg::Region & parent, std::unique_ptr<LambdaOperation> operation)
{
  return new LambdaNode(parent, std::move(operation));
}

rvsdg::Output *
LambdaNode::finalize(const std::vector<jlm::rvsdg::Output *> & results)
{
  /* check if finalized was already called */
  if (noutputs() > 0)
  {
    JLM_ASSERT(noutputs() == 1);
    return output();
  }

  if (GetOperation().type().NumResults() != results.size())
    throw util::Error("Incorrect number of results.");

  for (size_t n = 0; n < results.size(); n++)
  {
    auto & expected = GetOperation().type().ResultType(n);
    auto & received = *results[n]->Type();
    if (*results[n]->Type() != GetOperation().type().ResultType(n))
      throw util::Error("Expected " + expected.debug_string() + ", got " + received.debug_string());

    if (results[n]->region() != subregion())
      throw util::Error("Invalid operand region.");
  }

  for (const auto & origin : results)
    rvsdg::RegionResult::Create(*origin->region(), *origin, nullptr, origin->Type());

  return addOutput(std::make_unique<StructuralOutput>(this, GetOperation().Type()));
}

rvsdg::Output *
LambdaNode::output() const noexcept
{
  return StructuralNode::output(0);
}

LambdaNode *
LambdaNode::copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::Output *> & operands) const
{
  return util::assertedCast<LambdaNode>(rvsdg::Node::copy(region, operands));
}

LambdaNode *
LambdaNode::copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const
{
  const auto & op = GetOperation();
  auto lambda = Create(
      *region,
      std::unique_ptr<LambdaOperation>(util::assertedCast<LambdaOperation>(op.copy().release())));

  /* add context variables */
  rvsdg::SubstitutionMap subregionmap;
  for (const auto & cv : GetContextVars())
  {
    auto origin = smap.lookup(cv.input->origin());
    subregionmap.insert(cv.inner, lambda->AddContextVar(*origin).inner);
  }

  /* collect function arguments */
  auto args = GetFunctionArguments();
  auto newArgs = lambda->GetFunctionArguments();
  JLM_ASSERT(args.size() == newArgs.size());
  for (std::size_t n = 0; n < args.size(); ++n)
  {
    subregionmap.insert(args[n], newArgs[n]);
  }

  /* copy subregion */
  subregion()->copy(lambda->subregion(), subregionmap, false, false);

  /* collect function results */
  std::vector<jlm::rvsdg::Output *> results;
  for (auto result : GetFunctionResults())
    results.push_back(subregionmap.lookup(result->origin()));

  /* finalize lambda */
  auto o = lambda->finalize(results);
  smap.insert(output(), o);

  return lambda;
}

LambdaBuilder::LambdaBuilder(Region & region, std::vector<std::shared_ptr<const Type>> argtypes)
    : Node_(
          LambdaNode::Create(
              region,
              std::make_unique<LambdaOperation>(FunctionType::Create(std::move(argtypes), {}))))
{
  // Note that the above inserts a "placeholder" function type, for now.
  // This is to avoid requiring the caller to specify the return type(s)
  // already when starting to construct the object. It is sometimes easier
  // to let them be determined while building.
}

std::vector<Output *>
LambdaBuilder::Arguments()
{
  JLM_ASSERT(Node_);
  return Node_->GetFunctionArguments();
}

rvsdg::Region *
LambdaBuilder::GetRegion() noexcept
{
  JLM_ASSERT(Node_);
  return Node_->subregion();
}

LambdaNode::ContextVar
LambdaBuilder::AddContextVar(jlm::rvsdg::Output & origin)
{
  JLM_ASSERT(Node_);
  return Node_->AddContextVar(origin);
}

Output &
LambdaBuilder::Finalize(
    const std::vector<jlm::rvsdg::Output *> & results,
    std::unique_ptr<LambdaOperation> op)
{
  JLM_ASSERT(Node_);
  Node_->Operation_ = std::move(op);
  auto output = Node_->finalize(results);
  Node_ = nullptr;
  return *output;
}

}
