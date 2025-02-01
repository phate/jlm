/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2025 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/lambda.hpp>

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

[[nodiscard]] std::vector<rvsdg::output *>
LambdaNode::GetFunctionArguments() const
{
  std::vector<rvsdg::output *> arguments;
  const auto & type = GetOperation().Type();
  for (std::size_t n = 0; n < type->Arguments().size(); ++n)
  {
    arguments.push_back(subregion()->argument(n));
  }
  return arguments;
}

[[nodiscard]] std::vector<rvsdg::input *>
LambdaNode::GetFunctionResults() const
{
  std::vector<rvsdg::input *> results;
  for (std::size_t n = 0; n < subregion()->nresults(); ++n)
  {
    results.push_back(subregion()->result(n));
  }
  return results;
}

[[nodiscard]] LambdaNode::ContextVar
LambdaNode::MapInputContextVar(const rvsdg::input & input) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<LambdaNode>(input) == this);
  return ContextVar{ const_cast<rvsdg::input *>(&input),
                     subregion()->argument(GetOperation().Type()->NumArguments() + input.index()) };
}

[[nodiscard]] std::optional<LambdaNode::ContextVar>
LambdaNode::MapBinderContextVar(const rvsdg::output & output) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetOwnerRegion(output) == subregion());
  auto numArguments = GetOperation().Type()->NumArguments();
  if (output.index() >= numArguments)
  {
    return ContextVar{ input(output.index() - GetOperation().Type()->NumArguments()),
                       const_cast<rvsdg::output *>(&output) };
  }
  else
  {
    return std::nullopt;
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
LambdaNode::AddContextVar(jlm::rvsdg::output & origin)
{
  auto input = rvsdg::StructuralInput::create(this, &origin, origin.Type());
  auto argument = &rvsdg::RegionArgument::Create(*subregion(), input, origin.Type());
  return ContextVar{ input, argument };
}

LambdaNode *
LambdaNode::Create(rvsdg::Region & parent, std::unique_ptr<LambdaOperation> operation)
{
  return new LambdaNode(parent, std::move(operation));
}

rvsdg::output *
LambdaNode::finalize(const std::vector<jlm::rvsdg::output *> & results)
{
  /* check if finalized was already called */
  if (noutputs() > 0)
  {
    JLM_ASSERT(noutputs() == 1);
    return output();
  }

  if (GetOperation().type().NumResults() != results.size())
    throw util::error("Incorrect number of results.");

  for (size_t n = 0; n < results.size(); n++)
  {
    auto & expected = GetOperation().type().ResultType(n);
    auto & received = results[n]->type();
    if (results[n]->type() != GetOperation().type().ResultType(n))
      throw util::error("Expected " + expected.debug_string() + ", got " + received.debug_string());

    if (results[n]->region() != subregion())
      throw util::error("Invalid operand region.");
  }

  for (const auto & origin : results)
    rvsdg::RegionResult::Create(*origin->region(), *origin, nullptr, origin->Type());

  return append_output(std::make_unique<rvsdg::StructuralOutput>(this, GetOperation().Type()));
}

rvsdg::output *
LambdaNode::output() const noexcept
{
  return StructuralNode::output(0);
}

LambdaNode *
LambdaNode::copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::output *> & operands) const
{
  return util::AssertedCast<LambdaNode>(rvsdg::Node::copy(region, operands));
}

LambdaNode *
LambdaNode::copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const
{
  const auto & op = GetOperation();
  auto lambda = Create(
      *region,
      std::unique_ptr<LambdaOperation>(util::AssertedCast<LambdaOperation>(op.copy().release())));

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
  std::vector<jlm::rvsdg::output *> results;
  for (auto result : GetFunctionResults())
    results.push_back(subregionmap.lookup(result->origin()));

  /* finalize lambda */
  auto o = lambda->finalize(results);
  smap.insert(output(), o);

  return lambda;
}

}
