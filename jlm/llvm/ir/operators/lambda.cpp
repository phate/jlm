/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::llvm::lambda
{

operation::~operation() = default;

operation::operation(
    std::shared_ptr<const jlm::rvsdg::FunctionType> type,
    std::string name,
    const jlm::llvm::linkage & linkage,
    jlm::llvm::attributeset attributes)
    : type_(std::move(type)),
      name_(std::move(name)),
      linkage_(linkage),
      attributes_(std::move(attributes))
{
  ArgumentAttributes_.resize(Type()->NumArguments());
}

std::string
operation::debug_string() const
{
  return util::strfmt("LAMBDA[", name(), "]");
}

bool
operation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const lambda::operation *>(&other);
  return op && op->type() == type() && op->name() == name() && op->linkage() == linkage()
      && op->attributes() == attributes();
}

std::unique_ptr<rvsdg::Operation>
operation::copy() const
{
  return std::make_unique<lambda::operation>(*this);
}

[[nodiscard]] const jlm::llvm::attributeset &
operation::GetArgumentAttributes(std::size_t index) const noexcept
{
  JLM_ASSERT(index < ArgumentAttributes_.size());
  return ArgumentAttributes_[index];
}

void
operation::SetArgumentAttributes(std::size_t index, const jlm::llvm::attributeset & attributes)
{
  JLM_ASSERT(index < ArgumentAttributes_.size());
  ArgumentAttributes_[index] = attributes;
}

/* lambda node class */

node::~node() = default;

node::node(rvsdg::Region & parent, std::unique_ptr<lambda::operation> op)
    : StructuralNode(&parent, 1),
      Operation_(std::move(op))
{}

lambda::operation &
node::GetOperation() const noexcept
{
  return *Operation_;
}

[[nodiscard]] std::vector<rvsdg::output *>
node::GetFunctionArguments() const
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
node::GetFunctionResults() const
{
  std::vector<rvsdg::input *> results;
  for (std::size_t n = 0; n < subregion()->nresults(); ++n)
  {
    results.push_back(subregion()->result(n));
  }
  return results;
}

[[nodiscard]] node::ContextVar
node::MapInputContextVar(const rvsdg::input & input) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<node>(input) == this);
  return ContextVar{ const_cast<rvsdg::input *>(&input),
                     subregion()->argument(GetOperation().Type()->NumArguments() + input.index()) };
}

[[nodiscard]] std::optional<node::ContextVar>
node::MapBinderContextVar(const rvsdg::output & output) const noexcept
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

[[nodiscard]] std::vector<node::ContextVar>
node::GetContextVars() const noexcept
{
  std::vector<ContextVar> vars;
  for (size_t n = 0; n < ninputs(); ++n)
  {
    vars.push_back(
        ContextVar{ input(n), subregion()->argument(n + GetOperation().Type()->NumArguments()) });
  }
  return vars;
}

node::ContextVar
node::AddContextVar(jlm::rvsdg::output & origin)
{
  auto input = rvsdg::StructuralInput::create(this, &origin, origin.Type());
  auto argument = &rvsdg::RegionArgument::Create(*subregion(), input, origin.Type());
  return ContextVar{ input, argument };
}

lambda::node *
node::create(
    rvsdg::Region * parent,
    std::shared_ptr<const jlm::rvsdg::FunctionType> type,
    const std::string & name,
    const llvm::linkage & linkage,
    const attributeset & attributes)
{
  auto op = std::make_unique<lambda::operation>(type, name, linkage, attributes);
  auto node = new lambda::node(*parent, std::move(op));

  for (auto & argumentType : type->Arguments())
    rvsdg::RegionArgument::Create(*node->subregion(), nullptr, argumentType);

  return node;
}

rvsdg::output *
node::finalize(const std::vector<jlm::rvsdg::output *> & results)
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
node::output() const noexcept
{
  return StructuralNode::output(0);
}

lambda::node *
node::copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::output *> & operands) const
{
  return util::AssertedCast<lambda::node>(rvsdg::Node::copy(region, operands));
}

lambda::node *
node::copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const
{
  const auto & op = GetOperation();
  auto lambda = create(region, op.Type(), op.name(), op.linkage(), op.attributes());

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
