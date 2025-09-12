/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/delta.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/util/strfmt.hpp>

namespace jlm::rvsdg
{

DeltaOperation::~DeltaOperation() noexcept = default;

DeltaNode::ContextVar
DeltaNode::AddContextVar(jlm::rvsdg::Output & origin)
{
  auto input = rvsdg::StructuralInput::create(this, &origin, origin.Type());
  auto argument = &rvsdg::RegionArgument::Create(*subregion(), input, origin.Type());
  return ContextVar{ input, argument };
}

[[nodiscard]] DeltaNode::ContextVar
DeltaNode::MapInputContextVar(const rvsdg::Input & input) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetOwnerNode<DeltaNode>(input) == this);
  return ContextVar{ const_cast<rvsdg::Input *>(&input), subregion()->argument(input.index()) };
}

[[nodiscard]] DeltaNode::ContextVar
DeltaNode::MapBinderContextVar(const rvsdg::Output & output) const noexcept
{
  JLM_ASSERT(rvsdg::TryGetOwnerRegion(output) == subregion());
  return ContextVar{ input(output.index()), const_cast<rvsdg::Output *>(&output) };
}

std::vector<DeltaNode::ContextVar>
DeltaNode::GetContextVars() const noexcept
{
  std::vector<ContextVar> vars;
  for (size_t n = 0; n < ninputs(); ++n)
  {
    vars.push_back(ContextVar{ input(n), subregion()->argument(n) });
  }
  return vars;
}

std::string
DeltaOperation::debug_string() const
{
  return util::strfmt("DELTA");
}

std::unique_ptr<rvsdg::Operation>
DeltaOperation::copy() const
{
  return std::make_unique<DeltaOperation>(*this);
}

bool
DeltaOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const DeltaOperation *>(&other);
  return op && op->constant_ == constant_ && *op->type_ == *type_;
}

DeltaNode::~DeltaNode() noexcept = default;

const DeltaOperation &
DeltaNode::GetOperation() const noexcept
{
  return *Operation_;
}

DeltaNode *
DeltaNode::copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::Output *> & operands) const
{
  return static_cast<DeltaNode *>(rvsdg::Node::copy(region, operands));
}

DeltaNode *
DeltaNode::copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const
{
  auto delta = Create(
      region,
      std::unique_ptr<DeltaOperation>(static_cast<DeltaOperation *>(Operation_->copy().release())));

  // add context variables
  rvsdg::SubstitutionMap subregionmap;
  for (auto & cv : GetContextVars())
  {
    auto origin = smap.lookup(cv.input->origin());
    auto newCtxVar = delta->AddContextVar(*origin);
    subregionmap.insert(cv.inner, newCtxVar.inner);
  }

  // copy subregion
  subregion()->copy(delta->subregion(), subregionmap, false, false);

  // finalize delta
  auto result = subregionmap.lookup(delta->result().origin());
  auto o = &delta->finalize(result);
  smap.insert(&output(), o);

  return delta;
}

rvsdg::Output &
DeltaNode::output() const noexcept
{
  return *StructuralNode::output(0);
}

rvsdg::Input &
DeltaNode::result() const noexcept
{
  return *subregion()->result(0);
}

rvsdg::Output &
DeltaNode::finalize(jlm::rvsdg::Output * origin)
{
  // check if finalized was already called
  if (noutputs() > 0)
  {
    JLM_ASSERT(noutputs() == 1);
    return output();
  }

  auto & expected = Type();
  auto & received = *origin->Type();
  if (*expected != received)
    throw util::Error("Expected " + expected->debug_string() + ", got " + received.debug_string());

  if (origin->region() != subregion())
    throw util::Error("Invalid operand region.");

  rvsdg::RegionResult::Create(*origin->region(), *origin, nullptr, origin->Type());

  return *append_output(
      std::make_unique<rvsdg::StructuralOutput>(this, Operation_->ReferenceType()));
}

}
