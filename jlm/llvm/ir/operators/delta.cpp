/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/rvsdg/substitution.hpp>

namespace jlm::llvm
{

DeltaOperation::~DeltaOperation() noexcept = default;

std::string
DeltaOperation::debug_string() const
{
  return util::strfmt("DELTA[", name(), "]");
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
  return op && op->name_ == name_ && op->linkage_ == linkage_ && op->constant_ == constant_
      && op->Section_ == Section_ && *op->type_ == *type_;
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
  auto delta = Create(region, Type(), name(), linkage(), Section(), constant());

  /* add context variables */
  rvsdg::SubstitutionMap subregionmap;
  for (auto & cv : ctxvars())
  {
    auto origin = smap.lookup(cv.origin());
    auto newcv = delta->add_ctxvar(origin);
    subregionmap.insert(cv.argument(), newcv);
  }

  /* copy subregion */
  subregion()->copy(delta->subregion(), subregionmap, false, false);

  /* finalize delta */
  auto result = subregionmap.lookup(delta->result()->origin());
  auto o = delta->finalize(result);
  smap.insert(output(), o);

  return delta;
}

DeltaNode::ctxvar_range
DeltaNode::ctxvars()
{
  cviterator end(nullptr);

  if (ncvarguments() == 0)
    return ctxvar_range(end, end);

  cviterator begin(input(0));
  return ctxvar_range(begin, end);
}

DeltaNode::ctxvar_constrange
DeltaNode::ctxvars() const
{
  cvconstiterator end(nullptr);

  if (ncvarguments() == 0)
    return ctxvar_constrange(end, end);

  cvconstiterator begin(input(0));
  return ctxvar_constrange(begin, end);
}

delta::cvargument *
DeltaNode::add_ctxvar(jlm::rvsdg::Output * origin)
{
  auto input = delta::cvinput::create(this, origin);
  return delta::cvargument::create(subregion(), input);
}

delta::cvinput *
DeltaNode::input(size_t n) const noexcept
{
  return static_cast<delta::cvinput *>(StructuralNode::input(n));
}

delta::cvargument *
DeltaNode::cvargument(size_t n) const noexcept
{
  return util::AssertedCast<delta::cvargument>(subregion()->argument(n));
}

delta::output *
DeltaNode::output() const noexcept
{
  return static_cast<delta::output *>(StructuralNode::output(0));
}

delta::result *
DeltaNode::result() const noexcept
{
  return static_cast<delta::result *>(subregion()->result(0));
}

delta::output *
DeltaNode::finalize(jlm::rvsdg::Output * origin)
{
  /* check if finalized was already called */
  if (noutputs() > 0)
  {
    JLM_ASSERT(noutputs() == 1);
    return output();
  }

  auto & expected = type();
  auto & received = *origin->Type();
  if (expected != received)
    throw util::error("Expected " + expected.debug_string() + ", got " + received.debug_string());

  if (origin->region() != subregion())
    throw util::error("Invalid operand region.");

  delta::result::create(origin);

  return delta::output::create(this, PointerType::Create());
}

namespace delta
{

cvinput::~cvinput()
{}

cvargument *
cvinput::argument() const noexcept
{
  return static_cast<cvargument *>(arguments.first());
}

/* delta output class */

output::~output()
{}

/* delta context variable argument class */

cvargument::~cvargument()
{}

cvargument &
cvargument::Copy(rvsdg::Region & region, rvsdg::StructuralInput * input)
{
  auto deltaInput = util::AssertedCast<delta::cvinput>(input);
  return *cvargument::create(&region, deltaInput);
}

/* delta result class */

result::~result()
{}

result &
result::Copy(rvsdg::Output & origin, rvsdg::StructuralOutput * output)
{
  JLM_ASSERT(output == nullptr);
  return *result::create(&origin);
}

}
}
