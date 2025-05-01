/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/rvsdg/substitution.hpp>

namespace jlm::llvm
{

namespace delta
{

/* delta operator */

operation::~operation()
{}

std::string
operation::debug_string() const
{
  return util::strfmt("DELTA[", name(), "]");
}

std::unique_ptr<rvsdg::Operation>
operation::copy() const
{
  return std::make_unique<delta::operation>(*this);
}

bool
operation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const delta::operation *>(&other);
  return op && op->name_ == name_ && op->linkage_ == linkage_ && op->constant_ == constant_
      && op->Section_ == Section_ && *op->type_ == *type_;
}

/* delta node */

node::~node()
{}

const delta::operation &
node::GetOperation() const noexcept
{
  return *Operation_;
}

delta::node *
node::copy(rvsdg::Region * region, const std::vector<jlm::rvsdg::output *> & operands) const
{
  return static_cast<delta::node *>(rvsdg::Node::copy(region, operands));
}

delta::node *
node::copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const
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

node::ctxvar_range
node::ctxvars()
{
  cviterator end(nullptr);

  if (ncvarguments() == 0)
    return ctxvar_range(end, end);

  cviterator begin(input(0));
  return ctxvar_range(begin, end);
}

node::ctxvar_constrange
node::ctxvars() const
{
  cvconstiterator end(nullptr);

  if (ncvarguments() == 0)
    return ctxvar_constrange(end, end);

  cvconstiterator begin(input(0));
  return ctxvar_constrange(begin, end);
}

cvargument *
node::add_ctxvar(jlm::rvsdg::output * origin)
{
  auto input = cvinput::create(this, origin);
  return cvargument::create(subregion(), input);
}

cvinput *
node::input(size_t n) const noexcept
{
  return static_cast<cvinput *>(StructuralNode::input(n));
}

cvargument *
node::cvargument(size_t n) const noexcept
{
  return util::AssertedCast<delta::cvargument>(subregion()->argument(n));
}

delta::output *
node::output() const noexcept
{
  return static_cast<delta::output *>(StructuralNode::output(0));
}

delta::result *
node::result() const noexcept
{
  return static_cast<delta::result *>(subregion()->result(0));
}

delta::output *
node::finalize(jlm::rvsdg::output * origin)
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

  return output::create(this, PointerType::Create());
}

/* delta context variable input class */

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
result::Copy(rvsdg::output & origin, rvsdg::StructuralOutput * output)
{
  JLM_ASSERT(output == nullptr);
  return *result::create(&origin);
}

}
}
