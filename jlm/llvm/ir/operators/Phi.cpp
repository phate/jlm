/*
 * Copyright 2012 2013 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * Copyright 2012 2013 2014 Helge Bahmann <hcb@chaoticmind.net>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators/Phi.hpp>
#include <jlm/rvsdg/substitution.hpp>

namespace jlm {
namespace phi {

/* phi operation class */

operation::~operation()
{}

std::string
operation::debug_string() const
{
  return "PHI";
}

std::unique_ptr<jive::operation>
operation::copy() const
{
  return std::unique_ptr<jive::operation>(new phi::operation(*this));
}

/* phi node class */

node::~node()
{}

cvinput *
node::input(size_t n) const noexcept
{
  return static_cast<cvinput*>(structural_node::input(n));
}

rvoutput *
node::output(size_t n) const noexcept
{
  return static_cast<rvoutput*>(structural_node::output(n));
}

cvargument *
node::add_ctxvar(jive::output * origin)
{
  auto input = cvinput::create(this, origin, origin->type());
  return cvargument::create(subregion(), input, origin->type());
}

phi::node *
node::copy(
  jive::region * region,
  jive::substitution_map & smap) const
{
  phi::builder pb;
  pb.begin(region);

  /* add context variables */
  jive::substitution_map subregionmap;
  for (auto it = begin_cv(); it != end_cv(); it++) {
    auto origin = smap.lookup(it->origin());
    if (!origin) throw jive::compiler_error("Operand not provided by susbtitution map.");

    auto newcv = pb.add_ctxvar(origin);
    subregionmap.insert(it->argument(), newcv);
  }

  /* add recursion variables */
  std::vector<rvoutput*> newrvs;
  for (auto it = begin_rv(); it != end_rv(); it++) {
    auto newrv = pb.add_recvar(it->type());
    subregionmap.insert(it->argument(), newrv->argument());
    newrvs.push_back(newrv);
  }

  /* copy subregion */
  subregion()->copy(pb.subregion(), subregionmap, false, false);

  /* finalize phi */
  for (auto it = begin_rv(); it != end_rv(); it++) {
    auto neworigin = subregionmap.lookup(it->result()->origin());
    newrvs[it->index()]->set_rvorigin(neworigin);
  }

  return pb.end();
}

/* phi builder class */

rvoutput *
builder::add_recvar(const jive::type & type)
{
  if (!node_)
    return nullptr;

  auto argument = rvargument::create(subregion(), type);
  auto output = rvoutput::create(node_, argument, type);
  rvresult::create(subregion(), argument, output, type);
  argument->output_ = output;

  return output;
}

phi::node *
builder::end()
{
  if (!node_)
    return nullptr;

  for (auto it = node_->begin_rv(); it != node_->end_rv(); it++) {
    if (it->result()->origin() == it->argument())
      throw jive::compiler_error("Recursion variable not properly set.");
  }

  auto node = node_;
  node_ = nullptr;

  return node;
}

/* phi context variable input class */

cvinput::~cvinput()
{}

/* phi recursion variable output class */

rvoutput::~rvoutput()
{}

/* phi recursion variable output class */

rvargument::~rvargument()
{}

/* phi context variable argument class */

cvargument::~cvargument()
{}

/* phi recursion variable result class */

rvresult::~rvresult()
{}

}}

