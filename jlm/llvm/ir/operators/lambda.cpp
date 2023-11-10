/*
 * Copyright 2018 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

#include <deque>

namespace jlm::llvm::lambda
{

/* lambda operation class */

operation::~operation()
= default;

std::string
operation::debug_string() const
{
  return util::strfmt("LAMBDA[", name(), "]");
}

bool
operation::operator==(const jlm::rvsdg::operation & other) const noexcept
{
  auto op = dynamic_cast<const lambda::operation*>(&other);
  return op
         && op->type() == type()
         && op->name() == name()
         && op->linkage() == linkage()
         && op->attributes() == attributes();
}

std::unique_ptr<jlm::rvsdg::operation>
operation::copy() const
{
  return std::unique_ptr<jlm::rvsdg::operation>(new operation(*this));
}

/* lambda node class */

node::~node()
= default;

node::fctargument_range
node::fctarguments()
{
  fctargiterator end(nullptr);

  if (nfctarguments() == 0)
    return {end, end};

  fctargiterator begin(fctargument(0));
  return {begin, end};
}

node::fctargument_constrange
node::fctarguments() const
{
  fctargconstiterator end(nullptr);

  if (nfctarguments() == 0)
    return {end, end};

  fctargconstiterator begin(fctargument(0));
  return {begin, end};
}

node::ctxvar_range
node::ctxvars()
{
  cviterator end(nullptr);

  if (ncvarguments() == 0)
    return {end, end};

  cviterator begin(input(0));
  return {begin, end};
}

node::ctxvar_constrange
node::ctxvars() const
{
  cvconstiterator end(nullptr);

  if (ncvarguments() == 0)
    return {end, end};

  cvconstiterator begin(input(0));
  return {begin, end};
}

node::fctresult_range
node::fctresults()
{
  fctresiterator end(nullptr);

  if (nfctresults() == 0)
    return {end, end};

  fctresiterator begin(fctresult(0));
  return {begin, end};
}

node::fctresult_constrange
node::fctresults() const
{
  fctresconstiterator end(nullptr);

  if (nfctresults() == 0)
    return {end, end};

  fctresconstiterator begin(fctresult(0));
  return {begin, end};
}

cvinput *
node::input(size_t n) const noexcept
{
  return util::AssertedCast<cvinput>(structural_node::input(n));
}

lambda::output *
node::output() const noexcept
{
  return util::AssertedCast<lambda::output>(structural_node::output(0));
}

lambda::fctargument *
node::fctargument(size_t n) const noexcept
{
  return util::AssertedCast<lambda::fctargument>(subregion()->argument(n));
}

lambda::cvargument *
node::cvargument(size_t n) const noexcept
{
  return input(n)->argument();
}

lambda::result *
node::fctresult(size_t n) const noexcept
{
  return util::AssertedCast<lambda::result>(subregion()->result(n));
}

cvargument *
node::add_ctxvar(jlm::rvsdg::output * origin)
{
  auto input = cvinput::create(this, origin);
  return cvargument::create(subregion(), input);
}

lambda::node *
node::create(
  jlm::rvsdg::region * parent,
  const FunctionType & type,
  const std::string & name,
  const llvm::linkage & linkage,
  const attributeset & attributes)
{
  lambda::operation op(type, name, linkage, attributes);
  auto node = new lambda::node(parent, std::move(op));

  for (auto & argumentType : type.Arguments())
    lambda::fctargument::create(node->subregion(), argumentType);

  return node;
}

lambda::output *
node::finalize(const std::vector<jlm::rvsdg::output*> & results)
{
  /* check if finalized was already called */
  if (noutputs() > 0) {
    JLM_ASSERT(noutputs() == 1);
    return output();
  }

  if (type().NumResults() != results.size())
    throw util::error("Incorrect number of results.");

  for (size_t n = 0; n < results.size(); n++) {
    auto & expected = type().ResultType(n);
    auto & received = results[n]->type();
    if (results[n]->type() != type().ResultType(n))
      throw util::error("Expected " + expected.debug_string() + ", got " + received.debug_string());

    if (results[n]->region() != subregion())
      throw util::error("Invalid operand region.");
  }

  for (const auto & origin : results)
    lambda::result::create(origin);

  return output::create(this, PointerType());
}

lambda::node *
node::copy(
  jlm::rvsdg::region * region,
  const std::vector<jlm::rvsdg::output*> & operands) const
{
  return util::AssertedCast<lambda::node>(jlm::rvsdg::node::copy(region, operands));
}

lambda::node *
node::copy(jlm::rvsdg::region * region, jlm::rvsdg::substitution_map & smap) const
{
  auto lambda = create(region, type(), name(), linkage(), attributes());

  /* add context variables */
  jlm::rvsdg::substitution_map subregionmap;
  for (auto & cv : ctxvars()) {
    auto origin = smap.lookup(cv.origin());
    auto newcv = lambda->add_ctxvar(origin);
    subregionmap.insert(cv.argument(), newcv);
  }

  /* collect function arguments */
  for (size_t n = 0; n < nfctarguments(); n++){
    lambda->fctargument(n)->set_attributes(fctargument(n)->attributes());
    subregionmap.insert(fctargument(n), lambda->fctargument(n));
  }

  /* copy subregion */
  subregion()->copy(lambda->subregion(), subregionmap, false, false);

  /* collect function results */
  std::vector<jlm::rvsdg::output*> results;
  for (auto & result : fctresults())
    results.push_back(subregionmap.lookup(result.origin()));

  /* finalize lambda */
  auto o = lambda->finalize(results);
  smap.insert(output(), o);

  return lambda;
}

std::unique_ptr<node::CallSummary>
node::ComputeCallSummary() const
{
  std::deque<rvsdg::input*> worklist;
  worklist.insert(worklist.end(), output()->begin(), output()->end());

  std::vector<CallNode*> directCalls;
  rvsdg::result * rvsdgExport = nullptr;
  std::vector<rvsdg::input*> otherUsers;

  while (!worklist.empty()) {
    auto input = worklist.front();
    worklist.pop_front();

    if (auto cvinput = dynamic_cast<lambda::cvinput*>(input)) {
      auto argument = cvinput->argument();
      worklist.insert(worklist.end(), argument->begin(), argument->end());
      continue;
    }

    if (auto gamma_input = dynamic_cast<rvsdg::gamma_input*>(input)) {
      for (auto & argument : *gamma_input)
        worklist.insert(worklist.end(), argument.begin(), argument.end());
      continue;
    }

    if (auto result = is_gamma_result(input)) {
      auto output = result->output();
      worklist.insert(worklist.end(), output->begin(), output->end());
      continue;
    }

    if (auto theta_input = dynamic_cast<rvsdg::theta_input*>(input)) {
      auto argument = theta_input->argument();
      worklist.insert(worklist.end(), argument->begin(), argument->end());
      continue;
    }

    if (auto result = is_theta_result(input)) {
      auto output = result->output();
      worklist.insert(worklist.end(), output->begin(), output->end());
      continue;
    }

    if (auto cvinput = dynamic_cast<phi::cvinput*>(input)) {
      auto argument = cvinput->argument();
      worklist.insert(worklist.end(), argument->begin(), argument->end());
      continue;
    }

    if (auto rvresult = dynamic_cast<phi::rvresult*>(input)) {
      auto argument = rvresult->argument();
      worklist.insert(worklist.end(), argument->begin(), argument->end());

      auto output = rvresult->output();
      worklist.insert(worklist.end(), output->begin(), output->end());
      continue;
    }

    if (auto cvinput = dynamic_cast<delta::cvinput*>(input)) {
      auto argument = cvinput->arguments.first();
      worklist.insert(worklist.end(), argument->begin(), argument->end());
      continue;
    }

    if (auto deltaResult = dynamic_cast<delta::result*>(input))
    {
      otherUsers.emplace_back(deltaResult);
      continue;
    }

    auto inputNode = rvsdg::input::GetNode(*input);
    if (is<CallOperation>(inputNode) && input == inputNode->input(0)) {
      directCalls.emplace_back(util::AssertedCast<CallNode>(inputNode));
      continue;
    }

    auto result = dynamic_cast<rvsdg::result*>(input);
    if (result != nullptr
        && input->region() == graph()->root())
    {
      rvsdgExport = result;
      continue;
    }

    auto simpleInput = dynamic_cast<rvsdg::simple_input*>(input);
    if (simpleInput != nullptr)
    {
      otherUsers.emplace_back(simpleInput);
      continue;
    }

    JLM_UNREACHABLE("This should have never happened!");
  }

  return CallSummary::Create(
    rvsdgExport,
    std::move(directCalls),
    std::move(otherUsers));
}

/* lambda context variable input class */

cvinput::~cvinput()
= default;

cvargument *
cvinput::argument() const noexcept
{
  return util::AssertedCast<cvargument>(arguments.first());
}

/* lambda output class */

output::~output()
= default;

/* lambda function argument class */

fctargument::~fctargument()
= default;

/* lambda context variable argument class */

cvargument::~cvargument()
= default;

/* lambda result class */

result::~result()
= default;

}
