/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"

namespace jlm::tests
{

GraphImport &
GraphImport::Copy(rvsdg::Region & region, rvsdg::structural_input * input)
{
  return GraphImport::Create(*region.graph(), Type(), Name());
}

GraphExport &
GraphExport::Copy(rvsdg::output & origin, rvsdg::structural_output * output)
{
  JLM_ASSERT(output == nullptr);
  return GraphExport::Create(origin, Name());
}

/* unary operation */

unary_op::~unary_op() noexcept
{}

bool
unary_op::operator==(const rvsdg::operation & other) const noexcept
{
  auto op = dynamic_cast<const unary_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

rvsdg::unop_reduction_path_t
unary_op::can_reduce_operand(const rvsdg::output * operand) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::output *
unary_op::reduce_operand(rvsdg::unop_reduction_path_t path, rvsdg::output * operand) const
{
  return nullptr;
}

std::string
unary_op::debug_string() const
{
  return "UNARY_TEST_NODE";
}

std::unique_ptr<rvsdg::operation>
unary_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new unary_op(*this));
}

/* binary operation */

binary_op::~binary_op() noexcept
{}

bool
binary_op::operator==(const rvsdg::operation & other) const noexcept
{
  auto op = dynamic_cast<const binary_op *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

rvsdg::binop_reduction_path_t
binary_op::can_reduce_operand_pair(const rvsdg::output * op1, const rvsdg::output * op2)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::output *
binary_op::reduce_operand_pair(
    rvsdg::binop_reduction_path_t path,
    rvsdg::output * op1,
    rvsdg::output * op2) const
{
  return nullptr;
}

enum rvsdg::binary_op::flags
binary_op::flags() const noexcept
{
  return flags_;
}

std::string
binary_op::debug_string() const
{
  return "BINARY_TEST_OP";
}

std::unique_ptr<rvsdg::operation>
binary_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new binary_op(*this));
}

test_op::~test_op()
{}

bool
test_op::operator==(const operation & o) const noexcept
{
  auto other = dynamic_cast<const test_op *>(&o);
  if (!other)
    return false;

  if (narguments() != other->narguments() || nresults() != other->nresults())
    return false;

  for (size_t n = 0; n < narguments(); n++)
  {
    if (argument(n) != other->argument(n))
      return false;
  }

  for (size_t n = 0; n < nresults(); n++)
  {
    if (result(n) != other->result(n))
      return false;
  }

  return true;
}

std::string
test_op::debug_string() const
{
  return "test_op";
}

std::unique_ptr<rvsdg::operation>
test_op::copy() const
{
  return std::unique_ptr<operation>(new test_op(*this));
}

/* structural operation */

structural_op::~structural_op() noexcept
{}

std::string
structural_op::debug_string() const
{
  return "STRUCTURAL_TEST_NODE";
}

std::unique_ptr<rvsdg::operation>
structural_op::copy() const
{
  return std::unique_ptr<rvsdg::operation>(new structural_op(*this));
}

structural_node::~structural_node()
{}

structural_node *
structural_node::copy(rvsdg::Region * parent, rvsdg::SubstitutionMap & smap) const
{
  graph()->mark_denormalized();
  auto node = structural_node::create(parent, nsubregions());

  /* copy inputs */
  for (size_t n = 0; n < ninputs(); n++)
  {
    auto origin = smap.lookup(input(n)->origin());
    auto neworigin = origin ? origin : input(n)->origin();
    auto new_input = rvsdg::structural_input::create(node, neworigin, input(n)->Type());
    smap.insert(input(n), new_input);
  }

  /* copy outputs */
  for (size_t n = 0; n < noutputs(); n++)
  {
    auto new_output = rvsdg::structural_output::create(node, output(n)->Type());
    smap.insert(output(n), new_output);
  }

  /* copy regions */
  for (size_t n = 0; n < nsubregions(); n++)
    subregion(n)->copy(node->subregion(n), smap, true, true);

  return node;
}

StructuralNodeInput &
structural_node::AddInput(rvsdg::output & origin)
{
  auto input =
      std::unique_ptr<StructuralNodeInput>(new StructuralNodeInput(*this, origin, origin.Type()));
  return *util::AssertedCast<StructuralNodeInput>(add_input(std::move(input)));
}

StructuralNodeInput &
structural_node::AddInputWithArguments(rvsdg::output & origin)
{
  auto & input = AddInput(origin);
  for (size_t n = 0; n < nsubregions(); n++)
  {
    StructuralNodeArgument::Create(*subregion(n), input);
  }

  return input;
}

StructuralNodeOutput &
structural_node::AddOutput(std::shared_ptr<const rvsdg::Type> type)
{
  auto output =
      std::unique_ptr<StructuralNodeOutput>(new StructuralNodeOutput(*this, std::move(type)));
  return *util::AssertedCast<StructuralNodeOutput>(add_output(std::move(output)));
}

StructuralNodeOutput &
structural_node::AddOutputWithResults(const std::vector<rvsdg::output *> & origins)
{
  if (origins.size() != nsubregions())
    throw util::error("Insufficient number of origins.");

  auto & output = AddOutput(origins[0]->Type());
  for (size_t n = 0; n < nsubregions(); n++)
  {
    StructuralNodeResult::Create(*origins[n], output);
  }

  return output;
}

StructuralNodeInput::~StructuralNodeInput() noexcept = default;

StructuralNodeOutput::~StructuralNodeOutput() noexcept = default;

StructuralNodeArgument::~StructuralNodeArgument() noexcept = default;

StructuralNodeArgument &
StructuralNodeArgument::Copy(rvsdg::Region & region, rvsdg::structural_input * input)
{
  auto structuralNodeInput = util::AssertedCast<StructuralNodeInput>(input);
  return structuralNodeInput != nullptr ? Create(region, *structuralNodeInput)
                                        : Create(region, Type());
}

StructuralNodeResult::~StructuralNodeResult() noexcept = default;

StructuralNodeResult &
StructuralNodeResult::Copy(rvsdg::output & origin, rvsdg::structural_output * output)
{
  auto structuralNodeOutput = util::AssertedCast<StructuralNodeOutput>(output);
  return structuralNodeOutput != nullptr ? Create(origin, *structuralNodeOutput) : Create(origin);
}

}
