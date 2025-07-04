/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-operation.hpp"

namespace jlm::tests
{

GraphImport &
GraphImport::Copy(rvsdg::Region & region, rvsdg::StructuralInput *)
{
  return GraphImport::Create(*region.graph(), Type(), Name());
}

GraphExport &
GraphExport::Copy(rvsdg::Output & origin, rvsdg::StructuralOutput * output)
{
  JLM_ASSERT(output == nullptr);
  return GraphExport::Create(origin, Name());
}

TestUnaryOperation::~TestUnaryOperation() noexcept = default;

bool
TestUnaryOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const TestUnaryOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

rvsdg::unop_reduction_path_t
TestUnaryOperation::can_reduce_operand(const rvsdg::Output *) const noexcept
{
  return rvsdg::unop_reduction_none;
}

rvsdg::Output *
TestUnaryOperation::reduce_operand(rvsdg::unop_reduction_path_t, rvsdg::Output *) const
{
  return nullptr;
}

std::string
TestUnaryOperation::debug_string() const
{
  return "TestUnaryOperation";
}

std::unique_ptr<rvsdg::Operation>
TestUnaryOperation::copy() const
{
  return std::make_unique<TestUnaryOperation>(*this);
}

TestBinaryOperation::~TestBinaryOperation() noexcept = default;

bool
TestBinaryOperation::operator==(const Operation & other) const noexcept
{
  auto op = dynamic_cast<const TestBinaryOperation *>(&other);
  return op && op->argument(0) == argument(0) && op->result(0) == result(0);
}

rvsdg::binop_reduction_path_t
TestBinaryOperation::can_reduce_operand_pair(const rvsdg::Output *, const rvsdg::Output *)
    const noexcept
{
  return rvsdg::binop_reduction_none;
}

rvsdg::Output *
TestBinaryOperation::reduce_operand_pair(
    rvsdg::binop_reduction_path_t,
    rvsdg::Output *,
    rvsdg::Output *) const
{
  return nullptr;
}

enum rvsdg::BinaryOperation::flags
TestBinaryOperation::flags() const noexcept
{
  return flags_;
}

std::string
TestBinaryOperation::debug_string() const
{
  return "TestBinaryOperation";
}

std::unique_ptr<rvsdg::Operation>
TestBinaryOperation::copy() const
{
  return std::make_unique<TestBinaryOperation>(*this);
}

TestOperation::~TestOperation() noexcept = default;

bool
TestOperation::operator==(const Operation & o) const noexcept
{
  auto other = dynamic_cast<const TestOperation *>(&o);
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
TestOperation::debug_string() const
{
  return "TestOperation";
}

std::unique_ptr<rvsdg::Operation>
TestOperation::copy() const
{
  return std::make_unique<TestOperation>(*this);
}

TestStructuralOperation::~TestStructuralOperation() noexcept = default;

std::string
TestStructuralOperation::debug_string() const
{
  return "TestStructuralOperation";
}

std::unique_ptr<rvsdg::Operation>
TestStructuralOperation::copy() const
{
  return std::make_unique<TestStructuralOperation>(*this);
}

TestStructuralNode::~TestStructuralNode() noexcept = default;

[[nodiscard]] const TestStructuralOperation &
TestStructuralNode::GetOperation() const noexcept
{
  static TestStructuralOperation singleton;
  return singleton;
}

TestStructuralNode *
TestStructuralNode::copy(rvsdg::Region * parent, rvsdg::SubstitutionMap & smap) const
{
  auto node = create(parent, nsubregions());

  /* copy inputs */
  for (size_t n = 0; n < ninputs(); n++)
  {
    auto origin = smap.lookup(input(n)->origin());
    auto neworigin = origin ? origin : input(n)->origin();
    auto new_input = rvsdg::StructuralInput::create(node, neworigin, input(n)->Type());
    smap.insert(input(n), new_input);
  }

  /* copy outputs */
  for (size_t n = 0; n < noutputs(); n++)
  {
    auto new_output = rvsdg::StructuralOutput::create(node, output(n)->Type());
    smap.insert(output(n), new_output);
  }

  /* copy regions */
  for (size_t n = 0; n < nsubregions(); n++)
    subregion(n)->copy(node->subregion(n), smap, true, true);

  return node;
}

StructuralNodeInput &
TestStructuralNode::AddInput(rvsdg::Output & origin)
{
  auto input =
      std::unique_ptr<StructuralNodeInput>(new StructuralNodeInput(*this, origin, origin.Type()));
  return *util::AssertedCast<StructuralNodeInput>(add_input(std::move(input)));
}

StructuralNodeInput &
TestStructuralNode::AddInputWithArguments(rvsdg::Output & origin)
{
  auto & input = AddInput(origin);
  for (size_t n = 0; n < nsubregions(); n++)
  {
    StructuralNodeArgument::Create(*subregion(n), input);
  }

  return input;
}

StructuralNodeOutput &
TestStructuralNode::AddOutput(std::shared_ptr<const rvsdg::Type> type)
{
  auto output =
      std::unique_ptr<StructuralNodeOutput>(new StructuralNodeOutput(*this, std::move(type)));
  return *util::AssertedCast<StructuralNodeOutput>(add_output(std::move(output)));
}

StructuralNodeOutput &
TestStructuralNode::AddOutputWithResults(const std::vector<rvsdg::Output *> & origins)
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
StructuralNodeArgument::Copy(rvsdg::Region & region, rvsdg::StructuralInput * input)
{
  auto structuralNodeInput = util::AssertedCast<StructuralNodeInput>(input);
  return structuralNodeInput != nullptr ? Create(region, *structuralNodeInput)
                                        : Create(region, Type());
}

StructuralNodeResult::~StructuralNodeResult() noexcept = default;

StructuralNodeResult &
StructuralNodeResult::Copy(rvsdg::Output & origin, rvsdg::StructuralOutput * output)
{
  auto structuralNodeOutput = util::AssertedCast<StructuralNodeOutput>(output);
  return structuralNodeOutput != nullptr ? Create(origin, *structuralNodeOutput) : Create(origin);
}

}
