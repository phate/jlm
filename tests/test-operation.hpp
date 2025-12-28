/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef TESTS_TEST_OPERATION_HPP
#define TESTS_TEST_OPERATION_HPP

#include <jlm/rvsdg/node.hpp>
#include <jlm/rvsdg/operation.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/type.hpp>

namespace jlm::tests
{

class TestStructuralOperation final : public rvsdg::StructuralOperation
{
public:
  ~TestStructuralOperation() noexcept override;

  [[nodiscard]] std::string
  debug_string() const override;

  [[nodiscard]] std::unique_ptr<Operation>
  copy() const override;
};

class TestStructuralNode final : public rvsdg::StructuralNode
{
public:
  ~TestStructuralNode() noexcept override;

private:
  TestStructuralNode(rvsdg::Region * parent, size_t nsubregions)
      : StructuralNode(parent, nsubregions)
  {}

public:
  /**
   * \brief A variable routed in a \ref TestStructuralNode
   */
  struct InputVar
  {
    rvsdg::StructuralInput * input{};
    std::vector<rvsdg::RegionArgument *> argument{};
  };

  /**
   * \brief A variable routed out of a \ref TestStructuralNode
   */
  struct OutputVar
  {
    rvsdg::StructuralOutput * output{};
    std::vector<rvsdg::RegionResult *> result{};
  };

  /**
   * Add an input WITHOUT subregion arguments to a \ref TestStructuralNode.
   *
   * @param origin Value to be routed in.
   * @return The created input variable.
   */
  rvsdg::StructuralInput &
  addInputOnly(rvsdg::Output & origin);

  /**
   * Add an input WITH subregion arguments to a \ref TestStructuralNode.
   *
   * @param origin Value to be routed in.
   * @return Description of input variable.
   */
  InputVar
  addInputWithArguments(rvsdg::Output & origin);

  /**
   * Removes the input with the given \p index,
   * and any subregion arguments associated with it.
   * All such arguments must be dead.
   * @param index the index of the input to remove.
   */
  void
  removeInputAndArguments(size_t index);

  /**
   * Add subregion arguments WITHOUT an input to a \ref TestStructuralNode.
   * @param type The argument type
   * @return Description of input variable.
   */
  InputVar
  addArguments(const std::shared_ptr<const rvsdg::Type> & type);

  /**
   * Add an output WITHOUT subregion results to a \ref TestStructuralNode.
   *
   * @param type The output type
   * @return The created output variable.
   */
  rvsdg::StructuralOutput &
  addOutputOnly(std::shared_ptr<const rvsdg::Type> type);

  /**
   * Add an output WITH subregion results to a \ref TestStructuralNode.
   *
   * @param origins The values to be routed out.
   * @return Description of output variable.
   */
  OutputVar
  addOutputWithResults(const std::vector<rvsdg::Output *> & origins);

  /**
   * Removes the output with the given \p index,
   * and any subregion results associated with it.
   * @param index the index of the output to remove.
   */
  void
  removeOutputAndResults(size_t index);

  /**
   * Add subregion results WITHOUT output to a \ref TestStructuralNode.
   *
   * @param origins The values to be routed out.
   * @return Description of output variable.
   */
  OutputVar
  addResults(const std::vector<rvsdg::Output *> & origins);

  [[nodiscard]] const TestStructuralOperation &
  GetOperation() const noexcept override;

  static TestStructuralNode *
  create(rvsdg::Region * parent, size_t nsubregions)
  {
    return new TestStructuralNode(parent, nsubregions);
  }

  TestStructuralNode *
  copy(rvsdg::Region * region, rvsdg::SubstitutionMap & smap) const override;
};

}

#endif
