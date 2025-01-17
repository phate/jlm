/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RHLS2FIRRTL_BASE_HLS_HPP
#define JLM_HLS_BACKEND_RHLS2FIRRTL_BASE_HLS_HPP

#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

#include <fstream>

namespace jlm::hls
{

bool
isForbiddenChar(char c);

class BaseHLS
{
public:
  virtual ~BaseHLS();

  std::string
  run(llvm::RvsdgModule & rm)
  {
    JLM_ASSERT(node_map.empty());
    // ensure consistent naming across runs
    create_node_names(get_hls_lambda(rm)->subregion());
    return GetText(rm);
  }

  static int
  JlmSize(const jlm::rvsdg::Type * type);

  /**
   * @return The size of a pointer in bits.
   */
  [[nodiscard]] static size_t
  GetPointerSizeInBits()
  {
    return 64;
  }

private:
  virtual std::string
  extension() = 0;

protected:
  std::unordered_map<const rvsdg::Node *, std::string> node_map;
  std::unordered_map<jlm::rvsdg::output *, std::string> output_map;

  std::string
  get_node_name(const rvsdg::Node * node);

  static std::string
  get_port_name(jlm::rvsdg::input * port);

  static std::string
  get_port_name(jlm::rvsdg::output * port);

  const llvm::lambda::node *
  get_hls_lambda(llvm::RvsdgModule & rm);

  void
  create_node_names(rvsdg::Region * r);

  virtual std::string
  GetText(llvm::RvsdgModule & rm) = 0;

  static std::string
  get_base_file_name(const llvm::RvsdgModule & rm);

  /**
   * Extracts all region arguments of the given kernel that represent memory responses.
   * They can provide multiple values within a single execution of the region.
   * @param lambda the lambda node holding the hls kernel
   * @return the arguments that represent memory responses
   */
  std::vector<rvsdg::RegionArgument *>
  get_mem_resps(const llvm::lambda::node & lambda)
  {
    std::vector<rvsdg::RegionArgument *> mem_resps;
    for (auto arg : lambda.subregion()->Arguments())
    {
      if (rvsdg::is<bundletype>(arg->type()))
        mem_resps.push_back(arg);
    }
    return mem_resps;
  }

  /**
   * Extracts all region results of the given kernel that represent memory requests.
   * They can take multiple values within a single execution of the region.
   * @param lambda the lambda node holding the hls kernel
   * @return the results that represent memory requests
   */
  std::vector<rvsdg::RegionResult *>
  get_mem_reqs(const llvm::lambda::node & lambda)
  {
    std::vector<rvsdg::RegionResult *> mem_resps;
    for (auto result : lambda.subregion()->Results())
    {
      if (rvsdg::is<bundletype>(result->type()))
        mem_resps.push_back(result);
    }
    return mem_resps;
  }

  /**
   * Extracts all region arguments of the given kernel that represent kernel inputs,
   * which may include kernel arguments, state types, and context variables (always in that order).
   * It will not return any arguments that represent memory responses.
   * @param lambda the lambda node holding the hls kernel
   * @return the arguments of the lambda that represent kernel inputs
   */
  std::vector<rvsdg::RegionArgument *>
  get_reg_args(const llvm::lambda::node & lambda)
  {
    std::vector<rvsdg::RegionArgument *> args;
    for (auto argument : lambda.subregion()->Arguments())
    {
      if (!rvsdg::is<bundletype>(argument->type()))
        args.push_back(argument);
    }
    return args;
  }

  /**
   * Extracts all region results from the given kernel that represent results from execution,
   * as opposed to results used for making memory requests.
   * @param lambda the lambda node holding the hls kernel
   * @return the results of the lambda that represent the kernel outputs
   */
  std::vector<rvsdg::RegionResult *>
  get_reg_results(const llvm::lambda::node & lambda)
  {
    std::vector<rvsdg::RegionResult *> results;
    for (auto result : lambda.subregion()->Results())
    {
      if (!rvsdg::is<bundletype>(result->type()))
        results.push_back(result);
    }
    return results;
  }
};

}

#endif // JLM_HLS_BACKEND_RHLS2FIRRTL_BASE_HLS_HPP
