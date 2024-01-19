/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RHLS2FIRRTL_BASE_HLS_HPP
#define JLM_HLS_BACKEND_RHLS2FIRRTL_BASE_HLS_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/hls/ir/hls.hpp>

#include <fstream>

namespace jlm::hls
{

bool
isForbiddenChar(char c);

class BaseHLS
{
public:
  std::string
  run(llvm::RvsdgModule & rm)
  {
    assert(node_map.empty());
    // ensure consistent naming across runs
    create_node_names(get_hls_lambda(rm)->subregion());
    return get_text(rm);
  }

private:
  virtual std::string
  extension() = 0;

protected:
  std::unordered_map<const jlm::rvsdg::node *, std::string> node_map;
  std::unordered_map<jlm::rvsdg::output *, std::string> output_map;

  std::string
  get_node_name(const jlm::rvsdg::node * node);

  static std::string
  get_port_name(jlm::rvsdg::input * port);

  static std::string
  get_port_name(jlm::rvsdg::output * port);

  const llvm::lambda::node *
  get_hls_lambda(llvm::RvsdgModule & rm);

  int
  JlmSize(const jlm::rvsdg::type * type);

  void
  create_node_names(jlm::rvsdg::region * r);

  virtual std::string
  get_text(llvm::RvsdgModule & rm) = 0;

  static std::string
  get_base_file_name(const llvm::RvsdgModule & rm);

  std::vector<jlm::rvsdg::argument *>
  get_mem_resps(const llvm::lambda::node * lambda)
  {
    std::vector<jlm::rvsdg::argument *> mem_resps;
    for (size_t i = 0; i < lambda->subregion()->narguments(); ++i)
    {
      auto arg = lambda->subregion()->argument(i);
      if (dynamic_cast<const jlm::hls::bundletype *>(&arg->type()))
      {
        mem_resps.push_back(lambda->subregion()->argument(i));
      }
    }
    return mem_resps;
  }

  std::vector<jlm::rvsdg::result *>
  get_mem_reqs(const llvm::lambda::node * lambda)
  {
    std::vector<jlm::rvsdg::result *> mem_resps;
    for (size_t i = 0; i < lambda->subregion()->nresults(); ++i)
    {
      if (dynamic_cast<const jlm::hls::bundletype *>(&lambda->subregion()->result(i)->type()))
      {
        mem_resps.push_back(lambda->subregion()->result(i));
      }
    }
    return mem_resps;
  }

  std::vector<jlm::rvsdg::argument *>
  get_reg_args(const llvm::lambda::node * lambda)
  {
    std::vector<jlm::rvsdg::argument *> args;
    for (size_t i = 0; i < lambda->subregion()->narguments(); ++i)
    {
      auto argtype = &lambda->subregion()->argument(i)->type();
      if (!dynamic_cast<const jlm::hls::bundletype *>(
              argtype) /*&& !dynamic_cast<const jlm::rvsdg::statetype *>(argtype)*/)
      {
        args.push_back(lambda->subregion()->argument(i));
      }
    }
    return args;
  }

  std::vector<jlm::rvsdg::result *>
  get_reg_results(const llvm::lambda::node * lambda)
  {
    std::vector<jlm::rvsdg::result *> results;
    for (size_t i = 0; i < lambda->subregion()->nresults(); ++i)
    {
      auto argtype = &lambda->subregion()->result(i)->type();
      if (!dynamic_cast<const jlm::hls::bundletype *>(
              argtype) /*&& !dynamic_cast<const jlm::rvsdg::statetype *>(argtype)*/)
      {
        results.push_back(lambda->subregion()->result(i));
      }
    }
    return results;
  }
};

}

#endif // JLM_HLS_BACKEND_RHLS2FIRRTL_BASE_HLS_HPP
