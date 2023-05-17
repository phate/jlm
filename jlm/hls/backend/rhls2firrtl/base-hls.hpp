/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RHLS2FIRRTL_BASE_HLS_HPP
#define JLM_HLS_BACKEND_RHLS2FIRRTL_BASE_HLS_HPP

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>

#include <fstream>

namespace jlm::hls
{

bool
isForbiddenChar(char c);

class BaseHLS {
public:
  std::string
  run(llvm::RvsdgModule &rm) {
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
  get_node_name(const jlm::rvsdg::node *node);

  static std::string
  get_port_name(jlm::rvsdg::input *port);

  static std::string
  get_port_name(jlm::rvsdg::output *port);

  const llvm::lambda::node *
  get_hls_lambda(llvm::RvsdgModule &rm);

  int
  JlmSize(const jlm::rvsdg::type *type);

  void
  create_node_names(jlm::rvsdg::region *r);

  virtual std::string
  get_text(llvm::RvsdgModule &rm) = 0;

  static std::string
  get_base_file_name(const llvm::RvsdgModule &rm);
};

}

#endif //JLM_HLS_BACKEND_RHLS2FIRRTL_BASE_HLS_HPP
