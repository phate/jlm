/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_BACKEND_RVSDG2JLM_CONTEXT_HPP
#define JLM_LLVM_BACKEND_RVSDG2JLM_CONTEXT_HPP

#include <jlm/llvm/ir/basic-block.hpp>
#include <jlm/rvsdg/node.hpp>

#include <unordered_map>

namespace jlm::llvm
{

class cfg_node;
class ipgraph_module;
class variable;

namespace rvsdg2jlm
{

class context final
{
public:
  inline context(ipgraph_module & im)
      : cfg_(nullptr),
        module_(im),
        lpbb_(nullptr)
  {}

  context(const context &) = delete;

  context(context &&) = delete;

  context &
  operator=(const context &) = delete;

  context &
  operator=(context &&) = delete;

  inline ipgraph_module &
  module() const noexcept
  {
    return module_;
  }

  inline void
  insert(const rvsdg::output * output, const llvm::variable * v)
  {
    JLM_ASSERT(ports_.find(output) == ports_.end());
    JLM_ASSERT(*output->Type() == *v->Type());
    ports_[output] = v;
  }

  inline const llvm::variable *
  variable(const rvsdg::output * port)
  {
    auto it = ports_.find(port);
    JLM_ASSERT(it != ports_.end());
    return it->second;
  }

  inline basic_block *
  lpbb() const noexcept
  {
    return lpbb_;
  }

  inline void
  set_lpbb(basic_block * lpbb) noexcept
  {
    lpbb_ = lpbb;
  }

  inline llvm::cfg *
  cfg() const noexcept
  {
    return cfg_;
  }

  inline void
  set_cfg(llvm::cfg * cfg) noexcept
  {
    cfg_ = cfg;
  }

private:
  llvm::cfg * cfg_;
  ipgraph_module & module_;
  basic_block * lpbb_;
  std::unordered_map<const rvsdg::output *, const llvm::variable *> ports_;
};

}
}

#endif
