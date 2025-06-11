/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_FRONTEND_LLVMCONVERSIONCONTEXT_HPP
#define JLM_LLVM_FRONTEND_LLVMCONVERSIONCONTEXT_HPP

#include <jlm/llvm/ir/cfg-node.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/tac.hpp>
#include <jlm/llvm/ir/TypeConverter.hpp>

#include <llvm/IR/DerivedTypes.h>

#include <unordered_map>

namespace llvm
{
class BasicBlock;
class Function;
class Value;
}

namespace jlm::llvm
{

class ControlFlowGraph;
class cfg_node;
class clg_node;
class InterProceduralGraphModule;
class Variable;

class basic_block_map final
{
public:
  inline bool
  has(const ::llvm::BasicBlock * bb) const noexcept
  {
    return llvm2jlm_.find(bb) != llvm2jlm_.end();
  }

  inline bool
  has(const BasicBlock * bb) const noexcept
  {
    return jlm2llvm_.find(bb) != jlm2llvm_.end();
  }

  inline BasicBlock *
  get(const ::llvm::BasicBlock * bb) const noexcept
  {
    JLM_ASSERT(has(bb));
    return llvm2jlm_.find(bb)->second;
  }

  inline const ::llvm::BasicBlock *
  get(const BasicBlock * bb) const noexcept
  {
    JLM_ASSERT(has(bb));
    return jlm2llvm_.find(bb)->second;
  }

  inline void
  insert(const ::llvm::BasicBlock * bb1, BasicBlock * bb2)
  {
    JLM_ASSERT(!has(bb1));
    JLM_ASSERT(!has(bb2));
    llvm2jlm_[bb1] = bb2;
    jlm2llvm_[bb2] = bb1;
  }

  BasicBlock *
  operator[](const ::llvm::BasicBlock * bb) const
  {
    return get(bb);
  }

  const ::llvm::BasicBlock *
  operator[](const BasicBlock * bb) const
  {
    return get(bb);
  }

private:
  std::unordered_map<const ::llvm::BasicBlock *, BasicBlock *> llvm2jlm_;
  std::unordered_map<const BasicBlock *, const ::llvm::BasicBlock *> jlm2llvm_;
};

class context final
{
public:
  inline context(InterProceduralGraphModule & im)
      : module_(im),
        node_(nullptr),
        iostate_(nullptr),
        memory_state_(nullptr)
  {}

  const llvm::Variable *
  result() const noexcept
  {
    return result_;
  }

  inline void
  set_result(const llvm::Variable * result)
  {
    result_ = result;
  }

  llvm::Variable *
  iostate() const noexcept
  {
    return iostate_;
  }

  void
  set_iostate(llvm::Variable * state)
  {
    iostate_ = state;
  }

  inline llvm::Variable *
  memory_state() const noexcept
  {
    return memory_state_;
  }

  inline void
  set_memory_state(llvm::Variable * state)
  {
    memory_state_ = state;
  }

  inline bool
  has(const ::llvm::BasicBlock * bb) const noexcept
  {
    return bbmap_.has(bb);
  }

  inline bool
  has(const BasicBlock * bb) const noexcept
  {
    return bbmap_.has(bb);
  }

  inline BasicBlock *
  get(const ::llvm::BasicBlock * bb) const noexcept
  {
    return bbmap_.get(bb);
  }

  inline const ::llvm::BasicBlock *
  get(const BasicBlock * bb) const noexcept
  {
    return bbmap_.get(bb);
  }

  inline void
  set_basic_block_map(const basic_block_map & bbmap)
  {
    bbmap_ = bbmap;
  }

  inline bool
  has_value(const ::llvm::Value * value) const noexcept
  {
    return vmap_.find(value) != vmap_.end();
  }

  inline const llvm::Variable *
  lookup_value(const ::llvm::Value * value) const noexcept
  {
    JLM_ASSERT(has_value(value));
    return vmap_.find(value)->second;
  }

  inline void
  insert_value(const ::llvm::Value * value, const llvm::Variable * variable)
  {
    JLM_ASSERT(!has_value(value));
    vmap_[value] = variable;
  }

  [[nodiscard]] InterProceduralGraphModule &
  module() const noexcept
  {
    return module_;
  }

  inline void
  set_node(ipgraph_node * node) noexcept
  {
    node_ = node;
  }

  inline ipgraph_node *
  node() const noexcept
  {
    return node_;
  }

  TypeConverter &
  GetTypeConverter() noexcept
  {
    return TypeConverter_;
  }

private:
  InterProceduralGraphModule & module_;
  basic_block_map bbmap_;
  ipgraph_node * node_;
  const llvm::Variable * result_;
  llvm::Variable * iostate_;
  llvm::Variable * memory_state_;
  std::unordered_map<const ::llvm::Value *, const llvm::Variable *> vmap_;
  TypeConverter TypeConverter_;
};

}

#endif
