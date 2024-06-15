/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_FRONTEND_LLVMCONVERSIONCONTEXT_HPP
#define JLM_LLVM_FRONTEND_LLVMCONVERSIONCONTEXT_HPP

#include <jlm/llvm/frontend/LlvmTypeConversion.hpp>
#include <jlm/llvm/ir/cfg-node.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/tac.hpp>

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

class cfg;
class cfg_node;
class clg_node;
class ipgraph_module;
class variable;

class basic_block_map final
{
public:
  inline bool
  has(const ::llvm::BasicBlock * bb) const noexcept
  {
    return llvm2jlm_.find(bb) != llvm2jlm_.end();
  }

  inline bool
  has(const basic_block * bb) const noexcept
  {
    return jlm2llvm_.find(bb) != jlm2llvm_.end();
  }

  inline basic_block *
  get(const ::llvm::BasicBlock * bb) const noexcept
  {
    JLM_ASSERT(has(bb));
    return llvm2jlm_.find(bb)->second;
  }

  inline const ::llvm::BasicBlock *
  get(const basic_block * bb) const noexcept
  {
    JLM_ASSERT(has(bb));
    return jlm2llvm_.find(bb)->second;
  }

  inline void
  insert(const ::llvm::BasicBlock * bb1, basic_block * bb2)
  {
    JLM_ASSERT(!has(bb1));
    JLM_ASSERT(!has(bb2));
    llvm2jlm_[bb1] = bb2;
    jlm2llvm_[bb2] = bb1;
  }

  basic_block *
  operator[](const ::llvm::BasicBlock * bb) const
  {
    return get(bb);
  }

  const ::llvm::BasicBlock *
  operator[](const basic_block * bb) const
  {
    return get(bb);
  }

private:
  std::unordered_map<const ::llvm::BasicBlock *, basic_block *> llvm2jlm_;
  std::unordered_map<const basic_block *, const ::llvm::BasicBlock *> jlm2llvm_;
};

class context final
{
public:
  inline context(ipgraph_module & im)
      : module_(im),
        node_(nullptr),
        iostate_(nullptr),
        memory_state_(nullptr)
  {}

  const llvm::variable *
  result() const noexcept
  {
    return result_;
  }

  inline void
  set_result(const llvm::variable * result)
  {
    result_ = result;
  }

  llvm::variable *
  iostate() const noexcept
  {
    return iostate_;
  }

  void
  set_iostate(llvm::variable * state)
  {
    iostate_ = state;
  }

  inline llvm::variable *
  memory_state() const noexcept
  {
    return memory_state_;
  }

  inline void
  set_memory_state(llvm::variable * state)
  {
    memory_state_ = state;
  }

  inline bool
  has(const ::llvm::BasicBlock * bb) const noexcept
  {
    return bbmap_.has(bb);
  }

  inline bool
  has(const basic_block * bb) const noexcept
  {
    return bbmap_.has(bb);
  }

  inline basic_block *
  get(const ::llvm::BasicBlock * bb) const noexcept
  {
    return bbmap_.get(bb);
  }

  inline const ::llvm::BasicBlock *
  get(const basic_block * bb) const noexcept
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

  inline const llvm::variable *
  lookup_value(const ::llvm::Value * value) const noexcept
  {
    JLM_ASSERT(has_value(value));
    return vmap_.find(value)->second;
  }

  inline void
  insert_value(const ::llvm::Value * value, const llvm::variable * variable)
  {
    JLM_ASSERT(!has_value(value));
    vmap_[value] = variable;
  }

  const StructType::Declaration *
  lookup_declaration(const ::llvm::StructType * type)
  {
    // Return declaration if we already created one for this type instance
    if (auto it = declarations_.find(type); it != declarations_.end())
    {
      return it->second;
    }

    // Otherwise create a new one and return it
    auto declaration = StructType::Declaration::Create();
    for (size_t n = 0; n < type->getNumElements(); n++)
    {
      declaration->Append(ConvertType(type->getElementType(n), *this));
    }

    declarations_[type] = declaration.get();
    return &module().AddStructTypeDeclaration(std::move(declaration));
  }

  inline ipgraph_module &
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

private:
  ipgraph_module & module_;
  basic_block_map bbmap_;
  ipgraph_node * node_;
  const llvm::variable * result_;
  llvm::variable * iostate_;
  llvm::variable * memory_state_;
  std::unordered_map<const ::llvm::Value *, const llvm::variable *> vmap_;
  std::unordered_map<const ::llvm::StructType *, const StructType::Declaration *> declarations_;
};

}

#endif
