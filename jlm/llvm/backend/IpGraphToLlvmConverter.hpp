/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_BACKEND_IPGRAPHTOLLVMCONVERTER_HPP
#define JLM_LLVM_BACKEND_IPGRAPHTOLLVMCONVERTER_HPP

#include <jlm/llvm/ir/attribute.hpp>
#include <jlm/llvm/ir/tac.hpp>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/IRBuilder.h>

#include <memory>

namespace llvm
{
class Instruction;
class LLVMContext;
class Module;
class Value;
}

namespace jlm::rvsdg
{
class SimpleOperation;
}

namespace jlm::llvm
{
class FunctionToPointerOperation;

class ControlFlowGraph;
class DataNode;
class ConstantFP;
class PoisonValueOperation;
class CallOperation;
class LoadNonVolatileOperation;
class LoadVolatileOperation;
class StoreVolatileOperation;
class ConstantDataArray;
class ConstantArrayOperation;
class ConstantAggregateZeroOperation;
class ConstantStruct;
class ConstantPointerNullOperation;
class ShuffleVectorOperation;
class VectorSelectOperation;
class malloc_op;
class FreeOperation;
class MemCpyNonVolatileOperation;
class MemCpyVolatileOperation;
class MemoryStateMergeOperation;
class MemoryStateSplitOperation;
class LambdaEntryMemoryStateSplitOperation;
class CallEntryMemoryStateMergeOperation;
class CallExitMemoryStateSplitOperation;
class ControlFlowGraphNode;
class ExtractValue;
class FunctionNode;
class InterProceduralGraphModule;
class LambdaExitMemoryStateMergeOperation;
class PointerToFunctionOperation;
class ThreeAddressCode;
class Variable;

class IpGraphToLlvmConverter final
{
  class Context;

public:
  ~IpGraphToLlvmConverter() noexcept;

  IpGraphToLlvmConverter();

  IpGraphToLlvmConverter(const IpGraphToLlvmConverter &) = delete;

  IpGraphToLlvmConverter(IpGraphToLlvmConverter &&) = delete;

  IpGraphToLlvmConverter &
  operator=(const IpGraphToLlvmConverter &) = delete;

  IpGraphToLlvmConverter &
  operator=(IpGraphToLlvmConverter &&) = delete;

  // FIXME: InterProceduralGraphModule should be const, but we still need to create variables to
  // translate expressions.
  std::unique_ptr<::llvm::Module>
  ConvertModule(InterProceduralGraphModule & ipGraphModule, ::llvm::LLVMContext & llvmContext);

  static ::llvm::Attribute::AttrKind
  ConvertAttributeKind(const Attribute::kind & kind);

  static std::unique_ptr<::llvm::Module>
  CreateAndConvertModule(InterProceduralGraphModule & ipGraphModule, ::llvm::LLVMContext & ctx);

private:
  void
  convert_ipgraph();

  const ::llvm::GlobalValue::LinkageTypes &
  convert_linkage(const llvm::linkage & linkage);

  void
  convert_data_node(const DataNode & node);

  void
  convert_function(const FunctionNode & node);

  void
  convert_cfg(ControlFlowGraph & cfg, ::llvm::Function & f);

  std::vector<ControlFlowGraphNode *>
  ConvertBasicBlocks(const ControlFlowGraph & controlFlowGraph, ::llvm::Function & function);

  ::llvm::AttributeList
  convert_attributes(const FunctionNode & f);

  ::llvm::AttributeSet
  convert_attributes(const attributeset & attributeSet);

  ::llvm::Attribute
  ConvertStringAttribute(const llvm::string_attribute & attribute);

  ::llvm::Attribute
  ConvertTypeAttribute(const llvm::type_attribute & attribute);

  ::llvm::Attribute
  ConvertIntAttribute(const llvm::int_attribute & attribute);

  ::llvm::Attribute
  ConvertEnumAttribute(const llvm::enum_attribute & attribute);

  void
  create_terminator_instruction(const llvm::ControlFlowGraphNode * node);

  void
  create_switch(const ControlFlowGraphNode * node);

  void
  create_conditional_branch(const ControlFlowGraphNode * node);

  void
  create_unconditional_branch(const ControlFlowGraphNode * node);

  void
  create_return(const ControlFlowGraphNode * node);

  void
  convert_tacs(const tacsvector_t & tacs);

  void
  convert_instruction(const llvm::ThreeAddressCode & tac, const llvm::ControlFlowGraphNode * node);

  ::llvm::Value *
  convert_operation(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & arguments,
      ::llvm::IRBuilder<> & builder);

  template<class OP>
  ::llvm::Value *
  convert(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const PointerToFunctionOperation &,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert(
      const FunctionToPointerOperation &,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert(
      const CallExitMemoryStateSplitOperation &,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert(
      const CallEntryMemoryStateMergeOperation &,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert(
      const LambdaExitMemoryStateMergeOperation &,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert(
      const LambdaEntryMemoryStateSplitOperation &,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert(
      const MemoryStateSplitOperation &,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert(
      const MemoryStateMergeOperation &,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert(
      const MemCpyVolatileOperation &,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const MemCpyNonVolatileOperation &,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const FreeOperation & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const malloc_op & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const ExtractValue & op,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  template<::llvm::Instruction::CastOps OPCODE>
  ::llvm::Value *
  convert_cast(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const VectorSelectOperation &,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_vectorbinary(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_vectorunary(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_insertelement(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const ShuffleVectorOperation & op,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_extractelement(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_constantdatavector(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_constantvector(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert_ctl2bits(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert_select(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const ConstantPointerNullOperation & operation,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert(
      const ConstantStruct & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert_valist(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert_fpneg(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_fpbin(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_fpcmp(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_ptrcmp(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const ConstantAggregateZeroOperation & op,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert(
      const ConstantArrayOperation & op,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert(
      const ConstantDataArray & op,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  template<typename T>
  std::vector<T>
  get_fpdata(const std::vector<const Variable *> & args);

  template<typename T>
  std::vector<T>
  get_bitdata(const std::vector<const Variable *> & args);

  ::llvm::Value *
  convert_getelementptr(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_alloca(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const StoreVolatileOperation & operation,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_store(
      const rvsdg::SimpleOperation & operation,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  void
  CreateStoreInstruction(
      const Variable * address,
      const Variable * value,
      bool isVolatile,
      size_t alignment,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const LoadVolatileOperation & operation,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const LoadNonVolatileOperation & operation,
      const std::vector<const Variable *> & operands,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  CreateLoadInstruction(
      const rvsdg::ValueType & loadedType,
      const Variable * address,
      bool isVolatile,
      size_t alignment,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_phi(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_branch(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert_match(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const CallOperation & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert(
      const PoisonValueOperation & operation,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert_undef(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> &);

  ::llvm::Value *
  convert(
      const ConstantFP & op,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_ctlconstant(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  ConverterIntegerConstant(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> &,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  CreateICmpInstruction(
      const ::llvm::CmpInst::Predicate predicate,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  CreateBinOpInstruction(
      const ::llvm::Instruction::BinaryOps opcode,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> & builder);

  ::llvm::Value *
  convert_assignment(
      const rvsdg::SimpleOperation & op,
      const std::vector<const Variable *> & args,
      ::llvm::IRBuilder<> &);

  std::unique_ptr<Context> Context_;
};

}

#endif // JLM_LLVM_BACKEND_IPGRAPHTOLLVMCONVERTER_HPP
