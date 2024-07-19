/*
 * Copyright 2021 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_HLS_BACKEND_RHLS2FIRRTL_RHLSTOFIRRTLCONVERTER_HPP
#define JLM_HLS_BACKEND_RHLS2FIRRTL_RHLSTOFIRRTLCONVERTER_HPP

#include <jlm/hls/backend/rhls2firrtl/base-hls.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Verifier.h>

#include <circt/Dialect/FIRRTL/FIREmitter.h>
#include <circt/Dialect/FIRRTL/FIRRTLDialect.h>
#include <circt/Dialect/FIRRTL/FIRRTLOps.h>
#include <circt/Dialect/FIRRTL/FIRRTLTypes.h>
#include <circt/Dialect/FIRRTL/Namespace.h>
#include <circt/Support/LLVM.h>

namespace jlm::hls
{

class RhlsToFirrtlConverter : public BaseHLS
{
  std::string
  extension() override
  {
    return ".fir";
  }

public:
  std::string
  get_text(llvm::RvsdgModule & rvsdgModule) override
  {
    return "MLIR/FIRRTL generator";
  }

  RhlsToFirrtlConverter()
      : Context_(std::make_unique<::mlir::MLIRContext>())
  {
    Context_->getOrLoadDialect<circt::firrtl::FIRRTLDialect>();
    Builder_ = std::make_unique<::mlir::OpBuilder>(Context_.get());
  }

  RhlsToFirrtlConverter(const RhlsToFirrtlConverter &) = delete;

  RhlsToFirrtlConverter(RhlsToFirrtlConverter &&) = delete;

  RhlsToFirrtlConverter &
  operator=(const RhlsToFirrtlConverter &) = delete;

  RhlsToFirrtlConverter &
  operator=(RhlsToFirrtlConverter &&) = delete;

  circt::firrtl::CircuitOp
  MlirGen(const llvm::lambda::node * lamdaNode);

  void
  WriteModuleToFile(const circt::firrtl::FModuleOp fModuleOp, const jlm::rvsdg::node * node);

  void
  WriteCircuitToFile(const circt::firrtl::CircuitOp circuit, std::string name);

  std::string
  ToString(llvm::RvsdgModule & rvsdgModule)
  {
    // Generate a FIRRTL circuit of the rvsdgModule
    auto lambdaNode = get_hls_lambda(rvsdgModule);
    auto mlirGen = RhlsToFirrtlConverter();
    auto circuit = mlirGen.MlirGen(lambdaNode);
    // Write the FIRRTL to a file
    return mlirGen.toString(circuit);
  }

  std::unique_ptr<mlir::ModuleOp>
  ConvertToMduleOp(llvm::RvsdgModule & rvsdgModule)
  {
    auto lambdaNode = get_hls_lambda(rvsdgModule);
    auto mlirGen = RhlsToFirrtlConverter();
    auto circuit = mlirGen.MlirGen(lambdaNode);
    std::unique_ptr<mlir::ModuleOp> module =
        std::make_unique<mlir::ModuleOp>(mlir::ModuleOp::create(Builder_->getUnknownLoc()));
    module->push_back(circuit);
    return module;
  }

private:
  std::string
  toString(const circt::firrtl::CircuitOp circuit);

  std::unordered_map<std::string, circt::firrtl::FModuleOp> modules;
  // FIRRTL generating functions
  std::unordered_map<jlm::rvsdg::simple_node *, circt::firrtl::InstanceOp>
  MlirGen(hls::loop_node * loopNode, mlir::Block * body, mlir::Block * circuitBody);
  circt::firrtl::FModuleOp
  MlirGen(jlm::rvsdg::region * subRegion, mlir::Block * circuitBody);
  circt::firrtl::FModuleOp
  MlirGen(const jlm::rvsdg::simple_node * node);
  // Operations
  circt::firrtl::FModuleOp
  MlirGenSink(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenLoopConstBuffer(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenFork(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenStateGate(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenMem(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenHlsMemResp(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenHlsMemReq(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenHlsLoad(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenHlsDLoad(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenHlsLocalMem(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenHlsStore(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenTrigger(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenPrint(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenAddrQueue(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenPredicationBuffer(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenBuffer(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenDMux(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenNDMux(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenBranch(const jlm::rvsdg::simple_node * node);
  circt::firrtl::FModuleOp
  MlirGenSimpleNode(const jlm::rvsdg::simple_node * node);

  // Helper functions
  void
  AddClockPort(::llvm::SmallVector<circt::firrtl::PortInfo> * ports);
  void
  AddResetPort(::llvm::SmallVector<circt::firrtl::PortInfo> * ports);
  void
  AddMemReqPort(::llvm::SmallVector<circt::firrtl::PortInfo> * ports);
  void
  AddMemResPort(::llvm::SmallVector<circt::firrtl::PortInfo> * ports);
  void
  AddBundlePort(
      ::llvm::SmallVector<circt::firrtl::PortInfo> * ports,
      circt::firrtl::Direction direction,
      std::string name,
      circt::firrtl::FIRRTLBaseType type);
  circt::firrtl::SubfieldOp
  GetSubfield(mlir::Block * body, mlir::Value value, int index);
  circt::firrtl::SubfieldOp
  GetSubfield(mlir::Block * body, mlir::Value value, ::llvm::StringRef fieldName);
  mlir::OpResult
  GetInstancePort(circt::firrtl::InstanceOp & instance, std::string portName);
  mlir::BlockArgument
  GetPort(circt::firrtl::FModuleOp & module, std::string portName);
  mlir::BlockArgument
  GetInPort(circt::firrtl::FModuleOp & module, size_t portNr);
  mlir::BlockArgument
  GetOutPort(circt::firrtl::FModuleOp & module, size_t portNr);
  void
  Connect(mlir::Block * body, mlir::Value sink, mlir::Value source);
  // Primary operations
  circt::firrtl::BitsPrimOp
  AddBitsOp(mlir::Block * body, mlir::Value value, int high, int low);
  circt::firrtl::AndPrimOp
  AddAndOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::NodeOp
  AddNodeOp(mlir::Block * body, mlir::Value value, std::string name);
  circt::firrtl::XorPrimOp
  AddXorOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::OrPrimOp
  AddOrOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::NotPrimOp
  AddNotOp(mlir::Block * body, mlir::Value first);
  circt::firrtl::AddPrimOp
  AddAddOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::SubPrimOp
  AddSubOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::MulPrimOp
  AddMulOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::DivPrimOp
  AddDivOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::DShrPrimOp
  AddDShrOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::DShlPrimOp
  AddDShlOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::RemPrimOp
  AddRemOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::EQPrimOp
  AddEqOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::NEQPrimOp
  AddNeqOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::GTPrimOp
  AddGtOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::GEQPrimOp
  AddGeqOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::LTPrimOp
  AddLtOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::LEQPrimOp
  AddLeqOp(mlir::Block * body, mlir::Value first, mlir::Value second);
  circt::firrtl::MuxPrimOp
  AddMuxOp(mlir::Block * body, mlir::Value select, mlir::Value high, mlir::Value low);
  circt::firrtl::AsSIntPrimOp
  AddAsSIntOp(mlir::Block * body, mlir::Value value);
  circt::firrtl::AsUIntPrimOp
  AddAsUIntOp(mlir::Block * body, mlir::Value value);
  circt::firrtl::PadPrimOp
  AddPadOp(mlir::Block * body, mlir::Value value, int amount);
  circt::firrtl::CvtPrimOp
  AddCvtOp(mlir::Block * body, mlir::Value value);
  circt::firrtl::WireOp
  AddWireOp(mlir::Block * body, std::string name, int size);
  circt::firrtl::WhenOp
  AddWhenOp(mlir::Block * body, mlir::Value condition, bool elseStatment);
  circt::firrtl::InstanceOp
  AddInstanceOp(mlir::Block * body, jlm::rvsdg::simple_node * node);
  circt::firrtl::ConstantOp
  GetConstant(mlir::Block * body, int size, int value);
  circt::firrtl::InvalidValueOp
  GetInvalid(mlir::Block * body, int size);
  void
  ConnectInvalid(mlir::Block * body, mlir::Value value);

  circt::firrtl::BitsPrimOp
  DropMSBs(mlir::Block * body, mlir::Value value, int amount);

  jlm::rvsdg::output *
  TraceArgument(jlm::rvsdg::argument * arg);
  jlm::rvsdg::simple_output *
  TraceStructuralOutput(jlm::rvsdg::structural_output * out);

  void
  InitializeMemReq(circt::firrtl::FModuleOp module);
  circt::firrtl::BundleType::BundleElement
  GetReadyElement();
  circt::firrtl::BundleType::BundleElement
  GetValidElement();
  mlir::BlockArgument
  GetClockSignal(circt::firrtl::FModuleOp module);
  mlir::BlockArgument
  GetResetSignal(circt::firrtl::FModuleOp module);
  circt::firrtl::FModuleOp
  nodeToModule(const jlm::rvsdg::simple_node * node, bool mem = false);
  circt::firrtl::IntType
  GetIntType(int size);
  circt::firrtl::IntType
  GetIntType(const jlm::rvsdg::type * type, int extend = 0);
  circt::firrtl::FIRRTLBaseType
  GetFirrtlType(const jlm::rvsdg::type * type);
  std::string
  GetModuleName(const jlm::rvsdg::node * node);
  bool
  IsIdentityMapping(const jlm::rvsdg::match_op & op);

  std::unordered_map<jlm::rvsdg::simple_node *, circt::firrtl::InstanceOp>
  createInstances(jlm::rvsdg::region * subRegion, mlir::Block * circuitBody, mlir::Block * body);
  void
  check_module(circt::firrtl::FModuleOp & module);

  std::unique_ptr<::mlir::OpBuilder> Builder_;
  std::unique_ptr<::mlir::MLIRContext> Context_;
};

} // namespace jlm::hls

#endif // JLM_HLS_BACKEND_RHLS2FIRRTL_RHLSTOFIRRTLCONVERTER_HPP
