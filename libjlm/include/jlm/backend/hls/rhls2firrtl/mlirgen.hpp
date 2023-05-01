/*
 * Copyright 2021 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RHLS2FIRRTL_MLIRGEN_HPP
#define JLM_BACKEND_HLS_RHLS2FIRRTL_MLIRGEN_HPP

#include <jlm/ir/hls/hls.hpp>
#include <jlm/llvm/ir/operators/GetElementPtr.hpp>
#include <jlm/llvm/ir/operators/load.hpp>
#include <jlm/llvm/ir/operators/operators.hpp>
#include <jlm/llvm/ir/operators/sext.hpp>
#include <jlm/llvm/ir/operators/store.hpp>
#include <jlm/rvsdg/bitstring/comparison.hpp>
#include <jlm/rvsdg/bitstring/type.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include "base-hls.hpp"

#ifdef CIRCT

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Translation.h"

#include "circt/Dialect/FIRRTL/FIREmitter.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Support/LLVM.h"

namespace jlm {
	namespace hls{
		class MLIRGenImpl : public BaseHLS {
			std::string
			extension() override {
				return ".fir";
				}
		public:
			std::string get_text(jlm::RvsdgModule &rvsdgModule) override {return "MLIR/FIRRTL generator";}
			MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}
			circt::firrtl::CircuitOp MlirGen(const jlm::lambda::node *lamdaNode);
			void WriteModuleToFile(const circt::firrtl::FModuleOp fModuleOp, const jive::node *node);
			void WriteCircuitToFile(const circt::firrtl::CircuitOp circuit, std::string name);
			std::string toString(const circt::firrtl::CircuitOp circuit);
		private:
			// Variables
			mlir::OpBuilder builder;
			// We don't have any locations (i.e., the originating line in the
			// source file) in RVSDG, so we set all operations to unkown
			mlir::Location location = builder.getUnknownLoc();
			std::unordered_map<std::string, circt::firrtl::FModuleOp> modules;
			// FIRRTL generating functions
			std::unordered_map<jive::simple_node *, circt::firrtl::InstanceOp>
			MlirGen(hls::loop_node *loopNode, mlir::Block *body, mlir::Block *circuitBody);
			circt::firrtl::FModuleOp MlirGen(jive::region *subRegion, mlir::Block *circuitBody);
			circt::firrtl::FModuleOp MlirGen(const jive::simple_node *node);
			// Operations
			circt::firrtl::FModuleOp MlirGenSink(const jive::simple_node *node);
			circt::firrtl::FModuleOp MlirGenFork(const jive::simple_node *node);
			circt::firrtl::FModuleOp MlirGenMem(const jive::simple_node *node);
			circt::firrtl::FModuleOp MlirGenTrigger(const jive::simple_node *node);
			circt::firrtl::FModuleOp MlirGenPrint(const jive::simple_node *node);
			circt::firrtl::FModuleOp MlirGenPredicationBuffer(const jive::simple_node *node);
			circt::firrtl::FModuleOp MlirGenBuffer(const jive::simple_node *node);
			circt::firrtl::FModuleOp MlirGenDMux(const jive::simple_node *node);
			circt::firrtl::FModuleOp MlirGenNDMux(const jive::simple_node *node);
			circt::firrtl::FModuleOp MlirGenBranch(const jive::simple_node *node);
			circt::firrtl::FModuleOp MlirGenSimpleNode(const jive::simple_node *node);

			// Helper functions
			void AddClockPort(llvm::SmallVector<circt::firrtl::PortInfo> *ports);
			void AddResetPort(llvm::SmallVector<circt::firrtl::PortInfo> *ports);
			void AddMemReqPort(llvm::SmallVector<circt::firrtl::PortInfo> *ports);
			void AddMemResPort(llvm::SmallVector<circt::firrtl::PortInfo> *ports);
			void AddBundlePort(
					llvm::SmallVector<circt::firrtl::PortInfo> *ports,
					circt::firrtl::Direction direction,
					std::string name,
					circt::firrtl::FIRRTLType type);
			circt::firrtl::SubfieldOp GetSubfield(
					mlir::Block *body,
					mlir::Value value,
					int index);
			circt::firrtl::SubfieldOp GetSubfield(
					mlir::Block *body,
					mlir::Value value,
					llvm::StringRef fieldName);
			mlir::BlockArgument GetPort(circt::firrtl::FModuleOp &module, std::string portName);
			mlir::BlockArgument GetInPort(circt::firrtl::FModuleOp &module, size_t portNr);
			mlir::BlockArgument GetOutPort(circt::firrtl::FModuleOp &module, size_t portNr);
			void Connect(mlir::Block *body, mlir::Value sink, mlir::Value source);
			void PartialConnect(mlir::Block *body, mlir::Value sink, mlir::Value source);
			// Primary operations
			circt::firrtl::BitsPrimOp AddBitsOp(mlir::Block *body,
							mlir::Value value,
							int high,
							int low);
			circt::firrtl::AndPrimOp AddAndOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::XorPrimOp AddXorOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::OrPrimOp AddOrOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::NotPrimOp AddNotOp(mlir::Block *body,
							mlir::Value first);
			circt::firrtl::AddPrimOp AddAddOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::SubPrimOp AddSubOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::MulPrimOp AddMulOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::DivPrimOp AddDivOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::DShrPrimOp AddDShrOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::DShlPrimOp AddDShlOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::RemPrimOp AddRemOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::EQPrimOp AddEqOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::NEQPrimOp AddNeqOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::GTPrimOp AddGtOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::GEQPrimOp AddGeqOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::LTPrimOp AddLtOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::LEQPrimOp AddLeqOp(mlir::Block *body,
							mlir::Value first,
							mlir::Value second);
			circt::firrtl::MuxPrimOp AddMuxOp(mlir::Block *body,
							mlir::Value select,
							mlir::Value high,
							mlir::Value low);
			circt::firrtl::AsSIntPrimOp AddAsSIntOp(mlir::Block *body,
							mlir::Value value);
			circt::firrtl::AsUIntPrimOp AddAsUIntOp(mlir::Block *body,
							mlir::Value value);
			circt::firrtl::PadPrimOp AddPadOp(mlir::Block *body,
							mlir::Value value,
							int amount);
			circt::firrtl::CvtPrimOp AddCvtOp(mlir::Block *body,
							mlir::Value value);
			circt::firrtl::WireOp AddWireOp(mlir::Block *body,
							std::string name,
							int size);
			circt::firrtl::WhenOp AddWhenOp(mlir::Block *body,
							mlir::Value condition,
							bool elseStatment);
			circt::firrtl::InstanceOp AddInstanceOp(mlir::Block *body,
							jive::simple_node *node);
			circt::firrtl::ConstantOp GetConstant(mlir::Block *body, int size, int value);
			circt::firrtl::InvalidValueOp GetInvalid(mlir::Block *body, int size);

			jive::output *TraceArgument(jive::argument *arg);
			jive::simple_output *TraceStructuralOutput(jive::structural_output *out);

			void InitializeMemReq(circt::firrtl::FModuleOp module);
			circt::firrtl::BundleType::BundleElement GetReadyElement();
			circt::firrtl::BundleType::BundleElement GetValidElement();
			mlir::BlockArgument GetClockSignal(circt::firrtl::FModuleOp module);
			mlir::BlockArgument GetResetSignal(circt::firrtl::FModuleOp module);
			circt::firrtl::FModuleOp nodeToModule(const jive::simple_node *node, bool mem=false);
			circt::firrtl::IntType GetIntType(int size);
			circt::firrtl::IntType GetIntType(const jive::type *type, int extend = 0);
			std::string GetModuleName(const jive::node *node);
			bool IsIdentityMapping(const jive::match_op &op);

		};

		class MLIRGen : public BaseHLS {
			std::string
			extension() override {
				return ".fir";
			}

			std::string get_text(jlm::RvsdgModule &rm) override {return "";}
		public:
			std::string
			run(jlm::RvsdgModule &rvsdgModule) {
				// Load the FIRRTLDialect
				mlir::MLIRContext context;
				context.getOrLoadDialect<circt::firrtl::FIRRTLDialect>();

				// Generate a FIRRTL circuit of the rvsdgModule
				auto lambdaNode = get_hls_lambda(rvsdgModule);
				auto mlirGen = MLIRGenImpl(context);
				auto circuit = mlirGen.MlirGen(lambdaNode);
				// Write the FIRRTL to a file
				return mlirGen.toString(circuit);
			}
			private:
		};

	} // namespace hls
} // namespace jlm

#else

namespace jlm {
        namespace hls{
                class MLIRGen : public BaseHLS {
                        std::string
                        extension() override {
                                return ".fir";
                        }
                        std::string get_text(jlm::RvsdgModule &rm) override {return "";}
                public:
			std::string
                        run(jlm::RvsdgModule &rvsdgModule) {
				throw jlm::error("This version of jlm-hls has not been compiled with support for the CIRCT backend\n");
                        }
                        private:
                };
        } // namespace hls
} // namespace jlm
 
#endif //CIRCT

#endif // JLM_BACKEND_HLS_RHLS2FIRRTL_MLIRGEN_HPP
