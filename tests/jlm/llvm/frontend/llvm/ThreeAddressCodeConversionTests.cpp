/*
 * Copyright 2024 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/frontend/InterProceduralGraphConversion.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/operators/Load.hpp>
#include <jlm/llvm/ir/operators/Store.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

#include <jlm/rvsdg/view.hpp>

#include <jlm/util/Statistics.hpp>

static std::unique_ptr<jlm::llvm::ControlFlowGraph>
SetupControlFlowGraph(
    jlm::llvm::InterProceduralGraphModule & ipgModule,
    const jlm::rvsdg::SimpleOperation & operation)
{
  using namespace jlm::llvm;

  auto cfg = ControlFlowGraph::create(ipgModule);

  std::vector<const Variable *> operands;
  for (size_t n = 0; n < operation.narguments(); n++)
  {
    auto & operandType = operation.argument(n);
    auto operand = cfg->entry()->append_argument(Argument::create("", operandType));
    operands.emplace_back(operand);
  }

  auto basicBlock = BasicBlock::create(*cfg);
  auto threeAddressCode = basicBlock->append_last(ThreeAddressCode::create(operation, operands));

  for (size_t n = 0; n < threeAddressCode->nresults(); n++)
  {
    auto result = threeAddressCode->result(n);
    cfg->exit()->append_result(result);
  }

  cfg->exit()->divert_inedges(basicBlock);
  basicBlock->add_outedge(cfg->exit());

  return cfg;
}

static std::unique_ptr<jlm::llvm::InterProceduralGraphModule>
SetupFunctionWithThreeAddressCode(const jlm::rvsdg::SimpleOperation & operation)
{
  using namespace jlm::llvm;

  auto ipgModule = InterProceduralGraphModule::create(jlm::util::FilePath(""), "", "");
  auto & ipgraph = ipgModule->ipgraph();

  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> operandTypes;
  for (size_t n = 0; n < operation.narguments(); n++)
  {
    operandTypes.emplace_back(operation.argument(n));
  }

  std::vector<std::shared_ptr<const jlm::rvsdg::Type>> resultTypes;
  for (size_t n = 0; n < operation.nresults(); n++)
  {
    resultTypes.emplace_back(operation.result(n));
  }

  auto functionType = jlm::rvsdg::FunctionType::Create(operandTypes, resultTypes);

  auto functionNode =
      FunctionNode::create(ipgraph, "test", functionType, Linkage::external_linkage);
  auto cfg = SetupControlFlowGraph(*ipgModule, operation);
  functionNode->add_cfg(std::move(cfg));
  ipgModule->create_variable(functionNode);

  return ipgModule;
}

static void
LoadVolatileConversion()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  LoadVolatileOperation operation(valueType, 3, 4);
  auto ipgModule = SetupFunctionWithThreeAddressCode(operation);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  auto rvsdgModule = ConvertInterProceduralGraphModule(*ipgModule, statisticsCollector);
  std::cout << jlm::rvsdg::view(&rvsdgModule->Rvsdg().GetRootRegion()) << std::flush;

  // Assert
  auto lambdaOutput = rvsdgModule->Rvsdg().GetRootRegion().result(0)->origin();
  auto lambda = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::LambdaNode>(*lambdaOutput);

  auto loadVolatileNode = lambda->subregion()->Nodes().begin().ptr();
  assert(is<LoadVolatileOperation>(loadVolatileNode));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/llvm/ThreeAddressCodeConversionTests-LoadVolatileConversion",
    LoadVolatileConversion)

static void
StoreVolatileConversion()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::ValueType::Create();
  StoreVolatileOperation operation(valueType, 3, 4);
  auto ipgModule = SetupFunctionWithThreeAddressCode(operation);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  auto rvsdgModule = ConvertInterProceduralGraphModule(*ipgModule, statisticsCollector);
  std::cout << jlm::rvsdg::view(&rvsdgModule->Rvsdg().GetRootRegion()) << std::flush;

  // Assert
  auto lambdaOutput = rvsdgModule->Rvsdg().GetRootRegion().result(0)->origin();
  auto lambda = jlm::rvsdg::TryGetOwnerNode<jlm::rvsdg::LambdaNode>(*lambdaOutput);

  auto storeVolatileNode = lambda->subregion()->Nodes().begin().ptr();
  assert(is<StoreVolatileOperation>(storeVolatileNode));
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/llvm/ThreeAddressCodeConversionTests-StoreVolatileConversion",
    StoreVolatileConversion)
