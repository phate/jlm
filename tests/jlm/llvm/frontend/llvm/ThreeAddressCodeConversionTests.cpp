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

static std::unique_ptr<jlm::llvm::cfg>
SetupControlFlowGraph(
    jlm::llvm::ipgraph_module & ipgModule,
    const jlm::rvsdg::simple_op & operation)
{
  using namespace jlm::llvm;

  auto cfg = jlm::llvm::cfg::create(ipgModule);

  std::vector<const variable *> operands;
  for (size_t n = 0; n < operation.narguments(); n++)
  {
    auto & operandType = operation.argument(n).Type();
    auto operand = cfg->entry()->append_argument(argument::create("", operandType));
    operands.emplace_back(operand);
  }

  auto basicBlock = basic_block::create(*cfg);
  auto threeAddressCode = basicBlock->append_last(tac::create(operation, operands));

  for (size_t n = 0; n < threeAddressCode->nresults(); n++)
  {
    auto result = threeAddressCode->result(n);
    cfg->exit()->append_result(result);
  }

  cfg->exit()->divert_inedges(basicBlock);
  basicBlock->add_outedge(cfg->exit());

  return cfg;
}

static std::unique_ptr<jlm::llvm::ipgraph_module>
SetupFunctionWithThreeAddressCode(const jlm::rvsdg::simple_op & operation)
{
  using namespace jlm::llvm;

  auto ipgModule = ipgraph_module::create(jlm::util::filepath(""), "", "");
  auto & ipgraph = ipgModule->ipgraph();

  std::vector<std::shared_ptr<const jlm::rvsdg::type>> operandTypes;
  for (size_t n = 0; n < operation.narguments(); n++)
  {
    operandTypes.emplace_back(operation.argument(n).Type());
  }

  std::vector<std::shared_ptr<const jlm::rvsdg::type>> resultTypes;
  for (size_t n = 0; n < operation.nresults(); n++)
  {
    resultTypes.emplace_back(operation.result(n).Type());
  }

  auto functionType = FunctionType::Create(operandTypes, resultTypes);

  auto functionNode =
      function_node::create(ipgraph, "test", functionType, linkage::external_linkage);
  auto cfg = SetupControlFlowGraph(*ipgModule, operation);
  functionNode->add_cfg(std::move(cfg));
  ipgModule->create_variable(functionNode);

  return ipgModule;
}

static int
LoadVolatileConversion()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  LoadVolatileOperation operation(valueType, 3, 4);
  auto ipgModule = SetupFunctionWithThreeAddressCode(operation);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  auto rvsdgModule = ConvertInterProceduralGraphModule(*ipgModule, statisticsCollector);
  std::cout << jlm::rvsdg::view(rvsdgModule->Rvsdg().root()) << std::flush;

  // Assert
  auto lambdaOutput = rvsdgModule->Rvsdg().root()->result(0)->origin();
  auto lambda = dynamic_cast<const lambda::node *>(jlm::rvsdg::node_output::node(lambdaOutput));

  auto loadVolatileNode = lambda->subregion()->nodes.first();
  assert(dynamic_cast<const LoadVolatileNode *>(loadVolatileNode));

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/llvm/ThreeAddressCodeConversionTests-LoadVolatileConversion",
    LoadVolatileConversion)

static int
StoreVolatileConversion()
{
  using namespace jlm::llvm;

  // Arrange
  auto valueType = jlm::tests::valuetype::Create();
  StoreVolatileOperation operation(valueType, 3, 4);
  auto ipgModule = SetupFunctionWithThreeAddressCode(operation);

  // Act
  jlm::util::StatisticsCollector statisticsCollector;
  auto rvsdgModule = ConvertInterProceduralGraphModule(*ipgModule, statisticsCollector);
  std::cout << jlm::rvsdg::view(rvsdgModule->Rvsdg().root()) << std::flush;

  // Assert
  auto lambdaOutput = rvsdgModule->Rvsdg().root()->result(0)->origin();
  auto lambda = dynamic_cast<const lambda::node *>(jlm::rvsdg::node_output::node(lambdaOutput));

  auto storeVolatileNode = lambda->subregion()->nodes.first();
  assert(dynamic_cast<const StoreVolatileNode *>(storeVolatileNode));

  return 0;
}

JLM_UNIT_TEST_REGISTER(
    "jlm/llvm/frontend/llvm/ThreeAddressCodeConversionTests-StoreVolatileConversion",
    StoreVolatileConversion)
