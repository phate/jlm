/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/frontend/ControlFlowRestructuring.hpp>
#include <jlm/llvm/frontend/InterProceduralGraphConversion.hpp>
#include <jlm/llvm/ir/aggregation.hpp>
#include <jlm/llvm/ir/Annotation.hpp>
#include <jlm/llvm/ir/cfg-structure.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/ipgraph.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/ssa.hpp>
#include <jlm/rvsdg/binary.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <stack>

namespace jlm::llvm
{

class VariableMap final
{
public:
  bool
  contains(const variable * v) const noexcept
  {
    return Map_.find(v) != Map_.end();
  }

  rvsdg::output *
  lookup(const variable * v) const
  {
    JLM_ASSERT(contains(v));
    return Map_.at(v);
  }

  void
  insert(const variable * v, rvsdg::output * o)
  {
    JLM_ASSERT(v->type() == o->type());
    Map_[v] = o;
  }

private:
  std::unordered_map<const variable *, rvsdg::output *> Map_;
};

class RegionalizedVariableMap final
{
public:
  ~RegionalizedVariableMap()
  {
    PopRegion();
    JLM_ASSERT(NumRegions() == 0);
  }

  RegionalizedVariableMap(const ipgraph_module & interProceduralGraphModule, rvsdg::region & region)
      : InterProceduralGraphModule_(interProceduralGraphModule)
  {
    PushRegion(region);
  }

  size_t
  NumRegions() const noexcept
  {
    JLM_ASSERT(VariableMapStack_.size() == RegionStack_.size());
    return VariableMapStack_.size();
  }

  llvm::VariableMap &
  VariableMap(size_t n) noexcept
  {
    JLM_ASSERT(n < NumRegions());
    return *VariableMapStack_[n];
  }

  llvm::VariableMap &
  GetTopVariableMap() noexcept
  {
    JLM_ASSERT(NumRegions() > 0);
    return VariableMap(NumRegions() - 1);
  }

  rvsdg::region &
  GetRegion(size_t n) noexcept
  {
    JLM_ASSERT(n < NumRegions());
    return *RegionStack_[n];
  }

  rvsdg::region &
  GetTopRegion() noexcept
  {
    JLM_ASSERT(NumRegions() > 0);
    return GetRegion(NumRegions() - 1);
  }

  void
  PushRegion(rvsdg::region & region)
  {
    VariableMapStack_.push_back(std::make_unique<llvm::VariableMap>());
    RegionStack_.push_back(&region);
  }

  void
  PopRegion()
  {
    VariableMapStack_.pop_back();
    RegionStack_.pop_back();
  }

  const ipgraph_module &
  GetInterProceduralGraphModule() const noexcept
  {
    return InterProceduralGraphModule_;
  }

private:
  const ipgraph_module & InterProceduralGraphModule_;
  std::vector<std::unique_ptr<llvm::VariableMap>> VariableMapStack_;
  std::vector<rvsdg::region *> RegionStack_;
};

class ControlFlowRestructuringStatistics final : public jlm::util::Statistics
{
public:
  ~ControlFlowRestructuringStatistics() override = default;

  ControlFlowRestructuringStatistics(jlm::util::filepath sourceFileName, std::string functionName)
      : Statistics(Statistics::Id::ControlFlowRecovery),
        NumNodes_(0),
        FunctionName_(std::move(functionName)),
        SourceFileName_(std::move(sourceFileName))
  {}

  void
  Start(const llvm::cfg & cfg) noexcept
  {
    NumNodes_ = cfg.nnodes();
    Timer_.start();
  }

  void
  End() noexcept
  {
    Timer_.stop();
  }

  std::string
  ToString() const override
  {
    return jlm::util::strfmt(
        "ControlFlowRestructuring ",
        SourceFileName_.to_str(),
        " ",
        FunctionName_,
        " ",
        "#Nodes:",
        NumNodes_,
        " ",
        "Time[ns]:",
        Timer_.ns());
  }

  static std::unique_ptr<ControlFlowRestructuringStatistics>
  Create(jlm::util::filepath sourceFileName, std::string functionName)
  {
    return std::make_unique<ControlFlowRestructuringStatistics>(
        std::move(sourceFileName),
        std::move(functionName));
  }

private:
  size_t NumNodes_;
  jlm::util::timer Timer_;
  std::string FunctionName_;
  jlm::util::filepath SourceFileName_;
};

class AggregationStatistics final : public jlm::util::Statistics
{
public:
  ~AggregationStatistics() override = default;

  AggregationStatistics(jlm::util::filepath sourceFileName, std::string functionName)
      : Statistics(jlm::util::Statistics::Id::Aggregation),
        NumNodes_(0),
        FunctionName_(std::move(functionName)),
        SourceFileName_(std::move(sourceFileName))
  {}

  void
  Start(const llvm::cfg & cfg) noexcept
  {
    NumNodes_ = cfg.nnodes();
    Timer_.start();
  }

  void
  End() noexcept
  {
    Timer_.stop();
  }

  std::string
  ToString() const override
  {
    return jlm::util::strfmt(
        "Aggregation ",
        SourceFileName_.to_str(),
        " ",
        FunctionName_,
        " ",
        "#Nodes:",
        NumNodes_,
        " ",
        "Time[ns]:",
        Timer_.ns());
  }

  static std::unique_ptr<AggregationStatistics>
  Create(jlm::util::filepath sourceFileName, std::string functionName)
  {
    return std::make_unique<AggregationStatistics>(
        std::move(sourceFileName),
        std::move(functionName));
  }

private:
  size_t NumNodes_;
  jlm::util::timer Timer_;
  std::string FunctionName_;
  jlm::util::filepath SourceFileName_;
};

class AnnotationStatistics final : public jlm::util::Statistics
{
public:
  ~AnnotationStatistics() override = default;

  AnnotationStatistics(jlm::util::filepath sourceFileName, std::string functionName)
      : Statistics(jlm::util::Statistics::Id::Annotation),
        NumThreeAddressCodes_(0),
        FunctionName_(std::move(functionName)),
        SourceFileName_(std::move(sourceFileName))
  {}

  void
  Start(const aggnode & node) noexcept
  {
    NumThreeAddressCodes_ = llvm::ntacs(node);
    Timer_.start();
  }

  void
  End() noexcept
  {
    Timer_.stop();
  }

  std::string
  ToString() const override
  {
    return jlm::util::strfmt(
        "Annotation ",
        SourceFileName_.to_str(),
        " ",
        FunctionName_,
        " ",
        "#ThreeAddressCodes:",
        NumThreeAddressCodes_,
        " ",
        "Time[ns]:",
        Timer_.ns());
  }

  static std::unique_ptr<AnnotationStatistics>
  Create(jlm::util::filepath sourceFileName, std::string functionName)
  {
    return std::make_unique<AnnotationStatistics>(
        std::move(sourceFileName),
        std::move(functionName));
  }

private:
  size_t NumThreeAddressCodes_;
  jlm::util::timer Timer_;
  std::string FunctionName_;
  jlm::util::filepath SourceFileName_;
};

class AggregationTreeToLambdaStatistics final : public jlm::util::Statistics
{
public:
  ~AggregationTreeToLambdaStatistics() override = default;

  AggregationTreeToLambdaStatistics(jlm::util::filepath sourceFileName, std::string functionName)
      : Statistics(jlm::util::Statistics::Id::JlmToRvsdgConversion),
        FunctionName_(std::move(functionName)),
        SourceFileName_(std::move(sourceFileName))
  {}

  void
  Start() noexcept
  {
    Timer_.start();
  }

  void
  End() noexcept
  {
    Timer_.stop();
  }

  std::string
  ToString() const override
  {
    return jlm::util::strfmt(
        "ControlFlowGraphToLambda ",
        SourceFileName_.to_str(),
        " ",
        FunctionName_,
        " ",
        "Time[ns]:",
        Timer_.ns());
  }

  static std::unique_ptr<AggregationTreeToLambdaStatistics>
  Create(jlm::util::filepath sourceFileName, std::string functionName)
  {
    return std::make_unique<AggregationTreeToLambdaStatistics>(
        std::move(sourceFileName),
        std::move(functionName));
  }

private:
  jlm::util::timer Timer_;
  std::string FunctionName_;
  jlm::util::filepath SourceFileName_;
};

class DataNodeToDeltaStatistics final : public jlm::util::Statistics
{
public:
  ~DataNodeToDeltaStatistics() override = default;

  DataNodeToDeltaStatistics(jlm::util::filepath sourceFileName, std::string dataNodeName)
      : Statistics(jlm::util::Statistics::Id::DataNodeToDelta),
        NumInitializationThreeAddressCodes_(0),
        DataNodeName_(std::move(dataNodeName)),
        SourceFileName_(std::move(sourceFileName))
  {}

  void
  Start(size_t numInitializationThreeAddressCodes) noexcept
  {
    NumInitializationThreeAddressCodes_ = numInitializationThreeAddressCodes;
    Timer_.start();
  }

  void
  End() noexcept
  {
    Timer_.stop();
  }

  std::string
  ToString() const override
  {
    return jlm::util::strfmt(
        "DataNodeToDeltaStatistics ",
        SourceFileName_.to_str(),
        " ",
        DataNodeName_,
        " ",
        "#InitializationThreeAddressCodes:",
        NumInitializationThreeAddressCodes_,
        " ",
        "Time[ns]:",
        Timer_.ns());
  }

  static std::unique_ptr<DataNodeToDeltaStatistics>
  Create(jlm::util::filepath sourceFileName, std::string dataNodeName)
  {
    return std::make_unique<DataNodeToDeltaStatistics>(
        std::move(sourceFileName),
        std::move(dataNodeName));
  }

private:
  jlm::util::timer Timer_;
  size_t NumInitializationThreeAddressCodes_;
  std::string DataNodeName_;
  jlm::util::filepath SourceFileName_;
};

class InterProceduralGraphToRvsdgStatistics final : public jlm::util::Statistics
{
public:
  ~InterProceduralGraphToRvsdgStatistics() override = default;

  explicit InterProceduralGraphToRvsdgStatistics(jlm::util::filepath sourceFileName)
      : Statistics(jlm::util::Statistics::Id::RvsdgConstruction),
        NumThreeAddressCodes_(0),
        NumRvsdgNodes_(0),
        SourceFileName_(std::move(sourceFileName))
  {}

  void
  Start(const ipgraph_module & interProceduralGraphModule) noexcept
  {
    NumThreeAddressCodes_ = llvm::ntacs(interProceduralGraphModule);
    Timer_.start();
  }

  void
  End(const rvsdg::graph & graph) noexcept
  {
    Timer_.stop();
    NumRvsdgNodes_ = rvsdg::nnodes(graph.root());
  }

  std::string
  ToString() const override
  {
    return jlm::util::strfmt(
        "InterProceduralGraphToRvsdg ",
        SourceFileName_.to_str(),
        " ",
        "#ThreeAddressCodes:",
        NumThreeAddressCodes_,
        " ",
        "#RvsdgNodes:",
        NumRvsdgNodes_,
        " ",
        "Time[ns]:",
        Timer_.ns());
  }

  static std::unique_ptr<InterProceduralGraphToRvsdgStatistics>
  Create(jlm::util::filepath sourceFileName)
  {
    return std::make_unique<InterProceduralGraphToRvsdgStatistics>(std::move(sourceFileName));
  }

private:
  size_t NumThreeAddressCodes_;
  size_t NumRvsdgNodes_;
  jlm::util::timer Timer_;
  jlm::util::filepath SourceFileName_;
};

class InterProceduralGraphToRvsdgStatisticsCollector final
{
public:
  explicit InterProceduralGraphToRvsdgStatisticsCollector(
      jlm::util::StatisticsCollector & statisticsCollector,
      jlm::util::filepath sourceFileName)
      : SourceFileName_(std::move(sourceFileName)),
        StatisticsCollector_(statisticsCollector)
  {}

  void
  CollectControlFlowRestructuringStatistics(
      const std::function<void(llvm::cfg *)> & restructureControlFlowGraph,
      llvm::cfg & cfg,
      std::string functionName)
  {
    auto statistics =
        ControlFlowRestructuringStatistics::Create(SourceFileName_, std::move(functionName));

    if (!StatisticsCollector_.GetSettings().IsDemanded(statistics->GetId()))
    {
      restructureControlFlowGraph(&cfg);
      return;
    }

    statistics->Start(cfg);
    restructureControlFlowGraph(&cfg);
    statistics->End();

    StatisticsCollector_.CollectDemandedStatistics(std::move(statistics));
  }

  std::unique_ptr<aggnode>
  CollectAggregationStatistics(
      const std::function<std::unique_ptr<aggnode>(llvm::cfg &)> & aggregateControlFlowGraph,
      llvm::cfg & cfg,
      std::string functionName)
  {
    auto statistics = AggregationStatistics::Create(SourceFileName_, std::move(functionName));

    if (!StatisticsCollector_.GetSettings().IsDemanded(statistics->GetId()))
      return aggregateControlFlowGraph(cfg);

    statistics->Start(cfg);
    auto aggregationTreeRoot = aggregateControlFlowGraph(cfg);
    statistics->End();

    StatisticsCollector_.CollectDemandedStatistics(std::move(statistics));

    return aggregationTreeRoot;
  }

  std::unique_ptr<AnnotationMap>
  CollectAnnotationStatistics(
      const std::function<std::unique_ptr<AnnotationMap>(const aggnode &)> &
          annotateAggregationTree,
      const aggnode & aggregationTreeRoot,
      std::string functionName)
  {
    auto statistics = AnnotationStatistics::Create(SourceFileName_, std::move(functionName));

    if (!StatisticsCollector_.GetSettings().IsDemanded(statistics->GetId()))
      return annotateAggregationTree(aggregationTreeRoot);

    statistics->Start(aggregationTreeRoot);
    auto demandMap = annotateAggregationTree(aggregationTreeRoot);
    statistics->End();

    StatisticsCollector_.CollectDemandedStatistics(std::move(statistics));

    return demandMap;
  }

  void
  CollectAggregationTreeToLambdaStatistics(
      const std::function<void()> & convertAggregationTreeToLambda,
      std::string functionName)
  {
    auto statistics =
        AggregationTreeToLambdaStatistics::Create(SourceFileName_, std::move(functionName));

    if (!StatisticsCollector_.GetSettings().IsDemanded(statistics->GetId()))
      return convertAggregationTreeToLambda();

    statistics->Start();
    convertAggregationTreeToLambda();
    statistics->End();

    StatisticsCollector_.CollectDemandedStatistics(std::move(statistics));
  }

  rvsdg::output *
  CollectDataNodeToDeltaStatistics(
      const std::function<rvsdg::output *()> & convertDataNodeToDelta,
      std::string dataNodeName,
      size_t NumInitializationThreeAddressCodes)
  {
    auto statistics = DataNodeToDeltaStatistics::Create(SourceFileName_, std::move(dataNodeName));

    if (!StatisticsCollector_.GetSettings().IsDemanded(statistics->GetId()))
      return convertDataNodeToDelta();

    statistics->Start(NumInitializationThreeAddressCodes);
    auto output = convertDataNodeToDelta();
    statistics->End();

    StatisticsCollector_.CollectDemandedStatistics(std::move(statistics));

    return output;
  }

  std::unique_ptr<RvsdgModule>
  CollectInterProceduralGraphToRvsdgStatistics(
      const std::function<std::unique_ptr<RvsdgModule>(const ipgraph_module &)> &
          convertInterProceduralGraphModule,
      const ipgraph_module & interProceduralGraphModule)
  {
    auto statistics = InterProceduralGraphToRvsdgStatistics::Create(SourceFileName_);

    if (!StatisticsCollector_.GetSettings().IsDemanded(statistics->GetId()))
      return convertInterProceduralGraphModule(interProceduralGraphModule);

    statistics->Start(interProceduralGraphModule);
    auto rvsdgModule = convertInterProceduralGraphModule(interProceduralGraphModule);
    statistics->End(rvsdgModule->Rvsdg());

    StatisticsCollector_.CollectDemandedStatistics(std::move(statistics));

    return rvsdgModule;
  }

private:
  const jlm::util::filepath SourceFileName_;
  jlm::util::StatisticsCollector & StatisticsCollector_;
};

static bool
requiresExport(const ipgraph_node & ipgNode)
{
  return ipgNode.hasBody() && is_externally_visible(ipgNode.linkage());
}

static void
ConvertAssignment(
    const llvm::tac & threeAddressCode,
    rvsdg::region & region,
    llvm::VariableMap & variableMap)
{
  JLM_ASSERT(is<assignment_op>(threeAddressCode.operation()));

  auto lhs = threeAddressCode.operand(0);
  auto rhs = threeAddressCode.operand(1);
  variableMap.insert(lhs, variableMap.lookup(rhs));
}

static void
ConvertSelect(
    const llvm::tac & threeAddressCode,
    rvsdg::region & region,
    llvm::VariableMap & variableMap)
{
  JLM_ASSERT(is<select_op>(threeAddressCode.operation()));
  JLM_ASSERT(threeAddressCode.noperands() == 3 && threeAddressCode.nresults() == 1);

  auto op = rvsdg::match_op(1, { { 1, 1 } }, 0, 2);
  auto p = variableMap.lookup(threeAddressCode.operand(0));
  auto predicate = rvsdg::simple_node::create_normalized(&region, op, { p })[0];

  auto gamma = rvsdg::gamma_node::create(predicate, 2);
  auto ev1 = gamma->add_entryvar(variableMap.lookup(threeAddressCode.operand(2)));
  auto ev2 = gamma->add_entryvar(variableMap.lookup(threeAddressCode.operand(1)));
  auto ex = gamma->add_exitvar({ ev1->argument(0), ev2->argument(1) });
  variableMap.insert(threeAddressCode.result(0), ex);
}

static void
ConvertBranch(
    const llvm::tac & threeAddressCode,
    rvsdg::region & region,
    llvm::VariableMap & variableMap)
{
  JLM_ASSERT(is<branch_op>(threeAddressCode.operation()));
  /*
   * Nothing needs to be done. Branches are simply ignored.
   */
}

template<class NODE, class OPERATION>
static void
Convert(const llvm::tac & threeAddressCode, rvsdg::region & region, llvm::VariableMap & variableMap)
{
  std::vector<rvsdg::output *> operands;
  for (size_t n = 0; n < threeAddressCode.noperands(); n++)
  {
    auto operand = threeAddressCode.operand(n);
    operands.push_back(variableMap.lookup(operand));
  }

  auto operation = jlm::util::AssertedCast<const OPERATION>(&threeAddressCode.operation());
  auto results = NODE::Create(region, *operation, operands);

  JLM_ASSERT(results.size() == threeAddressCode.nresults());
  for (size_t n = 0; n < threeAddressCode.nresults(); n++)
  {
    auto result = threeAddressCode.result(n);
    variableMap.insert(result, results[n]);
  }
}

static void
ConvertThreeAddressCode(
    const llvm::tac & threeAddressCode,
    rvsdg::region & region,
    llvm::VariableMap & variableMap)
{
  static std::unordered_map<
      std::type_index,
      std::function<void(const llvm::tac &, rvsdg::region &, llvm::VariableMap &)>>
      map({ { typeid(assignment_op), ConvertAssignment },
            { typeid(select_op), ConvertSelect },
            { typeid(branch_op), ConvertBranch },
            { typeid(CallOperation), Convert<CallNode, CallOperation> },
            { typeid(LoadOperation), Convert<LoadNode, LoadOperation> },
            { typeid(StoreOperation), Convert<StoreNode, StoreOperation> } });

  auto & op = threeAddressCode.operation();
  if (map.find(typeid(op)) != map.end())
    return map[typeid(op)](threeAddressCode, region, variableMap);

  std::vector<rvsdg::output *> operands;
  for (size_t n = 0; n < threeAddressCode.noperands(); n++)
    operands.push_back(variableMap.lookup(threeAddressCode.operand(n)));

  auto & simpleOperation = static_cast<const rvsdg::simple_op &>(threeAddressCode.operation());
  auto results = rvsdg::simple_node::create_normalized(&region, simpleOperation, operands);

  JLM_ASSERT(results.size() == threeAddressCode.nresults());
  for (size_t n = 0; n < threeAddressCode.nresults(); n++)
    variableMap.insert(threeAddressCode.result(n), results[n]);
}

static void
ConvertBasicBlock(
    const taclist & basicBlock,
    rvsdg::region & region,
    llvm::VariableMap & variableMap)
{
  for (const auto & threeAddressCode : basicBlock)
    ConvertThreeAddressCode(*threeAddressCode, region, variableMap);
}

static void
ConvertAggregationNode(
    const aggnode & aggregationNode,
    const AnnotationMap & demandMap,
    lambda::node & lambdaNode,
    RegionalizedVariableMap & regionalizedVariableMap);

static void
Convert(
    const entryaggnode & entryAggregationNode,
    const AnnotationMap & demandMap,
    lambda::node & lambdaNode,
    RegionalizedVariableMap & regionalizedVariableMap)
{
  auto & demandSet = demandMap.Lookup<EntryAnnotationSet>(entryAggregationNode);

  regionalizedVariableMap.PushRegion(*lambdaNode.subregion());

  auto & outerVariableMap =
      regionalizedVariableMap.VariableMap(regionalizedVariableMap.NumRegions() - 2);
  auto & topVariableMap = regionalizedVariableMap.GetTopVariableMap();

  /*
   * Add arguments
   */
  JLM_ASSERT(entryAggregationNode.narguments() == lambdaNode.nfctarguments());
  for (size_t n = 0; n < entryAggregationNode.narguments(); n++)
  {
    auto functionNodeArgument = entryAggregationNode.argument(n);
    auto lambdaNodeArgument = lambdaNode.fctargument(n);

    topVariableMap.insert(functionNodeArgument, lambdaNodeArgument);
    lambdaNodeArgument->set_attributes(functionNodeArgument->attributes());
  }

  /*
   * Add dependencies and undefined values
   */
  for (auto & v : demandSet.TopSet_.Variables())
  {
    if (outerVariableMap.contains(&v))
    {
      topVariableMap.insert(&v, lambdaNode.add_ctxvar(outerVariableMap.lookup(&v)));
    }
    else
    {
      auto value = UndefValueOperation::Create(*lambdaNode.subregion(), v.type());
      topVariableMap.insert(&v, value);
    }
  }
}

static void
Convert(
    const exitaggnode & exitAggregationNode,
    const AnnotationMap & demandMap,
    lambda::node & lambdaNode,
    RegionalizedVariableMap & regionalizedVariableMap)
{
  std::vector<rvsdg::output *> results;
  for (const auto & result : exitAggregationNode)
  {
    JLM_ASSERT(regionalizedVariableMap.GetTopVariableMap().contains(result));
    results.push_back(regionalizedVariableMap.GetTopVariableMap().lookup(result));
  }

  regionalizedVariableMap.PopRegion();
  lambdaNode.finalize(results);
}

static void
Convert(
    const blockaggnode & blockAggregationNode,
    const AnnotationMap & demandMap,
    lambda::node & lambdaNode,
    RegionalizedVariableMap & regionalizedVariableMap)
{
  ConvertBasicBlock(
      blockAggregationNode.tacs(),
      regionalizedVariableMap.GetTopRegion(),
      regionalizedVariableMap.GetTopVariableMap());
}

static void
Convert(
    const linearaggnode & linearAggregationNode,
    const AnnotationMap & demandMap,
    lambda::node & lambdaNode,
    RegionalizedVariableMap & regionalizedVariableMap)
{
  for (const auto & child : linearAggregationNode)
    ConvertAggregationNode(child, demandMap, lambdaNode, regionalizedVariableMap);
}

static void
Convert(
    const branchaggnode & branchAggregationNode,
    const AnnotationMap & demandMap,
    lambda::node & lambdaNode,
    RegionalizedVariableMap & regionalizedVariableMap)
{
  JLM_ASSERT(is<linearaggnode>(branchAggregationNode.parent()));

  /*
   * Find predicate
   */
  auto split = branchAggregationNode.parent()->child(branchAggregationNode.index() - 1);
  while (!is<blockaggnode>(split))
    split = split->child(split->nchildren() - 1);
  auto & sb = dynamic_cast<const blockaggnode *>(split)->tacs();
  JLM_ASSERT(is<branch_op>(sb.last()->operation()));
  auto predicate = regionalizedVariableMap.GetTopVariableMap().lookup(sb.last()->operand(0));

  auto gamma = rvsdg::gamma_node::create(predicate, branchAggregationNode.nchildren());

  /*
   * Add gamma inputs.
   */
  auto & demandSet = demandMap.Lookup<BranchAnnotationSet>(branchAggregationNode);
  std::unordered_map<const variable *, rvsdg::gamma_input *> gammaInputMap;
  for (auto & v : demandSet.InputVariables().Variables())
    gammaInputMap[&v] = gamma->add_entryvar(regionalizedVariableMap.GetTopVariableMap().lookup(&v));

  /*
   * Convert subregions.
   */
  std::unordered_map<const variable *, std::vector<rvsdg::output *>> xvmap;
  JLM_ASSERT(gamma->nsubregions() == branchAggregationNode.nchildren());
  for (size_t n = 0; n < gamma->nsubregions(); n++)
  {
    regionalizedVariableMap.PushRegion(*gamma->subregion(n));
    for (const auto & pair : gammaInputMap)
      regionalizedVariableMap.GetTopVariableMap().insert(pair.first, pair.second->argument(n));

    ConvertAggregationNode(
        *branchAggregationNode.child(n),
        demandMap,
        lambdaNode,
        regionalizedVariableMap);

    for (auto & v : demandSet.OutputVariables().Variables())
      xvmap[&v].push_back(regionalizedVariableMap.GetTopVariableMap().lookup(&v));
    regionalizedVariableMap.PopRegion();
  }

  /*
   * Add gamma outputs.
   */
  for (auto & v : demandSet.OutputVariables().Variables())
  {
    JLM_ASSERT(xvmap.find(&v) != xvmap.end());
    regionalizedVariableMap.GetTopVariableMap().insert(&v, gamma->add_exitvar(xvmap[&v]));
  }
}

static void
Convert(
    const loopaggnode & loopAggregationNode,
    const AnnotationMap & demandMap,
    lambda::node & lambdaNode,
    RegionalizedVariableMap & regionalizedVariableMap)
{
  auto & parentRegion = regionalizedVariableMap.GetTopRegion();

  auto theta = rvsdg::theta_node::create(&parentRegion);

  regionalizedVariableMap.PushRegion(*theta->subregion());
  auto & thetaVariableMap = regionalizedVariableMap.GetTopVariableMap();
  auto & outerVariableMap =
      regionalizedVariableMap.VariableMap(regionalizedVariableMap.NumRegions() - 2);

  /*
   * Add loop variables
   */
  auto & demandSet = demandMap.Lookup<LoopAnnotationSet>(loopAggregationNode);
  std::unordered_map<const variable *, rvsdg::theta_output *> thetaOutputMap;
  for (auto & v : demandSet.LoopVariables().Variables())
  {
    rvsdg::output * value = nullptr;
    if (!outerVariableMap.contains(&v))
    {
      value = UndefValueOperation::Create(parentRegion, v.type());
      outerVariableMap.insert(&v, value);
    }
    else
    {
      value = outerVariableMap.lookup(&v);
    }
    thetaOutputMap[&v] = theta->add_loopvar(value);
    thetaVariableMap.insert(&v, thetaOutputMap[&v]->argument());
  }

  /*
   * Convert loop body
   */
  JLM_ASSERT(loopAggregationNode.nchildren() == 1);
  ConvertAggregationNode(
      *loopAggregationNode.child(0),
      demandMap,
      lambdaNode,
      regionalizedVariableMap);

  /*
   * Update loop variables
   */
  for (auto & v : demandSet.LoopVariables().Variables())
  {
    JLM_ASSERT(thetaOutputMap.find(&v) != thetaOutputMap.end());
    thetaOutputMap[&v]->result()->divert_to(thetaVariableMap.lookup(&v));
  }

  /*
   * Find predicate
   */
  auto lblock = loopAggregationNode.child(0);
  while (lblock->nchildren() != 0)
    lblock = lblock->child(lblock->nchildren() - 1);
  JLM_ASSERT(is<blockaggnode>(lblock));
  auto & bb = static_cast<const blockaggnode *>(lblock)->tacs();
  JLM_ASSERT(is<branch_op>(bb.last()->operation()));
  auto predicate = bb.last()->operand(0);

  /*
   * Update variable map
   */
  theta->set_predicate(thetaVariableMap.lookup(predicate));
  regionalizedVariableMap.PopRegion();
  for (auto & v : demandSet.LoopVariables().Variables())
  {
    JLM_ASSERT(outerVariableMap.contains(&v));
    outerVariableMap.insert(&v, thetaOutputMap[&v]);
  }
}

template<class NODE>
static void
ConvertAggregationNode(
    const aggnode & aggregationNode,
    const AnnotationMap & demandMap,
    lambda::node & lambdaNode,
    RegionalizedVariableMap & regionalizedVariableMap)
{
  JLM_ASSERT(dynamic_cast<const NODE *>(&aggregationNode));
  auto & castedNode = *static_cast<const NODE *>(&aggregationNode);
  Convert(castedNode, demandMap, lambdaNode, regionalizedVariableMap);
}

static void
ConvertAggregationNode(
    const aggnode & aggregationNode,
    const AnnotationMap & demandMap,
    lambda::node & lambdaNode,
    RegionalizedVariableMap & regionalizedVariableMap)
{
  static std::unordered_map<
      std::type_index,
      std::function<
          void(const aggnode &, const AnnotationMap &, lambda::node &, RegionalizedVariableMap &)>>
      map({ { typeid(entryaggnode), ConvertAggregationNode<entryaggnode> },
            { typeid(exitaggnode), ConvertAggregationNode<exitaggnode> },
            { typeid(blockaggnode), ConvertAggregationNode<blockaggnode> },
            { typeid(linearaggnode), ConvertAggregationNode<linearaggnode> },
            { typeid(branchaggnode), ConvertAggregationNode<branchaggnode> },
            { typeid(loopaggnode), ConvertAggregationNode<loopaggnode> } });

  JLM_ASSERT(map.find(typeid(aggregationNode)) != map.end());
  map[typeid(aggregationNode)](aggregationNode, demandMap, lambdaNode, regionalizedVariableMap);
}

static void
RestructureControlFlowGraph(
    llvm::cfg & controlFlowGraph,
    const std::string & functionName,
    InterProceduralGraphToRvsdgStatisticsCollector & statisticsCollector)
{
  auto restructureControlFlowGraph = [](llvm::cfg * controlFlowGraph)
  {
    RestructureControlFlow(controlFlowGraph);
    straighten(*controlFlowGraph);
  };

  statisticsCollector.CollectControlFlowRestructuringStatistics(
      restructureControlFlowGraph,
      controlFlowGraph,
      functionName);
}

static std::unique_ptr<aggnode>
AggregateControlFlowGraph(
    llvm::cfg & controlFlowGraph,
    const std::string & functionName,
    InterProceduralGraphToRvsdgStatisticsCollector & statisticsCollector)
{
  auto aggregateControlFlowGraph = [](llvm::cfg & controlFlowGraph)
  {
    auto aggregationTreeRoot = aggregate(controlFlowGraph);
    aggnode::normalize(*aggregationTreeRoot);

    return aggregationTreeRoot;
  };

  auto aggregationTreeRoot = statisticsCollector.CollectAggregationStatistics(
      aggregateControlFlowGraph,
      controlFlowGraph,
      functionName);

  return aggregationTreeRoot;
}

static std::unique_ptr<AnnotationMap>
AnnotateAggregationTree(
    const aggnode & aggregationTreeRoot,
    const std::string & functionName,
    InterProceduralGraphToRvsdgStatisticsCollector & statisticsCollector)
{
  auto demandMap =
      statisticsCollector.CollectAnnotationStatistics(Annotate, aggregationTreeRoot, functionName);

  return demandMap;
}

static lambda::output *
ConvertAggregationTreeToLambda(
    const aggnode & aggregationTreeRoot,
    const AnnotationMap & demandMap,
    RegionalizedVariableMap & scopedVariableMap,
    const std::string & functionName,
    const FunctionType & functionType,
    const linkage & functionLinkage,
    const attributeset & functionAttributes,
    InterProceduralGraphToRvsdgStatisticsCollector & statisticsCollector)
{
  auto lambdaNode = lambda::node::create(
      &scopedVariableMap.GetTopRegion(),
      functionType,
      functionName,
      functionLinkage,
      functionAttributes);

  auto convertAggregationTreeToLambda = [&]()
  {
    ConvertAggregationNode(aggregationTreeRoot, demandMap, *lambdaNode, scopedVariableMap);
  };

  statisticsCollector.CollectAggregationTreeToLambdaStatistics(
      convertAggregationTreeToLambda,
      functionName);

  return lambdaNode->output();
}

static rvsdg::output *
ConvertControlFlowGraph(
    const function_node & functionNode,
    RegionalizedVariableMap & regionalizedVariableMap,
    InterProceduralGraphToRvsdgStatisticsCollector & statisticsCollector)
{
  auto & functionName = functionNode.name();
  auto & controlFlowGraph = *functionNode.cfg();

  destruct_ssa(controlFlowGraph);
  straighten(controlFlowGraph);
  purge(controlFlowGraph);

  RestructureControlFlowGraph(controlFlowGraph, functionName, statisticsCollector);

  auto aggregationTreeRoot =
      AggregateControlFlowGraph(controlFlowGraph, functionName, statisticsCollector);

  auto demandMap = AnnotateAggregationTree(*aggregationTreeRoot, functionName, statisticsCollector);

  auto lambdaOutput = ConvertAggregationTreeToLambda(
      *aggregationTreeRoot,
      *demandMap,
      regionalizedVariableMap,
      functionName,
      functionNode.fcttype(),
      functionNode.linkage(),
      functionNode.attributes(),
      statisticsCollector);

  return lambdaOutput;
}

static rvsdg::output *
ConvertFunctionNode(
    const function_node & functionNode,
    RegionalizedVariableMap & regionalizedVariableMap,
    InterProceduralGraphToRvsdgStatisticsCollector & statisticsCollector)
{
  auto & region = regionalizedVariableMap.GetTopRegion();

  /*
   * It is a function declaration as there is no control flow graph attached to the function. Simply
   * add a function import.
   */
  if (functionNode.cfg() == nullptr)
  {
    impport port(functionNode.fcttype(), functionNode.name(), functionNode.linkage());
    return region.graph()->add_import(port);
  }

  return ConvertControlFlowGraph(functionNode, regionalizedVariableMap, statisticsCollector);
}

static rvsdg::output *
ConvertDataNodeInitialization(
    const data_node_init & init,
    rvsdg::region & region,
    RegionalizedVariableMap & regionalizedVariableMap)
{
  auto & variableMap = regionalizedVariableMap.GetTopVariableMap();
  for (const auto & tac : init.tacs())
    ConvertThreeAddressCode(*tac, region, variableMap);

  return variableMap.lookup(init.value());
}

static rvsdg::output *
ConvertDataNode(
    const data_node & dataNode,
    RegionalizedVariableMap & regionalizedVariableMap,
    InterProceduralGraphToRvsdgStatisticsCollector & statisticsCollector)
{
  auto dataNodeInitialization = dataNode.initialization();

  auto convertDataNodeToDeltaNode = [&]() -> rvsdg::output *
  {
    auto & interProceduralGraphModule = regionalizedVariableMap.GetInterProceduralGraphModule();
    auto & region = regionalizedVariableMap.GetTopRegion();

    /*
     * We have a data node without initialization. Simply add an RVSDG import.
     */
    if (!dataNodeInitialization)
    {
      impport port(dataNode.GetValueType(), dataNode.name(), dataNode.linkage());
      return region.graph()->add_import(port);
    }

    /*
     * data node with initialization
     */
    auto deltaNode = delta::node::Create(
        &region,
        dataNode.GetValueType(),
        dataNode.name(),
        dataNode.linkage(),
        dataNode.Section(),
        dataNode.constant());
    auto & outerVariableMap = regionalizedVariableMap.GetTopVariableMap();
    regionalizedVariableMap.PushRegion(*deltaNode->subregion());

    /*
     * Add dependencies
     */
    for (const auto & dependency : dataNode)
    {
      auto dependencyVariable = interProceduralGraphModule.variable(dependency);
      auto argument = deltaNode->add_ctxvar(outerVariableMap.lookup(dependencyVariable));
      regionalizedVariableMap.GetTopVariableMap().insert(dependencyVariable, argument);
    }

    auto initOutput = ConvertDataNodeInitialization(
        *dataNodeInitialization,
        *deltaNode->subregion(),
        regionalizedVariableMap);
    auto deltaOutput = deltaNode->finalize(initOutput);
    regionalizedVariableMap.PopRegion();

    return deltaOutput;
  };

  auto deltaOutput = statisticsCollector.CollectDataNodeToDeltaStatistics(
      convertDataNodeToDeltaNode,
      dataNode.name(),
      dataNodeInitialization ? dataNodeInitialization->tacs().size() : 0);

  return deltaOutput;
}

static rvsdg::output *
ConvertInterProceduralGraphNode(
    const ipgraph_node & ipgNode,
    RegionalizedVariableMap & regionalizedVariableMap,
    InterProceduralGraphToRvsdgStatisticsCollector & statisticsCollector)
{
  if (auto functionNode = dynamic_cast<const function_node *>(&ipgNode))
    return ConvertFunctionNode(*functionNode, regionalizedVariableMap, statisticsCollector);

  if (auto dataNode = dynamic_cast<const data_node *>(&ipgNode))
    return ConvertDataNode(*dataNode, regionalizedVariableMap, statisticsCollector);

  JLM_UNREACHABLE("This should have never happened.");
}

static void
ConvertStronglyConnectedComponent(
    const std::unordered_set<const ipgraph_node *> & stronglyConnectedComponent,
    rvsdg::graph & graph,
    RegionalizedVariableMap & regionalizedVariableMap,
    InterProceduralGraphToRvsdgStatisticsCollector & statisticsCollector)
{
  auto & interProceduralGraphModule = regionalizedVariableMap.GetInterProceduralGraphModule();

  /*
   * It is a single node that is not self-recursive. We do not need a phi node to break any cycles.
   */
  if (stronglyConnectedComponent.size() == 1
      && !(*stronglyConnectedComponent.begin())->is_selfrecursive())
  {
    auto & ipgNode = *stronglyConnectedComponent.begin();

    auto output =
        ConvertInterProceduralGraphNode(*ipgNode, regionalizedVariableMap, statisticsCollector);

    auto ipgNodeVariable = interProceduralGraphModule.variable(ipgNode);
    regionalizedVariableMap.GetTopVariableMap().insert(ipgNodeVariable, output);

    if (requiresExport(*ipgNode))
      graph.add_export(output, { output->type(), ipgNodeVariable->name() });

    return;
  }

  phi::builder pb;
  pb.begin(graph.root());
  regionalizedVariableMap.PushRegion(*pb.subregion());

  auto & outerVariableMap =
      regionalizedVariableMap.VariableMap(regionalizedVariableMap.NumRegions() - 2);
  auto & phiVariableMap = regionalizedVariableMap.GetTopVariableMap();

  /*
   * Add recursion variables
   */
  std::unordered_map<const variable *, phi::rvoutput *> recursionVariables;
  for (const auto & ipgNode : stronglyConnectedComponent)
  {
    auto recursionVariable = pb.add_recvar(ipgNode->type());
    auto ipgNodeVariable = interProceduralGraphModule.variable(ipgNode);
    phiVariableMap.insert(ipgNodeVariable, recursionVariable->argument());
    JLM_ASSERT(recursionVariables.find(ipgNodeVariable) == recursionVariables.end());
    recursionVariables[ipgNodeVariable] = recursionVariable;
  }

  /*
   * Add phi node dependencies
   */
  for (const auto & ipgNode : stronglyConnectedComponent)
  {
    for (const auto & ipgNodeDependency : *ipgNode)
    {
      auto dependencyVariable = interProceduralGraphModule.variable(ipgNodeDependency);
      if (recursionVariables.find(dependencyVariable) == recursionVariables.end())
        phiVariableMap.insert(
            dependencyVariable,
            pb.add_ctxvar(outerVariableMap.lookup(dependencyVariable)));
    }
  }

  /*
   * Convert SCC nodes
   */
  for (const auto & ipgNode : stronglyConnectedComponent)
  {
    auto output =
        ConvertInterProceduralGraphNode(*ipgNode, regionalizedVariableMap, statisticsCollector);
    recursionVariables[interProceduralGraphModule.variable(ipgNode)]->set_rvorigin(output);
  }

  regionalizedVariableMap.PopRegion();
  pb.end();

  /*
   * Add phi outputs
   */
  for (const auto & ipgNode : stronglyConnectedComponent)
  {
    auto ipgNodeVariable = interProceduralGraphModule.variable(ipgNode);
    auto recursionVariable = recursionVariables[ipgNodeVariable];
    regionalizedVariableMap.GetTopVariableMap().insert(ipgNodeVariable, recursionVariable);
    if (requiresExport(*ipgNode))
      graph.add_export(recursionVariable, { recursionVariable->type(), ipgNodeVariable->name() });
  }
}

static std::unique_ptr<RvsdgModule>
ConvertInterProceduralGraphModule(
    const ipgraph_module & interProceduralGraphModule,
    InterProceduralGraphToRvsdgStatisticsCollector & statisticsCollector)
{
  auto rvsdgModule = RvsdgModule::Create(
      interProceduralGraphModule.source_filename(),
      interProceduralGraphModule.target_triple(),
      interProceduralGraphModule.data_layout());
  auto graph = &rvsdgModule->Rvsdg();

  auto nf = graph->node_normal_form(typeid(rvsdg::operation));
  nf->set_mutable(false);

  /* FIXME: we currently cannot handle flattened_binary_op in jlm2llvm pass */
  rvsdg::binary_op::normal_form(graph)->set_flatten(false);

  RegionalizedVariableMap regionalizedVariableMap(interProceduralGraphModule, *graph->root());

  auto stronglyConnectedComponents = interProceduralGraphModule.ipgraph().find_sccs();
  for (const auto & stronglyConnectedComponent : stronglyConnectedComponents)
    ConvertStronglyConnectedComponent(
        stronglyConnectedComponent,
        *graph,
        regionalizedVariableMap,
        statisticsCollector);

  return rvsdgModule;
}

std::unique_ptr<RvsdgModule>
ConvertInterProceduralGraphModule(
    const ipgraph_module & interProceduralGraphModule,
    jlm::util::StatisticsCollector & statisticsCollector)
{
  InterProceduralGraphToRvsdgStatisticsCollector interProceduralGraphToRvsdgStatisticsCollector(
      statisticsCollector,
      interProceduralGraphModule.source_filename());

  auto convertInterProceduralGraphModule = [&](const ipgraph_module & interProceduralGraphModule)
  {
    return ConvertInterProceduralGraphModule(
        interProceduralGraphModule,
        interProceduralGraphToRvsdgStatisticsCollector);
  };

  auto rvsdgModule =
      interProceduralGraphToRvsdgStatisticsCollector.CollectInterProceduralGraphToRvsdgStatistics(
          convertInterProceduralGraphModule,
          interProceduralGraphModule);

  return rvsdgModule;
}

}
