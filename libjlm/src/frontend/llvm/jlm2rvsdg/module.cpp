/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/frontend/llvm/jlm2rvsdg/module.hpp>
#include <jlm/frontend/llvm/jlm2rvsdg/restructuring.hpp>

#include <jlm/ir/aggregation.hpp>
#include <jlm/ir/annotation.hpp>
#include <jlm/ir/basic-block.hpp>
#include <jlm/ir/cfg-structure.hpp>
#include <jlm/ir/ipgraph.hpp>
#include <jlm/ir/ipgraph-module.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/ir/ssa.hpp>
#include <jlm/ir/tac.hpp>

#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

#include <jive/types/bitstring/type.hpp>
#include <jive/rvsdg/binary.hpp>
#include <jive/rvsdg/control.hpp>
#include <jive/rvsdg/gamma.hpp>
#include <jive/rvsdg/region.hpp>
#include <jive/rvsdg/theta.hpp>
#include <jive/rvsdg/type.hpp>

#include <stack>

static inline jive::output *
create_undef_value(jive::region * region, const jive::type & type)
{
	/*
		We currently cannot create an undef_constant_op of control type,
		as the operator expects a valuetype. Control type is a state
		type. Use a control constant as a poor man's replacement instead.
	*/
	if (auto ct = dynamic_cast<const jive::ctltype*>(&type))
		return jive_control_constant(region, ct->nalternatives(), 0);

	JLM_ASSERT(dynamic_cast<const jive::valuetype*>(&type));
	jlm::undef_constant_op op(*static_cast<const jive::valuetype*>(&type));
	return jive::simple_node::create_normalized(region, op, {})[0];
}

namespace jlm {

class vmap final {
public:
  bool
  contains(const variable * v) const noexcept
  {
    return map_.find(v) != map_.end();
  }

  jive::output *
  lookup(const variable * v) const
  {
    JLM_ASSERT(contains(v));
    return map_.at(v);
  }

  void
  insert(const variable * v, jive::output * o)
  {
    JLM_ASSERT(v->type() == o->type());
    map_[v] = o;
  }

private:
  std::unordered_map<const variable*, jive::output*> map_;
};

class scoped_vmap final {
public:
  inline
  ~scoped_vmap()
  {
    pop_scope();
    JLM_ASSERT(nscopes() == 0);
  }

  inline
  scoped_vmap(const ipgraph_module & im, jive::region * region)
    : module_(im)
  {
    push_scope(region);
  }

  inline size_t
  nscopes() const noexcept
  {
    JLM_ASSERT(vmaps_.size() == regions_.size());
    return vmaps_.size();
  }

  inline jlm::vmap &
  vmap(size_t n) noexcept
  {
    JLM_ASSERT(n < nscopes());
    return *vmaps_[n];
  }

  inline jlm::vmap &
  vmap() noexcept
  {
    JLM_ASSERT(nscopes() > 0);
    return vmap(nscopes()-1);
  }

  inline jive::region *
  region(size_t n) noexcept
  {
    JLM_ASSERT(n < nscopes());
    return regions_[n];
  }

  inline jive::region *
  region() noexcept
  {
    JLM_ASSERT(nscopes() > 0);
    return region(nscopes()-1);
  }

  inline void
  push_scope(jive::region * region)
  {
    vmaps_.push_back(std::make_unique<jlm::vmap>());
    regions_.push_back(region);
  }

  inline void
  pop_scope()
  {
    vmaps_.pop_back();
    regions_.pop_back();
  }

  const ipgraph_module &
  module() const noexcept
  {
    return module_;
  }

private:
  const ipgraph_module & module_;
  std::vector<std::unique_ptr<jlm::vmap>> vmaps_;
  std::vector<jive::region*> regions_;
};

class ControlFlowRestructuringStatistics final : public Statistics {
public:
	~ControlFlowRestructuringStatistics() override
	= default;

	ControlFlowRestructuringStatistics(
    filepath sourceFileName,
    std::string functionName)
	: Statistics(StatisticsDescriptor::StatisticsId::ControlFlowRecovery)
  , NumNodes_(0)
	, FunctionName_(std::move(functionName))
	, SourceFileName_(std::move(sourceFileName))
	{}

	void
	Start(const jlm::cfg & cfg) noexcept
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
		return strfmt("ControlFlowRestructuring ",
                  SourceFileName_.to_str(), " ",
                  FunctionName_, " ",
                  "#Nodes:", NumNodes_, " ",
                  "Time[ns]:", Timer_.ns());
	}

  static std::unique_ptr<ControlFlowRestructuringStatistics>
  Create(
    filepath sourceFileName,
    std::string functionName)
  {
    return std::make_unique<ControlFlowRestructuringStatistics>(
      std::move(sourceFileName),
      std::move(functionName));
  }

private:
	size_t NumNodes_;
	jlm::timer Timer_;
	std::string FunctionName_;
	filepath SourceFileName_;
};

class AggregationStatistics final : public Statistics {
public:
	~AggregationStatistics() override
	= default;

	AggregationStatistics(
    filepath sourceFileName,
    std::string functionName)
	: Statistics(StatisticsDescriptor::StatisticsId::Aggregation)
  , NumNodes_(0)
	, FunctionName_(std::move(functionName))
	, SourceFileName_(std::move(sourceFileName))
	{}

	void
	Start(const jlm::cfg & cfg) noexcept
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
		return strfmt("Aggregation ",
                  SourceFileName_.to_str(), " ",
                  FunctionName_, " ",
                  "#Nodes:", NumNodes_, " ",
                  "Time[ns]:", Timer_.ns());
	}

  static std::unique_ptr<AggregationStatistics>
  Create(
    filepath sourceFileName,
    std::string functionName)
  {
    return std::make_unique<AggregationStatistics>(
      std::move(sourceFileName),
      std::move(functionName));
  }

private:
	size_t NumNodes_;
	jlm::timer Timer_;
	std::string FunctionName_;
	filepath SourceFileName_;
};

class AnnotationStatistics final : public Statistics {
public:
	~AnnotationStatistics() override
	= default;

	AnnotationStatistics(
    filepath sourceFileName,
    std::string functionName)
	: Statistics(StatisticsDescriptor::StatisticsId::Annotation)
  , NumThreeAddressCodes_(0)
	, FunctionName_(std::move(functionName))
	, SourceFileName_(std::move(sourceFileName))
	{}

	void
	Start(const aggnode & node) noexcept
	{
		NumThreeAddressCodes_ = jlm::ntacs(node);
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
		return strfmt("Annotation ",
                  SourceFileName_.to_str(), " ",
                  FunctionName_, " ",
                  "#ThreeAddressCodes:", NumThreeAddressCodes_, " ",
                  "Time[ns]:", Timer_.ns());
	}

  static std::unique_ptr<AnnotationStatistics>
  Create(
    filepath sourceFileName,
    std::string functionName)
  {
    return std::make_unique<AnnotationStatistics>(
      std::move(sourceFileName),
      std::move(functionName));
  }

private:
	size_t NumThreeAddressCodes_;
	jlm::timer Timer_;
	std::string FunctionName_;
	filepath SourceFileName_;
};

class AggregationTreeToLambdaStatistics final : public Statistics {
public:
	~AggregationTreeToLambdaStatistics() override
	= default;

	AggregationTreeToLambdaStatistics(
    filepath sourceFileName,
    std::string functionName)
	: Statistics(StatisticsDescriptor::StatisticsId::JlmToRvsdgConversion)
  , FunctionName_(std::move(functionName))
	, SourceFileName_(std::move(sourceFileName))
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
		return strfmt("ControlFlowGraphToLambda ",
                  SourceFileName_.to_str(), " ",
                  FunctionName_, " ",
                  "Time[ns]:", Timer_.ns());
	}

  static std::unique_ptr<AggregationTreeToLambdaStatistics>
  Create(
    filepath sourceFileName,
    std::string functionName)
  {
    return std::make_unique<AggregationTreeToLambdaStatistics>(
      std::move(sourceFileName),
      std::move(functionName));
  }

private:
	jlm::timer Timer_;
	std::string FunctionName_;
	filepath SourceFileName_;
};

class DataNodeToDeltaStatistics final : public Statistics {
public:
  ~DataNodeToDeltaStatistics() override
  = default;

  DataNodeToDeltaStatistics(
    filepath sourceFileName,
    std::string dataNodeName)
  : Statistics(StatisticsDescriptor::StatisticsId::DataNodeToDelta)
  , NumInitializationThreeAddressCodes_(0)
  , DataNodeName_(std::move(dataNodeName))
  , SourceFileName_(std::move(sourceFileName))
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
    return strfmt("DataNodeToDeltaStatistics ",
                  SourceFileName_.to_str(), " ",
                  DataNodeName_, " ",
                  "#InitializationThreeAddressCodes:", NumInitializationThreeAddressCodes_, " ",
                  "Time[ns]:", Timer_.ns());
  }

  static std::unique_ptr<DataNodeToDeltaStatistics>
  Create(
    filepath sourceFileName,
    std::string dataNodeName)
  {
    return std::make_unique<DataNodeToDeltaStatistics>(
      std::move(sourceFileName),
      std::move(dataNodeName));
  }

private:
  jlm::timer Timer_;
  size_t NumInitializationThreeAddressCodes_;
  std::string DataNodeName_;
  filepath SourceFileName_;
};

class InterProceduralGraphToRvsdgStatistics final : public Statistics {
public:
	~InterProceduralGraphToRvsdgStatistics() override
	= default;

  explicit
	InterProceduralGraphToRvsdgStatistics(filepath sourceFileName)
	: Statistics(StatisticsDescriptor::StatisticsId::RvsdgConstruction)
  , NumThreeAddressCodes_(0)
	, NumRvsdgNodes_(0)
	, SourceFileName_(std::move(sourceFileName))
	{}

	void
	Start(const ipgraph_module & interProceduralGraphModule) noexcept
	{
		NumThreeAddressCodes_ = jlm::ntacs(interProceduralGraphModule);
		Timer_.start();
	}

	void
	End(const jive::graph & graph) noexcept
	{
		Timer_.stop();
		NumRvsdgNodes_ = jive::nnodes(graph.root());
	}

	std::string
	ToString() const override
	{
		return strfmt("InterProceduralGraphToRvsdg ",
                  SourceFileName_.to_str(), " ",
                  "#ThreeAddressCodes:", NumThreeAddressCodes_, " ",
                  "#RvsdgNodes:", NumRvsdgNodes_, " ",
                  "Time[ns]:", Timer_.ns());
	}

  static std::unique_ptr<InterProceduralGraphToRvsdgStatistics>
  Create(filepath sourceFileName)
  {
    return std::make_unique<InterProceduralGraphToRvsdgStatistics>(std::move(sourceFileName));
  }

private:
	size_t NumThreeAddressCodes_;
	size_t NumRvsdgNodes_;
	jlm::timer Timer_;
	filepath SourceFileName_;
};

class StatisticsCollector final
{
public:
  explicit
  StatisticsCollector(
    const StatisticsDescriptor & statisticsDescriptor,
    filepath sourceFileName)
  : SourceFileName_(std::move(sourceFileName))
  , StatisticsDescriptor_(statisticsDescriptor)
  {}

  void
  CollectControlFlowRestructuringStatistics(
    const std::function<void(jlm::cfg*)> & restructureControlFlowGraph,
    jlm::cfg & cfg,
    std::string functionName)
  {
    auto & statistics = Create<ControlFlowRestructuringStatistics>(std::move(functionName));
    if (!StatisticsDescriptor_.IsPrintable(statistics.GetStatisticsId())) {
      restructureControlFlowGraph(&cfg);
      return;
    }

    statistics.Start(cfg);
    restructureControlFlowGraph(&cfg);
    statistics.End();
  }

  std::unique_ptr<aggnode>
  CollectAggregationStatistics(
    const std::function<std::unique_ptr<aggnode>(jlm::cfg&)> & aggregateControlFlowGraph,
    jlm::cfg & cfg,
    std::string functionName)
  {
    auto & statistics = Create<AggregationStatistics>(std::move(functionName));
    if (!StatisticsDescriptor_.IsPrintable(statistics.GetStatisticsId()))
      return aggregateControlFlowGraph(cfg);

    statistics.Start(cfg);
    auto aggregationTreeRoot = aggregateControlFlowGraph(cfg);
    statistics.End();

    return aggregationTreeRoot;
  }

  demandmap
  CollectAnnotationStatistics(
    const std::function<demandmap(const aggnode&)> & annotateAggregationTree,
    const aggnode & aggregationTreeRoot,
    std::string functionName)
  {
    auto & statistics = Create<AnnotationStatistics>(std::move(functionName));
    if (!StatisticsDescriptor_.IsPrintable(statistics.GetStatisticsId()))
      return annotateAggregationTree(aggregationTreeRoot);

    statistics.Start(aggregationTreeRoot);
    auto demandMap = annotateAggregationTree(aggregationTreeRoot);
    statistics.End();

    return demandMap;
  }

  void
  CollectAggregationTreeToLambdaStatistics(
    const std::function<void()> & convertAggregationTreeToLambda,
    std::string functionName)
  {
    auto & statistics = Create<AggregationTreeToLambdaStatistics>(std::move(functionName));
    if (!StatisticsDescriptor_.IsPrintable(statistics.GetStatisticsId()))
      return convertAggregationTreeToLambda();

    statistics.Start();
    convertAggregationTreeToLambda();
    statistics.End();
  }

  jive::output *
  CollectDataNodeToDeltaStatistics(
    const std::function<jive::output*()> & convertDataNodeToDelta,
    std::string dataNodeName,
    size_t NumInitializationThreeAddressCodes)
  {
    auto & statistics = Create<DataNodeToDeltaStatistics>(std::move(dataNodeName));
    if (!StatisticsDescriptor_.IsPrintable(statistics.GetStatisticsId()))
      return convertDataNodeToDelta();

    statistics.Start(NumInitializationThreeAddressCodes);
    auto output = convertDataNodeToDelta();
    statistics.End();

    return output;
  }

  std::unique_ptr<RvsdgModule>
  CollectInterProceduralGraphToRvsdgStatistics(
    const std::function<std::unique_ptr<RvsdgModule>(const ipgraph_module&)> & convertInterProceduralGraphModule,
    const ipgraph_module & interProceduralGraphModule)
  {
    auto & statistics = CreateInterProceduralGraphToRvsdgStatistics();
    if (!StatisticsDescriptor_.IsPrintable(statistics.GetStatisticsId()))
      return convertInterProceduralGraphModule(interProceduralGraphModule);

    statistics.Start(interProceduralGraphModule);
    auto rvsdgModule = convertInterProceduralGraphModule(interProceduralGraphModule);
    statistics.End(rvsdgModule->Rvsdg());

    return rvsdgModule;
  }

  void
  PrintStatistics() const noexcept
  {
    for (auto & statistics : Statistics_)
      StatisticsDescriptor_.PrintStatistics(*statistics);
  }

private:
  template<class T> T &
  Create(std::string functionName)
  {
    auto statistics = T::Create(
      SourceFileName_,
      std::move(functionName));

    auto tmp = statistics.get();
    Statistics_.push_back(std::move(statistics));

    return *tmp;
  }

  InterProceduralGraphToRvsdgStatistics &
  CreateInterProceduralGraphToRvsdgStatistics()
  {
    auto statistics = InterProceduralGraphToRvsdgStatistics::Create(SourceFileName_);

    auto tmp = statistics.get();
    Statistics_.push_back(std::move(statistics));

    return *tmp;
  }

  const filepath SourceFileName_;
  std::vector<std::unique_ptr<Statistics>> Statistics_;
  const StatisticsDescriptor & StatisticsDescriptor_;
};

static bool
requiresExport(const ipgraph_node & node)
{
	return node.hasBody()
	    && is_externally_visible(node.linkage());
}

static void
convert_assignment(const jlm::tac & tac, jive::region * region, jlm::vmap & vmap)
{
	JLM_ASSERT(is<assignment_op>(tac.operation()));
	auto lhs = tac.operand(0);
	auto rhs = tac.operand(1);
	vmap.insert(lhs, vmap.lookup(rhs));
}

static void
convert_select(const jlm::tac & tac, jive::region * region, jlm::vmap & vmap)
{
	JLM_ASSERT(is<select_op>(tac.operation()));
	JLM_ASSERT(tac.noperands() == 3 && tac.nresults() == 1);

	auto op = jive::match_op(1, {{1, 1}}, 0, 2);
	auto p = vmap.lookup(tac.operand(0));
	auto predicate = jive::simple_node::create_normalized(region, op, {p})[0];

	auto gamma = jive::gamma_node::create(predicate, 2);
	auto ev1 = gamma->add_entryvar(vmap.lookup(tac.operand(2)));
	auto ev2 = gamma->add_entryvar(vmap.lookup(tac.operand(1)));
	auto ex = gamma->add_exitvar({ev1->argument(0), ev2->argument(1)});
	vmap.insert(tac.result(0), ex);
}

static void
convert_branch(const jlm::tac & tac, jive::region * region, jlm::vmap & vmap)
{
	JLM_ASSERT(is<branch_op>(tac.operation()));
}

template<class NODE> static void
Convert(
  const jlm::tac & threeAddressCode,
  jive::region * region,
  jlm::vmap & variableMap)
{
  std::vector<jive::output*> operands;
  for (size_t n = 0; n < threeAddressCode.noperands(); n++) {
    auto operand = threeAddressCode.operand(n);
    operands.push_back(variableMap.lookup(operand));
  }

  auto results = NODE::Create(operands);

  JLM_ASSERT(results.size() == threeAddressCode.nresults());
  for (size_t n = 0; n < threeAddressCode.nresults(); n++) {
    auto result = threeAddressCode.result(n);
    variableMap.insert(result, results[n]);
  }
}

static void
convert_tac(const jlm::tac & tac, jive::region * region, jlm::vmap & vmap)
{
	static std::unordered_map<
		std::type_index,
		std::function<void(const jlm::tac&, jive::region*, jlm::vmap&)>
	> map({
	  {std::type_index(typeid(assignment_op)), convert_assignment}
	, {std::type_index(typeid(select_op)), convert_select}
	, {std::type_index(typeid(branch_op)), convert_branch}
  , {typeid(CallOperation), Convert<CallNode>}
	});

	auto & op = tac.operation();
	if (map.find(typeid(op)) != map.end())
		return map[typeid(op)](tac, region, vmap);

	std::vector<jive::output*> operands;
	for (size_t n = 0; n < tac.noperands(); n++)
		operands.push_back(vmap.lookup(tac.operand(n)));

	auto results = jive::simple_node::create_normalized(region, static_cast<const jive::simple_op&>(
		tac.operation()), operands);

	JLM_ASSERT(results.size() == tac.nresults());
	for (size_t n = 0; n < tac.nresults(); n++)
		vmap.insert(tac.result(n), results[n]);
}

static void
convert_basic_block(const taclist & bb, jive::region * region, jlm::vmap & vmap)
{
	for (const auto & tac: bb)
		convert_tac(*tac, region, vmap);
}

static void
convert_node(
	const aggnode & node,
	const demandmap & dm,
	lambda::node * lambda,
	scoped_vmap & svmap);

static void
convert_entry_node(
	const aggnode & node,
	const demandmap & dm,
	lambda::node * lambda,
	scoped_vmap & svmap)
{
	JLM_ASSERT(is<entryaggnode>(&node));
	auto en = static_cast<const entryaggnode*>(&node);
	auto ds = dm.at(&node).get();

	svmap.push_scope(lambda->subregion());

	auto & pvmap = svmap.vmap(svmap.nscopes()-2);
	auto & vmap = svmap.vmap();

	/* add arguments */
	JLM_ASSERT(en->narguments() == lambda->nfctarguments());
	for (size_t n = 0; n < en->narguments(); n++) {
		auto jlmarg = en->argument(n);
		auto fctarg = lambda->fctargument(n);

		vmap.insert(jlmarg, fctarg);
		fctarg->set_attributes(jlmarg->attributes());
	}

	/* add dependencies and undefined values */
	for (const auto & v : ds->top) {
		if (pvmap.contains(v)) {
			vmap.insert(v, lambda->add_ctxvar(pvmap.lookup(v)));
		} else {
			auto value = create_undef_value(lambda->subregion(), v->type());
			vmap.insert(v, value);
		}
	}
}

static void
convert_exit_node(
	const aggnode & node,
	const demandmap & dm,
	lambda::node * lambda,
	scoped_vmap & svmap)
{
	JLM_ASSERT(is<exitaggnode>(&node));
	auto xn = static_cast<const exitaggnode*>(&node);

	std::vector<jive::output*> results;
	for (const auto & result : *xn) {
		JLM_ASSERT(svmap.vmap().contains(result));
		results.push_back(svmap.vmap().lookup(result));
	}

	svmap.pop_scope();
	lambda->finalize(results);
}

static void
convert_block_node(
	const aggnode & node,
	const demandmap & dm,
	lambda::node * lambda,
	scoped_vmap & svmap)
{
	JLM_ASSERT(is<blockaggnode>(&node));
	auto & bb = static_cast<const blockaggnode*>(&node)->tacs();
	convert_basic_block(bb, svmap.region(), svmap.vmap());
}

static void
convert_linear_node(
	const aggnode & node,
	const demandmap & dm,
	lambda::node * lambda,
	scoped_vmap & svmap)
{
	JLM_ASSERT(is<linearaggnode>(&node));

	for (const auto & child : node)
		convert_node(child, dm, lambda, svmap);
}

static void
convert_branch_node(
	const aggnode & node,
	const demandmap & dm,
	lambda::node * lambda,
	scoped_vmap & svmap)
{
	JLM_ASSERT(is<branchaggnode>(&node));
	JLM_ASSERT(is<linearaggnode>(node.parent()));

	auto split = node.parent()->child(node.index()-1);
	while (!is<blockaggnode>(split))
		split = split->child(split->nchildren()-1);
	auto & sb = dynamic_cast<const blockaggnode*>(split)->tacs();

	JLM_ASSERT(is<branch_op>(sb.last()->operation()));
	auto predicate = svmap.vmap().lookup(sb.last()->operand(0));
	auto gamma = jive::gamma_node::create(predicate, node.nchildren());

	/* add entry variables */
	auto & ds = dm.at(&node);
	std::unordered_map<const variable*, jive::gamma_input*> evmap;
	for (const auto & v : ds->top)
		evmap[v] = gamma->add_entryvar(svmap.vmap().lookup(v));

	/* convert branch cases */
	std::unordered_map<const variable*, std::vector<jive::output*>> xvmap;
	JLM_ASSERT(gamma->nsubregions() == node.nchildren());
	for (size_t n = 0; n < gamma->nsubregions(); n++) {
		svmap.push_scope(gamma->subregion(n));
		for (const auto & pair : evmap)
			svmap.vmap().insert(pair.first, pair.second->argument(n));

		convert_node(*node.child(n), dm, lambda, svmap);

		for (const auto & v : ds->bottom)
			xvmap[v].push_back(svmap.vmap().lookup(v));
		svmap.pop_scope();
	}

	/* add exit variables */
	for (const auto & v : ds->bottom) {
		JLM_ASSERT(xvmap.find(v) != xvmap.end());
		svmap.vmap().insert(v, gamma->add_exitvar(xvmap[v]));
	}
}

static void
convert_loop_node(
	const aggnode & node,
	const demandmap & dm,
	lambda::node * lambda,
	scoped_vmap & svmap)
{
	JIVE_DEBUG_ASSERT(is<loopaggnode>(&node));
	auto parent = svmap.region();

	auto theta = jive::theta_node::create(parent);

	svmap.push_scope(theta->subregion());
	auto & vmap = svmap.vmap();
	auto & pvmap = svmap.vmap(svmap.nscopes()-2);

	/* add loop variables */
	auto ds = dm.at(&node).get();
	JLM_ASSERT(ds->top == ds->bottom);
	std::unordered_map<const variable*, jive::theta_output*> lvmap;
	for (const auto & v : ds->top) {
		jive::output * value = nullptr;
		if (!pvmap.contains(v)) {
			value = create_undef_value(parent, v->type());
			JLM_ASSERT(value);
			pvmap.insert(v, value);
		} else {
			value = pvmap.lookup(v);
		}
		lvmap[v] = theta->add_loopvar(value);
		vmap.insert(v, lvmap[v]->argument());
	}

	/* convert loop body */
	JLM_ASSERT(node.nchildren() == 1);
	convert_node(*node.child(0), dm, lambda, svmap);

	/* update loop variables */
	for (const auto & v : ds->top) {
		JLM_ASSERT(lvmap.find(v) != lvmap.end());
		lvmap[v]->result()->divert_to(vmap.lookup(v));
	}

	/* find predicate */
	auto lblock = node.child(0);
	while (lblock->nchildren() != 0)
		lblock = lblock->child(lblock->nchildren()-1);
	JLM_ASSERT(is<blockaggnode>(lblock));
	auto & bb = static_cast<const blockaggnode*>(lblock)->tacs();
	JLM_ASSERT(is<branch_op>(bb.last()->operation()));
	auto predicate = bb.last()->operand(0);

	/* update variable map */
	theta->set_predicate(vmap.lookup(predicate));
	svmap.pop_scope();
	for (const auto & v : ds->bottom) {
		JLM_ASSERT(pvmap.contains(v));
		pvmap.insert(v, lvmap[v]);
	}
}

static void
convert_node(
	const aggnode & node,
	const demandmap & dm,
	lambda::node * lambda,
	scoped_vmap & svmap)
{
	static std::unordered_map<
		std::type_index,
		std::function<void(
			const aggnode&,
			const demandmap&,
			lambda::node*,
			scoped_vmap&)
		>
	> map ({
	  {typeid(entryaggnode), convert_entry_node}, {typeid(exitaggnode), convert_exit_node}
	, {typeid(blockaggnode), convert_block_node}, {typeid(linearaggnode), convert_linear_node}
	, {typeid(branchaggnode), convert_branch_node}, {typeid(loopaggnode), convert_loop_node}
	});

	JLM_ASSERT(map.find(typeid(node)) != map.end());
	map[typeid(node)](node, dm, lambda, svmap);
}

static void
RestructureControlFlowGraph(
  jlm::cfg & controlFlowGraph,
  const std::string & functionName,
  StatisticsCollector & statisticsCollector)
{
  auto restructureControlFlowGraph = [](jlm::cfg * controlFlowGraph)
  {
    restructure(controlFlowGraph);
    straighten(*controlFlowGraph);
  };

  statisticsCollector.CollectControlFlowRestructuringStatistics(
    restructureControlFlowGraph,
    controlFlowGraph,
    functionName);
}

static std::unique_ptr<aggnode>
AggregateControlFlowGraph(
  jlm::cfg & controlFlowGraph,
  const std::string & functionName,
  StatisticsCollector & statisticsCollector)
{
  auto aggregateControlFlowGraph = [](jlm::cfg & controlFlowGraph)
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

static demandmap
AnnotateAggregationTree(
  const aggnode & aggregationTreeRoot,
  const std::string & functionName,
  StatisticsCollector & statisticsCollector)
{
  auto demandMap = statisticsCollector.CollectAnnotationStatistics(
    annotate,
    aggregationTreeRoot,
    functionName);

  return demandMap;
}

static lambda::output *
ConvertAggregationTreeToLambda(
  const aggnode & aggregationTreeRoot,
  const demandmap & demandMap,
  scoped_vmap & scopedVariableMap,
  const std::string & functionName,
  const FunctionType & functionType,
  const linkage & functionLinkage,
  const attributeset & functionAttributes,
  StatisticsCollector & statisticsCollector)
{
  auto lambdaNode = lambda::node::create(
    scopedVariableMap.region(),
    functionType,
    functionName,
    functionLinkage,
    functionAttributes);

  auto convertAggregationTreeToLambda = [&]()
  {
    convert_node(aggregationTreeRoot, demandMap, lambdaNode, scopedVariableMap);
  };

  statisticsCollector.CollectAggregationTreeToLambdaStatistics(
    convertAggregationTreeToLambda,
    functionName);

  return lambdaNode->output();
}

static jive::output *
convert_cfg(
	const jlm::function_node & functionNode,
	scoped_vmap & svmap,
	StatisticsCollector & statisticsCollector)
{
  auto & functionName = functionNode.name();
	auto & controlFlowGraph = *functionNode.cfg();

	destruct_ssa(controlFlowGraph);
	straighten(controlFlowGraph);
	purge(controlFlowGraph);

  RestructureControlFlowGraph(
    controlFlowGraph,
    functionName,
    statisticsCollector);

  auto aggregationTreeRoot = AggregateControlFlowGraph(
    controlFlowGraph,
    functionName,
    statisticsCollector);

  auto demandMap = AnnotateAggregationTree(
    *aggregationTreeRoot,
    functionName,
    statisticsCollector);

  auto lambdaOutput = ConvertAggregationTreeToLambda(
    *aggregationTreeRoot,
    demandMap,
    svmap,
    functionName,
    functionNode.fcttype(),
    functionNode.linkage(),
    functionNode.attributes(),
    statisticsCollector);

	return lambdaOutput;
}

static jive::output *
construct_lambda(
	const ipgraph_node * node,
	scoped_vmap & svmap,
  StatisticsCollector & statisticsCollector)
{
	JLM_ASSERT(dynamic_cast<const function_node*>(node));
	auto & function = *static_cast<const function_node*>(node);
  auto region = svmap.region();

	if (function.cfg() == nullptr) {
		jlm::impport port(function.type(), function.name(), function.linkage());
		return region->graph()->add_import(port);
	}

	return convert_cfg(function, svmap, statisticsCollector);
}

static jive::output *
convert_initialization(const data_node_init & init, jive::region * region, scoped_vmap & svmap)
{
	auto & vmap = svmap.vmap();
	for (const auto & tac : init.tacs())
		convert_tac(*tac, region, vmap);

	return vmap.lookup(init.value());
}

static jive::output *
convert_data_node(
	const jlm::ipgraph_node * node,
	scoped_vmap & svmap,
	StatisticsCollector & statisticsCollector)
{
	JLM_ASSERT(dynamic_cast<const data_node*>(node));
	auto n = static_cast<const data_node*>(node);
  auto dataNodeInitialization = n->initialization();

  auto convertDataNodeToDeltaNode = [&]() -> jive::output*
  {
    auto & m = svmap.module();
    auto region = svmap.region();

    /* data node without initialization */
    if (!dataNodeInitialization) {
      jlm::impport port(n->type(), n->name(), n->linkage());
      return region->graph()->add_import(port);
    }

    /* data node with initialization */
    auto delta = delta::node::create(
      region,
      n->type(),
      n->name(),
      n->linkage(),
      n->constant());
    auto & pv = svmap.vmap();
    svmap.push_scope(delta->subregion());

    /* add dependencies */
    for (const auto & dp : *node) {
      auto v = m.variable(dp);
      auto argument = delta->add_ctxvar(pv.lookup(v));
      svmap.vmap().insert(v, argument);
    }

    auto deltaOutput = delta->finalize(convert_initialization(
      *dataNodeInitialization,
      delta->subregion(),
      svmap));
    svmap.pop_scope();

    return deltaOutput;
  };

  auto deltaOutput = statisticsCollector.CollectDataNodeToDeltaStatistics(
    convertDataNodeToDeltaNode,
    n->name(),
    dataNodeInitialization ? dataNodeInitialization->tacs().size() : 0);

	return deltaOutput;
}

static jive::output *
handleSingleNode(
	const ipgraph_node & node,
	scoped_vmap & svmap,
	StatisticsCollector & statisticsCollector)
{
	jive::output * output = nullptr;
	if (auto functionNode = dynamic_cast<const function_node*>(&node)) {
		output = construct_lambda(functionNode, svmap, statisticsCollector);
	} else if (auto dataNode = dynamic_cast<const data_node*>(&node)) {
		output = convert_data_node(dataNode, svmap, statisticsCollector);
	} else {
		JLM_UNREACHABLE("This should have never happened.");
	}

	return output;
}

static void
handle_scc(
	const std::unordered_set<const jlm::ipgraph_node*> & scc,
	jive::graph * graph,
	scoped_vmap & svmap,
	StatisticsCollector & statisticsCollector)
{
	auto & module = svmap.module();

	/*
		It is a single node that is not self-recursive. We do not
		need a phi node to break any cycles.
	*/
	if (scc.size() == 1 && !(*scc.begin())->is_selfrecursive()) {
		auto & node = *scc.begin();

		auto output = handleSingleNode(*node, svmap, statisticsCollector);

		auto v = module.variable(node);
		JLM_ASSERT(v);
		svmap.vmap().insert(v, output);

		if (requiresExport(*node))
			graph->add_export(output, {output->type(), v->name()});

		return;
	}

	phi::builder pb;
	pb.begin(graph->root());
	svmap.push_scope(pb.subregion());

	auto & pvmap = svmap.vmap(svmap.nscopes()-2);
	auto & vmap = svmap.vmap();

	/* add recursion variables */
	std::unordered_map<const variable*, phi::rvoutput*> recvars;
	for (const auto & node : scc) {
		auto rv = pb.add_recvar(node->type());
		auto v = module.variable(node);
		JLM_ASSERT(v);
		vmap.insert(v, rv->argument());
		JLM_ASSERT(recvars.find(v) == recvars.end());
		recvars[v] = rv;
	}

	/* add dependencies */
	for (const auto & node : scc) {
		for (const auto & dep : *node) {
			auto v = module.variable(dep);
			JLM_ASSERT(v);
			if (recvars.find(v) == recvars.end())
				vmap.insert(v, pb.add_ctxvar(pvmap.lookup(v)));
		}
	}

	/* convert SCC nodes */
	for (const auto & node : scc) {
		auto output = handleSingleNode(*node, svmap, statisticsCollector);
		recvars[module.variable(node)]->set_rvorigin(output);
	}

	svmap.pop_scope();
	pb.end();

	/* add phi outputs */
	for (const auto & node : scc) {
		auto v = module.variable(node);
		auto value = recvars[v];
		svmap.vmap().insert(v, value);
		if (requiresExport(*node))
			graph->add_export(value, {value->type(), v->name()});
	}
}

static std::unique_ptr<RvsdgModule>
convert_module(
  const ipgraph_module & interProceduralGraphModule,
  StatisticsCollector & statisticsCollector)
{
	auto rvsdgModule = RvsdgModule::Create(
    interProceduralGraphModule.source_filename(),
    interProceduralGraphModule.target_triple(),
    interProceduralGraphModule.data_layout());
	auto graph = &rvsdgModule->Rvsdg();

	auto nf = graph->node_normal_form(typeid(jive::operation));
	nf->set_mutable(false);

	/* FIXME: we currently cannot handle flattened_binary_op in jlm2llvm pass */
	jive::binary_op::normal_form(graph)->set_flatten(false);

	scoped_vmap svmap(interProceduralGraphModule, graph->root());

	/* convert ipgraph nodes */
	auto sccs = interProceduralGraphModule.ipgraph().find_sccs();
	for (const auto & scc : sccs)
		handle_scc(scc, graph, svmap, statisticsCollector);

	return rvsdgModule;
}

std::unique_ptr<RvsdgModule>
construct_rvsdg(
  const ipgraph_module & interProceduralGraphModule,
  const StatisticsDescriptor & statisticsDescriptor)
{
  StatisticsCollector statisticsCollector(
    statisticsDescriptor,
    interProceduralGraphModule.source_filename());

  auto convertInterProceduralGraphModule = [&](const ipgraph_module & interProceduralGraphModule)
  {
    return convert_module(interProceduralGraphModule, statisticsCollector);
  };

  auto rvsdgModule = statisticsCollector.CollectInterProceduralGraphToRvsdgStatistics(
    convertInterProceduralGraphModule,
    interProceduralGraphModule);

  statisticsCollector.PrintStatistics();

  return rvsdgModule;
}

}
