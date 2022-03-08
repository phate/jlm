/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/rvsdg/graph.hpp>
#include <jive/rvsdg/node.hpp>
#include <jive/rvsdg/structural-node.hpp>
#include <jive/rvsdg/traverser.hpp>

#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/strfmt.hpp>
#include <jlm/util/time.hpp>

/*
	FIXME: to be removed again
*/
#include <iostream>

namespace jlm::aa {

/** \brief Steensgaard analysis statistics class
 *
 */
class SteensgaardAnalysisStatistics final : public Statistics {
public:
  ~SteensgaardAnalysisStatistics() override = default;

  explicit
  SteensgaardAnalysisStatistics(jlm::filepath sourceFile)
  : Statistics(StatisticsDescriptor::StatisticsId::SteensgaardAnalysis)
  , numNodesBefore_(0)
  , sourceFile_(std::move(sourceFile))
  {}

  void
  start(const jive::graph & graph) noexcept
  {
    numNodesBefore_ = jive::nnodes(graph.root());
    timer_.start();
  }

  void
  end() noexcept
  {
    timer_.stop();
  }

  [[nodiscard]] std::string
  ToString() const override
  {
    return strfmt("SteensgaardAnalysis ",
                  sourceFile_.to_str(), " ",
                  "#RvsdgNodes:", numNodesBefore_, " ",
                  "Time[ns]:", timer_.ns());
  }

private:
  size_t numNodesBefore_;
  jlm::filepath sourceFile_;

  jlm::timer timer_;
};

/** \brief Steensgaard PointsTo graph construction statistics class
 *
 */
class SteensgaardPointsToGraphConstructionStatistics final : public Statistics {
public:
  ~SteensgaardPointsToGraphConstructionStatistics() override = default;

  explicit
  SteensgaardPointsToGraphConstructionStatistics(jlm::filepath sourceFile)
  : Statistics(StatisticsDescriptor::StatisticsId::SteensgaardPointsToGraphConstruction)
  , sourceFile_(std::move(sourceFile))
  , numDisjointSets_(0)
  , numLocations_(0)
  , numRegisterNodes_(0)
  , numAllocatorNodes_(0)
  , numImportNodes_(0)
  , numUnknownMemorySources_(0)
  {}

  void
  start(const LocationSet & locationSet)
  {
    numDisjointSets_ = locationSet.NumDisjointSets();
    numLocations_ = locationSet.NumLocations();
    timer_.start();
  }

  void
  stop(const PointsToGraph & pointsToGraph)
  {
    timer_.stop();
    numRegisterNodes_ = pointsToGraph.nregnodes();
    numAllocatorNodes_ = pointsToGraph.nallocnodes();
    numImportNodes_ = pointsToGraph.nimpnodes();
    numUnknownMemorySources_ = pointsToGraph.memunknown().nsources();
  }

  [[nodiscard]] std::string
  ToString() const override
  {
    return strfmt("SteensgaardPointsToGraphConstruction ",
                  sourceFile_.to_str(), " ",
                  "#DisjointSets:", numDisjointSets_, " ",
                  "#Locations:", numLocations_, " ",
                  "#RegisterNodes:", numRegisterNodes_, " ",
                  "#AllocatorNodes:", numAllocatorNodes_, " ",
                  "#ImportNodes:", numImportNodes_, " ",
                  "#UnknownMemorySources:", numUnknownMemorySources_, " ",
                  "Time[ns]:", timer_.ns());
  }
private:
  jlm::filepath sourceFile_;
  size_t numDisjointSets_;
  size_t numLocations_;
  size_t numRegisterNodes_;
  size_t numAllocatorNodes_;
  size_t numImportNodes_;
  size_t numUnknownMemorySources_;
  jlm::timer timer_;
};

/** \brief Location class
 *
 * This class represents an abstract location in the program.
 */
class Location {
public:
	virtual
	~Location() = default;

	constexpr explicit
	Location(
    bool unknown,
    bool pointsToExternalMemory)
	: unknown_(unknown)
  , PointsToExternalMemory_(pointsToExternalMemory)
	, pointsto_(nullptr)
	{}

	Location(const Location &) = delete;

	Location(Location &&) = delete;

	Location &
	operator=(const Location &) = delete;

	Location &
	operator=(Location &&) = delete;

	virtual std::string
	debug_string() const noexcept = 0;

	bool
	unknown() const noexcept
	{
		return unknown_;
	}

  bool
  PointsToExternalMemory() const noexcept
  {
    return PointsToExternalMemory_;
  }

	Location *
	GetPointsTo() const noexcept
	{
		return pointsto_;
	}

	void
	SetPointsTo(Location & location) noexcept
	{
		pointsto_ = &location;
	}

	void
	set_unknown(bool unknown) noexcept
	{
		unknown_ = unknown;
	}

  void
  SetPointsToExternalMemory(bool pointsToExternalMemory) noexcept
  {
    PointsToExternalMemory_ = pointsToExternalMemory;
  }

private:
	bool unknown_;
  bool PointsToExternalMemory_;
	Location * pointsto_;
};

class RegisterLocation final : public Location {
public:
	constexpr explicit
	RegisterLocation(
    const jive::output * output,
    bool unknown = false,
    bool pointsToExternalMemory = false)
	: Location(unknown, pointsToExternalMemory)
	, output_(output)
	{}

	const jive::output *
	output() const noexcept
	{
		return output_;
	}

	std::string
	debug_string() const noexcept override
	{
		auto node = jive::node_output::node(output_);
		auto index = output_->index();

		if (jive::is<jive::simple_op>(node)) {
			auto nodestr = node->operation().debug_string();
			auto outputstr = output_->type().debug_string();
			return strfmt(nodestr, ":", index, "[" + outputstr + "]");
		}

		if (is<lambda::cvargument>(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":cv:", index);
		}

		if (is<lambda::fctargument>(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":arg:", index);
		}

		if (is<delta::cvargument>(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":cv:", index);
		}

		if (is_gamma_argument(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":arg", index);
		}

		if (is_theta_argument(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":arg", index);
		}

		if (is_theta_output(output_)) {
			auto dbgstr = jive::node_output::node(output_)->operation().debug_string();
			return strfmt(dbgstr, ":out", index);
		}

		if (is_gamma_output(output_)) {
			auto dbgstr = jive::node_output::node(output_)->operation().debug_string();
			return strfmt(dbgstr, ":out", index);
		}

		if (is_import(output_)) {
			auto imp = static_cast<const jive::impport*>(&output_->port());
			return strfmt("imp:", imp->name());
		}

		if (is<phi::rvargument>(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":rvarg", index);
		}

		if (is<phi::cvargument>(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":cvarg", index);
		}

		return strfmt(jive::node_output::node(output_)->operation().debug_string(), ":", index);
	}

	static std::unique_ptr<Location>
	create(
    const jive::output * output,
    bool unknown,
    bool pointsToExternalMemory)
	{
		return std::unique_ptr<Location>(new RegisterLocation(output, unknown, pointsToExternalMemory));
	}

private:
	const jive::output * output_;
};

/** \brief MemoryLocation class
*
* This class represents an abstract memory location.
*/
class MemoryLocation : public Location {
public:
	constexpr
	MemoryLocation()
	: Location(false, false)
  {}
};

/** \brief AllocaLocation class
 *
 * This class represents an abstract stack location allocated by a alloca operation.
 */
class AllocaLocation final : public MemoryLocation {

  ~AllocaLocation() override = default;

  constexpr explicit
  AllocaLocation(const jive::node & node)
  : MemoryLocation()
  , node_(node)
  {
    JLM_ASSERT(is<alloca_op>(&node));
  }

public:
  const jive::node &
  Node() const noexcept
  {
    return node_;
  }

  std::string
  debug_string() const noexcept override
  {
    return node_.operation().debug_string();
  }

  static std::unique_ptr<Location>
  Create(const jive::node & node)
  {
    return std::unique_ptr<Location>(new AllocaLocation(node));
  }

private:
  const jive::node & node_;
};

/** \brief MallocLocation class
 *
 * This class represents an abstract heap location allocated by a malloc operation.
 */
class MallocLocation final : public MemoryLocation {

  ~MallocLocation() override = default;

  constexpr explicit
  MallocLocation(const jive::node & node)
  : MemoryLocation()
  , node_(node)
  {
    JLM_ASSERT(is<malloc_op>(&node));
  }

public:
  const jive::node &
  Node() const noexcept
  {
    return node_;
  }

  std::string
  debug_string() const noexcept override
  {
    return node_.operation().debug_string();
  }

  static std::unique_ptr<Location>
  Create(const jive::node & node)
  {
    return std::unique_ptr<Location>(new MallocLocation(node));
  }

private:
  const jive::node & node_;
};

/** \brief LambdaLocation class
 *
 * This class represents an abstract function location, statically allocated by a lambda operation.
 */
class LambdaLocation final : public MemoryLocation {

  ~LambdaLocation() override = default;

  constexpr explicit
  LambdaLocation(const lambda::node & lambda)
  : MemoryLocation()
  , lambda_(lambda)
  {}

public:
  const lambda::node &
  Node() const noexcept
  {
    return lambda_;
  }

  std::string
  debug_string() const noexcept override
  {
    return lambda_.operation().debug_string();
  }

  static std::unique_ptr<Location>
  Create(const lambda::node & node)
  {
    return std::unique_ptr<Location>(new LambdaLocation(node));
  }

private:
  const lambda::node & lambda_;
};

/** \brief DeltaLocation class
 *
 * This class represents an abstract global variable location, statically allocated by a delta operation.
 */
class DeltaLocation final : public MemoryLocation {

  ~DeltaLocation() override = default;

  constexpr explicit
  DeltaLocation(const delta::node & delta)
    : MemoryLocation()
    , delta_(delta)
  {}

public:
  const delta::node &
  Node() const noexcept
  {
    return delta_;
  }

  std::string
  debug_string() const noexcept override
  {
    return delta_.operation().debug_string();
  }

  static std::unique_ptr<Location>
  Create(const delta::node & node)
  {
    return std::unique_ptr<Location>(new DeltaLocation(node));
  }

private:
  const delta::node & delta_;
};

/** \brief FIXME: write documentation
*
* FIXME: This class should be derived from a meloc, but we do not
* have a node to hand in.
*/
class ImportLocation final : public Location {
public:
	~ImportLocation() override = default;

	ImportLocation(
		const jive::argument * argument,
		bool pointsToUnknown,
    bool pointsToExternalMemory)
	: Location(pointsToUnknown, pointsToExternalMemory)
	, argument_(argument)
	{
		JLM_ASSERT(dynamic_cast<const jlm::impport*>(&argument->port()));
	}

	const jive::argument *
	argument() const noexcept
	{
		return argument_;
	}

	std::string
	debug_string() const noexcept override
	{
		return "IMPORT[" + argument_->debug_string() + "]";
	}

	static std::unique_ptr<Location>
	create(const jive::argument * argument)
	{
		JLM_ASSERT(is<ptrtype>(argument->type()));
		auto ptr = static_cast<const ptrtype*>(&argument->type());

		bool pointsToUnknown = is<ptrtype>(ptr->pointee_type());
    /**
     * FIXME: We use pointsToUnknown for pointsToExternalMemory
     */
		return std::unique_ptr<Location>(
      new ImportLocation(argument, pointsToUnknown, pointsToUnknown));
	}

private:
	const jive::argument * argument_;
};

/** \brief FIXME: write documentation
*/
class DummyLocation final : public Location {
public:
	~DummyLocation() override = default;

	DummyLocation()
	: Location(false, false)
	{}

	std::string
	debug_string() const noexcept override
	{
		return "UNNAMED";
	}

	static std::unique_ptr<Location>
	create()
	{
		return std::make_unique<DummyLocation>();
	}
};

/**
 * LocationSet class
 */

LocationSet::~LocationSet() = default;

LocationSet::LocationSet() = default;

void
LocationSet::clear()
{
	map_.clear();
	djset_.clear();
	locations_.clear();
}

Location &
LocationSet::InsertRegisterLocation(
  const jive::output * output,
  bool unknown,
  bool pointsToExternalMemory)
{
	JLM_ASSERT(contains(output) == false);

	locations_.push_back(RegisterLocation::create(output, unknown, pointsToExternalMemory));
	auto location = locations_.back().get();

	map_[output] = location;
	djset_.insert(location);

	return *location;
}

Location &
LocationSet::InsertAllocaLocation(const jive::node & node)
{
  locations_.push_back(AllocaLocation::Create(node));
  auto location = locations_.back().get();
  djset_.insert(location);

  return *location;
}

Location &
LocationSet::InsertMallocLocation(const jive::node & node)
{
  locations_.push_back(MallocLocation::Create(node));
  auto location = locations_.back().get();
  djset_.insert(location);

  return *location;
}

Location &
LocationSet::InsertLambdaLocation(const lambda::node & node)
{
  locations_.push_back(LambdaLocation::Create(node));
  auto location = locations_.back().get();
  djset_.insert(location);

  return *location;
}

Location &
LocationSet::InsertDeltaLocation(const delta::node & node)
{
  locations_.push_back(DeltaLocation::Create(node));
  auto location = locations_.back().get();
  djset_.insert(location);

  return *location;
}

Location &
LocationSet::InsertDummyLocation()
{
	locations_.push_back(DummyLocation::create());
	auto location = locations_.back().get();
	djset_.insert(location);

	return *location;
}

Location &
LocationSet::InsertImportLocation(const jive::argument * argument)
{
	locations_.push_back(ImportLocation::create(argument));
	auto location = locations_.back().get();
	djset_.insert(location);

	return *location;
}

Location *
LocationSet::lookup(const jive::output * output)
{
	auto it = map_.find(output);
	return it == map_.end() ? nullptr : it->second;
}

bool
LocationSet::contains(const jive::output * output) const noexcept
{
	return map_.find(output) != map_.end();
}

Location &
LocationSet::FindOrInsertRegisterLocation(
  const jive::output * output,
  bool unknown,
  bool pointsToExternalMemory)
{
	if (auto location = lookup(output))
		return GetRootLocation(*location);

	return InsertRegisterLocation(output, unknown, pointsToExternalMemory);
}

Location &
LocationSet::GetRootLocation(Location & l) const
{
	return *set(l).value();
}

Location &
LocationSet::Find(const jive::output * output)
{
	auto location = lookup(output);
	JLM_ASSERT(location != nullptr);

	return GetRootLocation(*location);
}

Location &
LocationSet::Merge(Location & l1, Location & l2)
{
	return *djset_.merge(&l1, &l2)->value();
}

std::string
LocationSet::to_dot() const
{
	auto dot_node = [](const DisjointLocationSet::set & set)
	{
		auto root = set.value();

		std::string label;
		for (auto & l : set) {
			auto unknownstr = l->unknown() ? "{U}" : "";
			auto ptstr = strfmt("{pt:", (intptr_t) l->GetPointsTo(), "}");
			auto locstr = strfmt((intptr_t)l, " : ", l->debug_string());

			if (l == root) {
				label += strfmt("*", locstr, unknownstr, ptstr, "*\\n");
			} else {
				label += strfmt(locstr, "\\n");
			}
		}

		return strfmt("{ ", (intptr_t)&set, " [label = \"", label, "\"]; }");
	};

	auto dot_edge = [&](const DisjointLocationSet::set & set, const DisjointLocationSet::set & ptset)
	{
		return strfmt((intptr_t)&set, " -> ", (intptr_t)&ptset);
	};

	std::string str;
	str.append("digraph PointsToGraph {\n");

	for (auto & set : djset_) {
		str += dot_node(set) + "\n";

		auto pt = set.value()->GetPointsTo();
		if (pt != nullptr) {
			auto ptset = djset_.find(pt);
			str += dot_edge(set, *ptset) + "\n";
		}
	}

	str.append("}\n");

	return str;
}

/* steensgaard class */

Steensgaard::~Steensgaard() = default;

void
Steensgaard::join(Location & x, Location & y)
{
	std::function<Location*(Location*, Location*)>
	join = [&](Location * x, Location * y)
	{
		if (x == nullptr)
			return y;

		if (y == nullptr)
			return x;

		if (x == y)
			return x;

		auto & rootx = locationSet_.GetRootLocation(*x);
		auto & rooty = locationSet_.GetRootLocation(*y);
    rootx.SetPointsToExternalMemory(rootx.PointsToExternalMemory() || rooty.PointsToExternalMemory());
    rooty.SetPointsToExternalMemory(rootx.PointsToExternalMemory() || rooty.PointsToExternalMemory());
		rootx.set_unknown(rootx.unknown() || rooty.unknown());
		rooty.set_unknown(rootx.unknown() || rooty.unknown());
		auto & tmp = locationSet_.Merge(rootx, rooty);

		if (auto root = join(rootx.GetPointsTo(), rooty.GetPointsTo()))
      tmp.SetPointsTo(*root);

		return &tmp;
	};

	join(&x, &y);
}

void
Steensgaard::Analyze(const jive::simple_node & node)
{
  auto AnalyzeCall = [](auto & s, auto & n) { s.AnalyzeCall(*static_cast<const CallNode*>(&n)); };

	static std::unordered_map<
		std::type_index
	, std::function<void(Steensgaard&, const jive::simple_node&)>> nodes
	({
	  {typeid(alloca_op),             [](auto & s, auto & n){ s.AnalyzeAlloca(n);                }}
	, {typeid(malloc_op),             [](auto & s, auto & n){ s.AnalyzeMalloc(n);                }}
	, {typeid(LoadOperation),         [](auto & s, auto & n){ s.AnalyzeLoad(n);                  }}
	, {typeid(StoreOperation),        [](auto & s, auto & n){ s.AnalyzeStore(n);                 }}
	, {typeid(CallOperation),         AnalyzeCall                                                 }
	, {typeid(getelementptr_op),      [](auto & s, auto & n){ s.AnalyzeGep(n);                   }}
	, {typeid(bitcast_op),            [](auto & s, auto & n){ s.AnalyzeBitcast(n);               }}
	, {typeid(bits2ptr_op),           [](auto & s, auto & n){ s.AnalyzeBits2ptr(n);              }}
	, {typeid(ptr_constant_null_op),  [](auto & s, auto & n){ s.AnalyzeNull(n);                  }}
	, {typeid(UndefValueOperation),   [](auto & s, auto & n){ s.AnalyzeUndef(n);                 }}
	, {typeid(Memcpy),                [](auto & s, auto & n){ s.AnalyzeMemcpy(n);                }}
	, {typeid(ConstantArray),         [](auto & s, auto & n){ s.AnalyzeConstantArray(n);         }}
	, {typeid(ConstantStruct),        [](auto & s, auto & n){ s.AnalyzeConstantStruct(n);        }}
	, {typeid(ConstantAggregateZero), [](auto & s, auto & n){ s.AnalyzeConstantAggregateZero(n); }}
	, {typeid(ExtractValue),          [](auto & s, auto & n){ s.AnalyzeExtractValue(n);          }}
	});

	auto & op = node.operation();
	if (nodes.find(typeid(op)) != nodes.end()) {
		nodes[typeid(op)](*this, node);
		return;
	}

	/*
		Ensure that we really took care of all pointer-producing instructions
	*/
	for (size_t n = 0; n < node.noutputs(); n++) {
		if (jive::is<ptrtype>(node.output(n)->type()))
			JLM_UNREACHABLE("We should have never reached this statement.");
	}
}

void
Steensgaard::AnalyzeAlloca(const jive::simple_node & node)
{
	JLM_ASSERT(is<alloca_op>(&node));

	std::function<bool(const jive::valuetype&)>
	IsVaListAlloca = [&](const jive::valuetype & type)
	{
		auto structType = dynamic_cast<const structtype*>(&type);

		if (structType != nullptr
		&& structType->name() == "struct.__va_list_tag")
			return true;

		if (structType != nullptr) {
			auto declaration = structType->declaration();

			for (size_t n = 0; n < declaration->nelements(); n++) {
				if (IsVaListAlloca(declaration->element(n)))
					return true;
			}
		}

		if (auto arrayType = dynamic_cast<const arraytype*>(&type))
			return IsVaListAlloca(arrayType->element_type());

		return false;
	};

	auto & allocaOutputLocation = locationSet_.FindOrInsertRegisterLocation(node.output(0), false, false);
	auto & allocaLocation = locationSet_.InsertAllocaLocation(node);
  allocaOutputLocation.SetPointsTo(allocaLocation);

	auto & op = *dynamic_cast<const alloca_op*>(&node.operation());
	/*
		FIXME: We should discover such an alloca already at construction time
		and not by traversing the type here.
	*/
	if (IsVaListAlloca(op.value_type())) {
		/*
			FIXME: We should be able to do better than just pointing to unknown.
		*/
		allocaLocation.set_unknown(true);
	}
}

void
Steensgaard::AnalyzeMalloc(const jive::simple_node & node)
{
	JLM_ASSERT(is<malloc_op>(&node));

	auto & mallocOutputLocation = locationSet_.FindOrInsertRegisterLocation(node.output(0), false, false);
  auto & mallocLocation = locationSet_.InsertMallocLocation(node);
  mallocOutputLocation.SetPointsTo(mallocLocation);
}

void
Steensgaard::AnalyzeLoad(const jive::simple_node & node)
{
	JLM_ASSERT(is<LoadOperation>(&node));

	if (!is<ptrtype>(node.output(0)->type()))
		return;

	auto & address = locationSet_.Find(node.input(0)->origin());
	auto & result = locationSet_.FindOrInsertRegisterLocation(
    node.output(0),
    address.unknown(),
    address.PointsToExternalMemory());

	if (address.GetPointsTo() == nullptr) {
    address.SetPointsTo(result);
		return;
	}

	join(result, *address.GetPointsTo());
}

void
Steensgaard::AnalyzeStore(const jive::simple_node & node)
{
	JLM_ASSERT(is<StoreOperation>(&node));

	auto address = node.input(0)->origin();
	auto value = node.input(1)->origin();

	if (!is<ptrtype>(value->type()))
		return;

	auto & addressLocation = locationSet_.Find(address);
	auto & valueLocation = locationSet_.Find(value);

	if (addressLocation.GetPointsTo() == nullptr) {
    addressLocation.SetPointsTo(valueLocation);
		return;
	}

	join(*addressLocation.GetPointsTo(), valueLocation);
}

void
Steensgaard::AnalyzeCall(const CallNode & callNode)
{
	auto handle_direct_call = [&](const CallNode & call, const lambda::node & lambda)
	{
		/*
			FIXME: What about varargs
		*/

		/* handle call node arguments */
		JLM_ASSERT(lambda.nfctarguments() == call.ninputs()-1);
		for (size_t n = 1; n < call.ninputs(); n++) {
			auto callArgument = call.input(n)->origin();
			auto lambdaArgument = lambda.fctargument(n-1);

			if (!is<ptrtype>(callArgument->type()))
				continue;

			auto & callArgumentLocation = locationSet_.Find(callArgument);
			auto & lambdaArgumentLocation = locationSet_.contains(lambdaArgument)
				? locationSet_.Find(lambdaArgument)
				: locationSet_.FindOrInsertRegisterLocation(lambdaArgument, false, false);

			join(callArgumentLocation, lambdaArgumentLocation);
		}

		/* handle call node results */
		auto subregion = lambda.subregion();
		JLM_ASSERT(subregion->nresults() == callNode.noutputs());
		for (size_t n = 0; n < call.noutputs(); n++) {
			auto callResult = call.output(n);
			auto lambdaResult = subregion->result(n)->origin();

			if (!is<ptrtype>(callResult->type()))
				continue;

			auto & callResultLocation = locationSet_.FindOrInsertRegisterLocation(callResult, false, false);
			auto & lambdaResultLocation = locationSet_.contains(lambdaResult)
				? locationSet_.Find(lambdaResult)
				: locationSet_.FindOrInsertRegisterLocation(lambdaResult, false, false);

			join(callResultLocation, lambdaResultLocation);
		}
	};

	auto handle_indirect_call = [&](const CallNode & call)
	{
		/*
			Nothing can be done for the call/lambda arguments, as it is
			an indirect call and the lambda node cannot be retrieved.
		*/

		/* handle call node results */
		for (size_t n = 0; n < call.noutputs(); n++) {
			auto callres = call.output(n);
			if (!is<ptrtype>(callres->type()))
				continue;

      locationSet_.FindOrInsertRegisterLocation(callres, true, true);
		}
	};

	if (auto lambda = CallNode::IsDirectCall(callNode)) {
		handle_direct_call(callNode, *lambda);
		return;
	}

	handle_indirect_call(callNode);
}

void
Steensgaard::AnalyzeGep(const jive::simple_node & node)
{
	JLM_ASSERT(is<getelementptr_op>(&node));

	auto & base = locationSet_.Find(node.input(0)->origin());
	auto & value = locationSet_.FindOrInsertRegisterLocation(node.output(0), false, false);

	join(base, value);
}

void
Steensgaard::AnalyzeBitcast(const jive::simple_node & node)
{
	JLM_ASSERT(is<bitcast_op>(&node));

	auto input = node.input(0);
	if (!is<ptrtype>(input->type()))
		return;

	auto & operand = locationSet_.Find(input->origin());
	auto & result = locationSet_.FindOrInsertRegisterLocation(node.output(0), false, false);

	join(operand, result);
}

void
Steensgaard::AnalyzeBits2ptr(const jive::simple_node & node)
{
	JLM_ASSERT(is<bits2ptr_op>(&node));

  locationSet_.FindOrInsertRegisterLocation(node.output(0), true, true);
}

void
Steensgaard::AnalyzeExtractValue(const jive::simple_node & node)
{
	JLM_ASSERT(is<ExtractValue>(&node));

	auto result = node.output(0);
	if (!is<ptrtype>(result->type()))
		return;

  locationSet_.FindOrInsertRegisterLocation(result, true, true);
}

void
Steensgaard::AnalyzeNull(const jive::simple_node & node)
{
	JLM_ASSERT(is<ptr_constant_null_op>(&node));

	/*
		FIXME: This should not point to unknown, but to a NULL memory location.
	*/
  locationSet_.FindOrInsertRegisterLocation(node.output(0), true, true);
}

void
Steensgaard::AnalyzeConstantAggregateZero(const jive::simple_node & node)
{
	JLM_ASSERT(is<ConstantAggregateZero>(&node));

	/*
		FIXME: This not point to unknown, but to a NULL memory location.
	*/
  locationSet_.FindOrInsertRegisterLocation(node.output(0), true, true);
}

void
Steensgaard::AnalyzeUndef(const jive::simple_node & node)
{
	JLM_ASSERT(is<UndefValueOperation>(&node));
	auto output = node.output(0);

	if (!is<ptrtype>(output->type()))
		return;

	/*
		FIXME: Overthink whether it is correct that undef points to unknown.
	*/
  locationSet_.FindOrInsertRegisterLocation(node.output(0), true, true);
}

void
Steensgaard::AnalyzeConstantArray(const jive::simple_node & node)
{
	JLM_ASSERT(is<ConstantArray>(&node));

	for (size_t n = 0; n < node.ninputs(); n++) {
		auto input = node.input(n);

		if (locationSet_.contains(input->origin())) {
			auto & originLocation = locationSet_.Find(input->origin());
			auto & outputLocation = locationSet_.FindOrInsertRegisterLocation(node.output(0), false, false);
			join(outputLocation, originLocation);
		}
	}
}

void
Steensgaard::AnalyzeConstantStruct(const jive::simple_node & node)
{
	JLM_ASSERT(is<ConstantStruct>(&node));

	for (size_t n = 0; n < node.ninputs(); n++) {
		auto input = node.input(n);

		if (locationSet_.contains(input->origin())) {
			auto & originLocation = locationSet_.Find(input->origin());
			auto & outputLocation = locationSet_.FindOrInsertRegisterLocation(node.output(0), false, false);
			join(outputLocation, originLocation);
		}
	}
}

void
Steensgaard::AnalyzeMemcpy(const jive::simple_node & node)
{
	JLM_ASSERT(is<Memcpy>(&node));

	/*
		FIXME: handle unknown
	*/

	/*
		FIXME: write some documentation about the implementation
	*/

	auto & dstAddress = locationSet_.Find(node.input(0)->origin());
	auto & srcAddress = locationSet_.Find(node.input(1)->origin());

	if (srcAddress.GetPointsTo() == nullptr) {
		/*
			If we do not know where the source address points to yet(!),
			insert a dummy location so we have something to work with.
		*/
		auto & dummyLocation = locationSet_.InsertDummyLocation();
    srcAddress.SetPointsTo(dummyLocation);
	}

	if (dstAddress.GetPointsTo() == nullptr) {
		/*
			If we do not know where the destination address points to yet(!),
			insert a dummy location so we have somehting to work with.
		*/
		auto & dummyLocation = locationSet_.InsertDummyLocation();
    dstAddress.SetPointsTo(dummyLocation);
	}

	auto & srcMemory = locationSet_.GetRootLocation(*srcAddress.GetPointsTo());
	auto & dstMemory = locationSet_.GetRootLocation(*dstAddress.GetPointsTo());

	if (srcMemory.GetPointsTo() == nullptr) {
		auto & dummyLocation = locationSet_.InsertDummyLocation();
    srcMemory.SetPointsTo(dummyLocation);
	}

	if (dstMemory.GetPointsTo() == nullptr) {
		auto & dummyLocation = locationSet_.InsertDummyLocation();
    dstMemory.SetPointsTo(dummyLocation);
	}

	join(*srcMemory.GetPointsTo(), *dstMemory.GetPointsTo());
}

void
Steensgaard::Analyze(const lambda::node & lambda)
{
	if (lambda.direct_calls()) {
		/* handle context variables */
		for (auto & cv : lambda.ctxvars()) {
			if (!jive::is<ptrtype>(cv.type()))
				continue;

			auto & origin = locationSet_.Find(cv.origin());
			auto & argument = locationSet_.FindOrInsertRegisterLocation(cv.argument(), false, false);
			join(origin, argument);
		}

		/* handle function arguments */
		for (auto & argument : lambda.fctarguments()) {
			if (!jive::is<ptrtype>(argument.type()))
				continue;

      locationSet_.FindOrInsertRegisterLocation(&argument, false, false);
		}

		Analyze(*lambda.subregion());

		auto & lambdaOutputLocation = locationSet_.FindOrInsertRegisterLocation(lambda.output(), false, false);
    auto & lambdaLocation = locationSet_.InsertLambdaLocation(lambda);
    lambdaOutputLocation.SetPointsTo(lambdaLocation);
	} else {
		/* handle context variables */
		for (auto & cv : lambda.ctxvars()) {
			if (!jive::is<ptrtype>(cv.type()))
				continue;

			auto & origin = locationSet_.Find(cv.origin());
			auto & argument = locationSet_.FindOrInsertRegisterLocation(cv.argument(), false, false);
			join(origin, argument);
		}

		/* handle function arguments */
		for (auto & argument : lambda.fctarguments()) {
			if (!jive::is<ptrtype>(argument.type()))
				continue;

      locationSet_.FindOrInsertRegisterLocation(&argument, true, true);
		}

		Analyze(*lambda.subregion());

		auto & lambdaOutputLocation = locationSet_.FindOrInsertRegisterLocation(lambda.output(), false, false);
    auto & lambdaLocation = locationSet_.InsertLambdaLocation(lambda);
    lambdaOutputLocation.SetPointsTo(lambdaLocation);
	}
}

void
Steensgaard::Analyze(const delta::node & delta)
{
	/*
		Handle context variables
	*/
	for (auto & input : delta.ctxvars()) {
		if (!is<ptrtype>(input.type()))
			continue;

		auto & origin = locationSet_.Find(input.origin());
		auto & argument = locationSet_.FindOrInsertRegisterLocation(input.arguments.first(), false, false);
		join(origin, argument);
	}

	Analyze(*delta.subregion());

	auto & deltaOutputLocation = locationSet_.FindOrInsertRegisterLocation(delta.output(), false, false);
	auto & deltaLocation = locationSet_.InsertDeltaLocation(delta);
  deltaOutputLocation.SetPointsTo(deltaLocation);

	auto origin = delta.result()->origin();
	if (locationSet_.contains(origin)) {
		auto & resultLocation = locationSet_.Find(origin);
		join(deltaLocation, resultLocation);
	}
}

void
Steensgaard::Analyze(const phi::node & phi)
{
	/* handle context variables */
	for (auto cv = phi.begin_cv(); cv != phi.end_cv(); cv++) {
		if (!is<ptrtype>(cv->type()))
			continue;

		auto & origin = locationSet_.Find(cv->origin());
		auto & argument = locationSet_.FindOrInsertRegisterLocation(cv->argument(), false, false);
		join(origin, argument);
	}

	/* handle recursion variable arguments */
	for (auto rv = phi.begin_rv(); rv != phi.end_rv(); rv++) {
		if (!is<ptrtype>(rv->type()))
			continue;

    locationSet_.FindOrInsertRegisterLocation(rv->argument(), false, false);
	}

	Analyze(*phi.subregion());

	/* handle recursion variable outputs */
	for (auto rv = phi.begin_rv(); rv != phi.end_rv(); rv++) {
		if (!is<ptrtype>(rv->type()))
			continue;

		auto & origin = locationSet_.Find(rv->result()->origin());
		auto & argument = locationSet_.Find(rv->argument());
		join(origin, argument);

		auto & output = locationSet_.FindOrInsertRegisterLocation(rv.output(), false, false);
		join(argument, output);
	}
}

void
Steensgaard::Analyze(const jive::gamma_node & node)
{
	/* handle entry variables */
	for (auto ev = node.begin_entryvar(); ev != node.end_entryvar(); ev++) {
		if (!jive::is<ptrtype>(ev->type()))
			continue;

		auto & originloc = locationSet_.Find(ev->origin());
		for (auto & argument : *ev) {
			auto & argumentloc = locationSet_.FindOrInsertRegisterLocation(&argument, false, false);
			join(argumentloc, originloc);
		}
	}

	/* handle subregions */
	for (size_t n = 0; n < node.nsubregions(); n++)
		Analyze(*node.subregion(n));

	/* handle exit variables */
	for (auto ex = node.begin_exitvar(); ex != node.end_exitvar(); ex++) {
		if (!jive::is<ptrtype>(ex->type()))
			continue;

		auto & outputloc = locationSet_.FindOrInsertRegisterLocation(ex.output(), false, false);
		for (auto & result : *ex) {
			auto & resultloc = locationSet_.Find(result.origin());
			join(outputloc, resultloc);
		}
	}
}

void
Steensgaard::Analyze(const jive::theta_node & theta)
{
	for (auto lv : theta) {
		if (!jive::is<ptrtype>(lv->type()))
			continue;

		auto & originloc = locationSet_.Find(lv->input()->origin());
		auto & argumentLocation = locationSet_.FindOrInsertRegisterLocation(lv->argument(), false, false);

		join(argumentLocation, originloc);
	}

	Analyze(*theta.subregion());

	for (auto lv : theta) {
		if (!jive::is<ptrtype>(lv->type()))
			continue;

		auto & originloc = locationSet_.Find(lv->result()->origin());
		auto & argumentloc = locationSet_.Find(lv->argument());
		auto & outputLocation = locationSet_.FindOrInsertRegisterLocation(lv, false, false);

		join(originloc, argumentloc);
		join(originloc, outputLocation);
	}
}

void
Steensgaard::Analyze(const jive::structural_node & node)
{
	auto analyzeLambda = [](auto& s, auto& n){s.Analyze(*static_cast<const lambda::node*>(&n));    };
	auto analyzeDelta  = [](auto& s, auto& n){s.Analyze(*static_cast<const delta::node*>(&n));     };
	auto analyzeGamma  = [](auto& s, auto& n){s.Analyze(*static_cast<const jive::gamma_node*>(&n));};
	auto analyzeTheta  = [](auto& s, auto& n){s.Analyze(*static_cast<const jive::theta_node*>(&n));};
	auto analyzePhi    = [](auto& s, auto& n){s.Analyze(*static_cast<const phi::node*>(&n));       };

	static std::unordered_map<
		std::type_index
	, std::function<void(Steensgaard&, const jive::structural_node&)>> nodes
	({
	  {typeid(lambda::operation), analyzeLambda }
	, {typeid(delta::operation),  analyzeDelta  }
	, {typeid(jive::gamma_op),    analyzeGamma  }
	, {typeid(jive::theta_op),    analyzeTheta  }
	, {typeid(phi::operation),    analyzePhi    }
	});

	auto & op = node.operation();
	JLM_ASSERT(nodes.find(typeid(op)) != nodes.end());
	nodes[typeid(op)](*this, node);
}

void
Steensgaard::Analyze(jive::region & region)
{
	using namespace jive;

	topdown_traverser traverser(&region);
	for (auto & node : traverser) {
		if (auto smpnode = dynamic_cast<const simple_node*>(node)) {
			Analyze(*smpnode);
			continue;
		}

		JLM_ASSERT(is<structural_op>(node));
		auto structnode = static_cast<const structural_node*>(node);
		Analyze(*structnode);
	}
}

void
Steensgaard::Analyze(const jive::graph & graph)
{
	auto add_imports = [](const jive::graph & graph, LocationSet & lset)
	{
		auto region = graph.root();
		for (size_t n = 0; n < region->narguments(); n++) {
			auto argument = region->argument(n);
			if (!jive::is<ptrtype>(argument->type()))
				continue;
			/* FIXME: we should not add function imports */
			auto & imploc = lset.InsertImportLocation(argument);
			auto & ptr = lset.FindOrInsertRegisterLocation(argument, false, false);
      ptr.SetPointsTo(imploc);
		}
	};

	add_imports(graph, locationSet_);
	Analyze(*graph.root());
}

std::unique_ptr<PointsToGraph>
Steensgaard::Analyze(
  const RvsdgModule & module,
  const StatisticsDescriptor & sd)
{
	ResetState();

  /**
   * Perform Steensgaard analysis
   */
  SteensgaardAnalysisStatistics steensgardStatistics(module.SourceFileName());
  steensgardStatistics.start(module.Rvsdg());
	Analyze(module.Rvsdg());
//	std::cout << locationSet_.to_dot() << std::flush;
  steensgardStatistics.end();
  sd.PrintStatistics(steensgardStatistics);


  /**
   * Construct PointsTo graph
   */
  SteensgaardPointsToGraphConstructionStatistics ptgConstructionStatistics(module.SourceFileName());
  ptgConstructionStatistics.start(locationSet_);
	auto pointsToGraph = ConstructPointsToGraph(locationSet_);
//	std::cout << PointsToGraph::ToDot(*pointsToGraph) << std::flush;
  ptgConstructionStatistics.stop(*pointsToGraph);
  sd.PrintStatistics(ptgConstructionStatistics);

	return pointsToGraph;
}

std::unique_ptr<PointsToGraph>
Steensgaard::ConstructPointsToGraph(const LocationSet & lset)
{
	auto ptg = PointsToGraph::Create();

	/*
		Create points-to graph nodes
	*/
	std::vector<PointsToGraph::MemoryNode*> memNodes;
	std::unordered_map<Location*, PointsToGraph::Node*> map;
	std::unordered_map<const disjointset<Location*>::set*, std::vector<PointsToGraph::MemoryNode*>> allocators;
	for (auto & set : lset) {
		for (auto & loc : set) {
			if (auto regloc = dynamic_cast<jlm::aa::RegisterLocation*>(loc)) {
				map[loc] = &PointsToGraph::RegisterNode::create(*ptg, regloc->output());
				continue;
			}

      if (auto allocaLocation = dynamic_cast<AllocaLocation*>(loc)) {
        auto node = &PointsToGraph::AllocatorNode::create(*ptg, &allocaLocation->Node());
        allocators[&set].push_back(node);
        memNodes.push_back(node);
        map[loc] = node;
        continue;
      }

      if (auto mallocLocation = dynamic_cast<MallocLocation*>(loc)) {
        auto node = &PointsToGraph::AllocatorNode::create(*ptg, &mallocLocation->Node());
        allocators[&set].push_back(node);
        memNodes.push_back(node);
        map[loc] = node;
        continue;
      }

      if (auto lambdaLocation = dynamic_cast<LambdaLocation*>(loc)) {
        auto node = &PointsToGraph::AllocatorNode::create(*ptg, &lambdaLocation->Node());
        allocators[&set].push_back(node);
        memNodes.push_back(node);
        map[loc] = node;
        continue;
      }

      if (auto deltaLocation = dynamic_cast<DeltaLocation*>(loc)) {
        auto node = &PointsToGraph::AllocatorNode::create(*ptg, &deltaLocation->Node());
        allocators[&set].push_back(node);
        memNodes.push_back(node);
        map[loc] = node;
        continue;
      }

			if (auto l = dynamic_cast<ImportLocation*>(loc)) {
				auto node = &PointsToGraph::ImportNode::create(*ptg, l->argument());
				allocators[&set].push_back(node);
				memNodes.push_back(node);
				map[loc] = node;
				continue;
			}

			if (dynamic_cast<DummyLocation*>(loc)) {
				continue;
			}

			JLM_UNREACHABLE("Unhandled location node.");
		}
	}

	/*
		Create points-to graph edges
	*/
	for (auto & set : lset) {
		bool pointsToUnknown = lset.set(**set.begin()).value()->unknown();
    bool pointsToExternalMemory = lset.set(**set.begin()).value()->PointsToExternalMemory();

		for (auto & loc : set) {
			if (dynamic_cast<DummyLocation*>(loc))
				continue;

			if (pointsToUnknown) {
				map[loc]->add_edge(ptg->memunknown());
			}

      if (pointsToExternalMemory) {
        map[loc]->add_edge(ptg->GetExternalMemoryNode());
      }

			auto pt = set.value()->GetPointsTo();
			if (pt == nullptr) {
				map[loc]->add_edge(ptg->memunknown());
				continue;
			}

			auto & ptset = lset.set(*pt);
			auto & memoryNodes = allocators[&ptset];
			if (memoryNodes.empty()) {
				/*
					The location points to a pointsTo set that contains
					no memory nodes. Thus, we have no idea where this pointer
					points to. Let's be conservative and let it just point to
					unknown.
				*/
				map[loc]->add_edge(ptg->memunknown());
				continue;
			}

			for (auto & allocator : allocators[&ptset])
				map[loc]->add_edge(*allocator);
		}
	}

	return ptg;
}

void
Steensgaard::ResetState()
{
	locationSet_.clear();
}

}
