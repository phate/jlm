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
    , NumNodesBefore_(0)
    , SourceFile_(std::move(sourceFile))
  {}

  void
  Start(const jive::graph & graph) noexcept
  {
    NumNodesBefore_ = jive::nnodes(graph.root());
    Timer_.start();
  }

  void
  Stop() noexcept
  {
    Timer_.stop();
  }

  [[nodiscard]] std::string
  ToString() const override
  {
    return strfmt("SteensgaardAnalysis ",
                  SourceFile_.to_str(), " ",
                  "#RvsdgNodes:", NumNodesBefore_, " ",
                  "Time[ns]:", Timer_.ns());
  }

private:
  size_t NumNodesBefore_;
  jlm::filepath SourceFile_;

  jlm::timer Timer_;
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
    , SourceFile_(std::move(sourceFile))
    , NumDisjointSets_(0)
    , NumLocations_(0)
    , NumNodes_(0)
    , NumAllocaNodes_(0)
    , NumDeltaNodes_(0)
    , NumImportNodes_(0)
    , NumLambdaNodes_(0)
    , NumMallocNodes_(0)
    , NumMemoryNodes_(0)
    , NumRegisterNodes_(0)
    , NumUnknownMemorySources_(0)
  {}

  void
  Start(const LocationSet & locationSet)
  {
    NumDisjointSets_ = locationSet.NumDisjointSets();
    NumLocations_ = locationSet.NumLocations();
    Timer_.start();
  }

  void
  Stop(const PointsToGraph & pointsToGraph)
  {
    Timer_.stop();
    NumNodes_ = pointsToGraph.NumNodes();
    NumAllocaNodes_ = pointsToGraph.NumAllocaNodes();
    NumDeltaNodes_ = pointsToGraph.NumDeltaNodes();
    NumImportNodes_ = pointsToGraph.NumImportNodes();
    NumLambdaNodes_ = pointsToGraph.NumLambdaNodes();
    NumMallocNodes_ = pointsToGraph.NumMallocNodes();
    NumMemoryNodes_ = pointsToGraph.NumMemoryNodes();
    NumRegisterNodes_ = pointsToGraph.NumRegisterNodes();
    NumUnknownMemorySources_ = pointsToGraph.GetUnknownMemoryNode().NumSources();
  }

  [[nodiscard]] std::string
  ToString() const override
  {
    return strfmt("SteensgaardPointsToGraphConstruction ",
                  SourceFile_.to_str(), " ",
                  "#DisjointSets:", NumDisjointSets_, " ",
                  "#Locations:", NumLocations_, " ",
                  "#Nodes:", NumNodes_, " ",
                  "#AllocaNodes:", NumAllocaNodes_, " ",
                  "#DeltaNodes:", NumDeltaNodes_, " ",
                  "#ImportNodes:", NumImportNodes_, " ",
                  "#LambdaNodes:", NumLambdaNodes_, " ",
                  "#MallocNodes:", NumMallocNodes_, " ",
                  "#MemoryNodes:", NumMemoryNodes_, " ",
                  "#RegisterNodes:", NumRegisterNodes_, " ",
                  "#UnknownMemorySources:", NumUnknownMemorySources_, " ",
                  "Time[ns]:", Timer_.ns());
  }

private:
  jlm::filepath SourceFile_;

  size_t NumDisjointSets_;
  size_t NumLocations_;

  size_t NumNodes_;
  size_t NumAllocaNodes_;
  size_t NumDeltaNodes_;
  size_t NumImportNodes_;
  size_t NumLambdaNodes_;
  size_t NumMallocNodes_;
  size_t NumMemoryNodes_;
  size_t NumRegisterNodes_;
  size_t NumUnknownMemorySources_;
  jlm::timer Timer_;
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
    bool pointsToUnknownMemory,
    bool pointsToExternalMemory)
    : PointsToUnknownMemory_(pointsToUnknownMemory)
    , PointsToExternalMemory_(pointsToExternalMemory)
    , PointsTo_(nullptr)
  {}

  Location(const Location &) = delete;

  Location(Location &&) = delete;

  Location &
  operator=(const Location &) = delete;

  Location &
  operator=(Location &&) = delete;

  [[nodiscard]] virtual std::string
  DebugString() const noexcept = 0;

  [[nodiscard]] bool
  PointsToUnknownMemory() const noexcept
  {
    return PointsToUnknownMemory_;
  }

  [[nodiscard]] bool
  PointsToExternalMemory() const noexcept
  {
    return PointsToExternalMemory_;
  }

  [[nodiscard]] Location *
  GetPointsTo() const noexcept
  {
    return PointsTo_;
  }

  void
  SetPointsTo(Location & location) noexcept
  {
    PointsTo_ = &location;
  }

  void
  SetPointsToUnknownMemory(bool pointsToUnknownMemory) noexcept
  {
    PointsToUnknownMemory_ = pointsToUnknownMemory;
  }

  void
  SetPointsToExternalMemory(bool pointsToExternalMemory) noexcept
  {
    PointsToExternalMemory_ = pointsToExternalMemory;
  }

private:
  bool PointsToUnknownMemory_;
  bool PointsToExternalMemory_;
  Location * PointsTo_;
};

class RegisterLocation final : public Location {
public:
  constexpr explicit
  RegisterLocation(
    const jive::output & output,
    bool pointsToUnknownMemory = false,
    bool pointsToExternalMemory = false)
    : Location(pointsToUnknownMemory, pointsToExternalMemory)
    , Output_(&output)
  {}

  [[nodiscard]] const jive::output &
  GetOutput() const noexcept
  {
    return *Output_;
  }

  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    auto node = jive::node_output::node(Output_);
    auto index = Output_->index();

    if (jive::is<jive::simple_op>(node)) {
      auto nodestr = node->operation().debug_string();
      auto outputstr = Output_->type().debug_string();
      return strfmt(nodestr, ":", index, "[" + outputstr + "]");
    }

    if (is<lambda::cvargument>(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return strfmt(dbgstr, ":cv:", index);
    }

    if (is<lambda::fctargument>(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return strfmt(dbgstr, ":arg:", index);
    }

    if (is<delta::cvargument>(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return strfmt(dbgstr, ":cv:", index);
    }

    if (is_gamma_argument(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return strfmt(dbgstr, ":arg", index);
    }

    if (is_theta_argument(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return strfmt(dbgstr, ":arg", index);
    }

    if (is_theta_output(Output_)) {
      auto dbgstr = jive::node_output::node(Output_)->operation().debug_string();
      return strfmt(dbgstr, ":out", index);
    }

    if (is_gamma_output(Output_)) {
      auto dbgstr = jive::node_output::node(Output_)->operation().debug_string();
      return strfmt(dbgstr, ":out", index);
    }

    if (is_import(Output_)) {
      auto import = AssertedCast<const jive::impport>(&Output_->port());
      return strfmt("imp:", import->name());
    }

    if (is<phi::rvargument>(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return strfmt(dbgstr, ":rvarg", index);
    }

    if (is<phi::cvargument>(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return strfmt(dbgstr, ":cvarg", index);
    }

    return strfmt(jive::node_output::node(Output_)->operation().debug_string(), ":", index);
  }

  static std::unique_ptr<Location>
  Create(
    const jive::output & output,
    bool pointsToUnknownMemory,
    bool pointsToExternalMemory)
  {
    return std::make_unique<RegisterLocation>(output, pointsToUnknownMemory, pointsToExternalMemory);
  }

private:
  const jive::output * Output_;
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
    , Node_(node)
  {
    JLM_ASSERT(is<alloca_op>(&node));
  }

public:
  [[nodiscard]] const jive::node &
  GetNode() const noexcept
  {
    return Node_;
  }

  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    return Node_.operation().debug_string();
  }

  static std::unique_ptr<Location>
  Create(const jive::node & node)
  {
    return std::unique_ptr<Location>(new AllocaLocation(node));
  }

private:
  const jive::node & Node_;
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
    , Node_(node)
  {
    JLM_ASSERT(is<malloc_op>(&node));
  }

public:
  [[nodiscard]] const jive::node &
  GetNode() const noexcept
  {
    return Node_;
  }

  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    return Node_.operation().debug_string();
  }

  static std::unique_ptr<Location>
  Create(const jive::node & node)
  {
    return std::unique_ptr<Location>(new MallocLocation(node));
  }

private:
  const jive::node & Node_;
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
    , Lambda_(lambda)
  {}

public:
  [[nodiscard]] const lambda::node &
  GetNode() const noexcept
  {
    return Lambda_;
  }

  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    return Lambda_.operation().debug_string();
  }

  static std::unique_ptr<Location>
  Create(const lambda::node & node)
  {
    return std::unique_ptr<Location>(new LambdaLocation(node));
  }

private:
  const lambda::node & Lambda_;
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
    , Delta_(delta)
  {}

public:
  [[nodiscard]] const delta::node &
  GetNode() const noexcept
  {
    return Delta_;
  }

  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    return Delta_.operation().debug_string();
  }

  static std::unique_ptr<Location>
  Create(const delta::node & node)
  {
    return std::unique_ptr<Location>(new DeltaLocation(node));
  }

private:
  const delta::node & Delta_;
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
    const jive::argument & argument,
    bool pointsToUnknownMemory,
    bool pointsToExternalMemory)
    : Location(pointsToUnknownMemory, pointsToExternalMemory)
    , Argument_(argument)
  {
    JLM_ASSERT(dynamic_cast<const jlm::impport*>(&argument.port()));
  }

  [[nodiscard]] const jive::argument &
  GetArgument() const noexcept
  {
    return Argument_;
  }

  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    return "IMPORT[" + Argument_.debug_string() + "]";
  }

  static std::unique_ptr<Location>
  Create(const jive::argument & argument)
  {
    auto pointerType = AssertedCast<const PointerType>(&argument.type());

    bool pointsToUnknownMemory = is<PointerType>(pointerType->GetElementType());
    /**
     * FIXME: We use pointsToUnknownMemory for pointsToExternalMemory
     */
    return std::unique_ptr<Location>(
      new ImportLocation(argument, pointsToUnknownMemory, pointsToUnknownMemory));
  }

private:
  const jive::argument & Argument_;
};

/** \brief FIXME: write documentation
*/
class DummyLocation final : public Location {
public:
  ~DummyLocation() override = default;

  DummyLocation()
    : Location(false, false)
  {}

  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    return "UNNAMED";
  }

  static std::unique_ptr<Location>
  Create()
  {
    return std::make_unique<DummyLocation>();
  }
};

LocationSet::~LocationSet()
= default;

LocationSet::LocationSet()
= default;

void
LocationSet::Clear()
{
  LocationMap_.clear();
  DisjointLocationSet_.clear();
  Locations_.clear();
}

Location &
LocationSet::InsertRegisterLocation(
  const jive::output & output,
  bool pointsToUnknownMemory,
  bool pointsToExternalMemory)
{
  JLM_ASSERT(!Contains(output));

  Locations_.push_back(RegisterLocation::Create(output, pointsToUnknownMemory, pointsToExternalMemory));
  auto location = Locations_.back().get();

  LocationMap_[&output] = location;
  DisjointLocationSet_.insert(location);

  return *location;
}

Location &
LocationSet::InsertAllocaLocation(const jive::node & node)
{
  Locations_.push_back(AllocaLocation::Create(node));
  auto location = Locations_.back().get();
  DisjointLocationSet_.insert(location);

  return *location;
}

Location &
LocationSet::InsertMallocLocation(const jive::node & node)
{
  Locations_.push_back(MallocLocation::Create(node));
  auto location = Locations_.back().get();
  DisjointLocationSet_.insert(location);

  return *location;
}

Location &
LocationSet::InsertLambdaLocation(const lambda::node & node)
{
  Locations_.push_back(LambdaLocation::Create(node));
  auto location = Locations_.back().get();
  DisjointLocationSet_.insert(location);

  return *location;
}

Location &
LocationSet::InsertDeltaLocation(const delta::node & node)
{
  Locations_.push_back(DeltaLocation::Create(node));
  auto location = Locations_.back().get();
  DisjointLocationSet_.insert(location);

  return *location;
}

Location &
LocationSet::InsertDummyLocation()
{
  Locations_.push_back(DummyLocation::Create());
  auto location = Locations_.back().get();
  DisjointLocationSet_.insert(location);

  return *location;
}

Location &
LocationSet::InsertImportLocation(const jive::argument & argument)
{
  Locations_.push_back(ImportLocation::Create(argument));
  auto location = Locations_.back().get();
  DisjointLocationSet_.insert(location);

  return *location;
}

Location *
LocationSet::Lookup(const jive::output & output)
{
  auto it = LocationMap_.find(&output);
  return it == LocationMap_.end() ? nullptr : it->second;
}

bool
LocationSet::Contains(const jive::output & output) const noexcept
{
  return LocationMap_.find(&output) != LocationMap_.end();
}

Location &
LocationSet::FindOrInsertRegisterLocation(
  const jive::output & output,
  bool pointsToUnknownMemory,
  bool pointsToExternalMemory)
{
  if (auto location = Lookup(output))
    return GetRootLocation(*location);

  return InsertRegisterLocation(output, pointsToUnknownMemory, pointsToExternalMemory);
}

Location &
LocationSet::GetRootLocation(Location & l) const
{
  return *GetSet(l).value();
}

Location &
LocationSet::Find(const jive::output & output)
{
  auto location = Lookup(output);
  JLM_ASSERT(location != nullptr);

  return GetRootLocation(*location);
}

Location &
LocationSet::Merge(Location & location1, Location & location2)
{
  return *DisjointLocationSet_.merge(&location1, &location2)->value();
}

std::string
LocationSet::ToDot() const
{
  auto dot_node = [](const DisjointLocationSet::set & set)
  {
    auto root = set.value();

    std::string label;
    for (auto & l : set) {
      auto unknownstr = l->PointsToUnknownMemory() ? "{U}" : "";
      auto ptstr = strfmt("{pt:", (intptr_t) l->GetPointsTo(), "}");
      auto locstr = strfmt((intptr_t)l, " : ", l->DebugString());

      if (l == root) {
        label += strfmt("*", locstr, unknownstr, ptstr, "*\\n");
      } else {
        label += strfmt(locstr, "\\n");
      }
    }

    return strfmt("{ ", (intptr_t)&set, " [label = \"", label, "\"]; }");
  };

  auto dot_edge = [&](const DisjointLocationSet::set & set, const DisjointLocationSet::set & pointsToSet)
  {
    return strfmt((intptr_t)&set, " -> ", (intptr_t)&pointsToSet);
  };

  std::string str;
  str.append("digraph PointsToGraph {\n");

  for (auto & set : DisjointLocationSet_) {
    str += dot_node(set) + "\n";

    auto pointsTo = set.value()->GetPointsTo();
    if (pointsTo != nullptr) {
      auto pointsToSet = DisjointLocationSet_.find(pointsTo);
      str += dot_edge(set, *pointsToSet) + "\n";
    }
  }

  str.append("}\n");

  return str;
}

Steensgaard::~Steensgaard()
= default;

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

    auto & rootx = LocationSet_.GetRootLocation(*x);
    auto & rooty = LocationSet_.GetRootLocation(*y);
    rootx.SetPointsToExternalMemory(rootx.PointsToExternalMemory() || rooty.PointsToExternalMemory());
    rooty.SetPointsToExternalMemory(rootx.PointsToExternalMemory() || rooty.PointsToExternalMemory());
    rootx.SetPointsToUnknownMemory(rootx.PointsToUnknownMemory() || rooty.PointsToUnknownMemory());
    rooty.SetPointsToUnknownMemory(rootx.PointsToUnknownMemory() || rooty.PointsToUnknownMemory());
    auto & tmp = LocationSet_.Merge(rootx, rooty);

    if (auto root = join(rootx.GetPointsTo(), rooty.GetPointsTo()))
      tmp.SetPointsTo(*root);

    return &tmp;
  };

  join(&x, &y);
}

void
Steensgaard::Analyze(const jive::simple_node & node)
{
  auto AnalyzeCall  = [](auto & s, auto & n) { s.AnalyzeCall(*AssertedCast<const CallNode>(&n)); };
  auto AnalyzeLoad  = [](auto & s, auto & n) { s.AnalyzeLoad(*AssertedCast<const LoadNode>(&n)); };
  auto AnalyzeStore = [](auto & s, auto & n) { s.AnalyzeStore(*AssertedCast<const StoreNode>(&n)); };

  static std::unordered_map<
    std::type_index
    , std::function<void(Steensgaard&, const jive::simple_node&)>> nodes
    ({
         {typeid(alloca_op),                    [](auto & s, auto & n){ s.AnalyzeAlloca(n);                }}
       , {typeid(malloc_op),                    [](auto & s, auto & n){ s.AnalyzeMalloc(n);                }}
       , {typeid(LoadOperation),                AnalyzeLoad                                                 }
       , {typeid(StoreOperation),               AnalyzeStore                                                }
       , {typeid(CallOperation),                AnalyzeCall                                                 }
       , {typeid(getelementptr_op),             [](auto & s, auto & n){ s.AnalyzeGep(n);                   }}
       , {typeid(bitcast_op),                   [](auto & s, auto & n){ s.AnalyzeBitcast(n);               }}
       , {typeid(bits2ptr_op),                  [](auto & s, auto & n){ s.AnalyzeBits2ptr(n);              }}
       , {typeid(ConstantPointerNullOperation), [](auto & s, auto & n){ s.AnalyzeConstantPointerNull(n);   }}
       , {typeid(UndefValueOperation),          [](auto & s, auto & n){ s.AnalyzeUndef(n);                 }}
       , {typeid(Memcpy),                       [](auto & s, auto & n){ s.AnalyzeMemcpy(n);                }}
       , {typeid(ConstantArray),                [](auto & s, auto & n){ s.AnalyzeConstantArray(n);         }}
       , {typeid(ConstantStruct),               [](auto & s, auto & n){ s.AnalyzeConstantStruct(n);        }}
       , {typeid(ConstantAggregateZero),        [](auto & s, auto & n){ s.AnalyzeConstantAggregateZero(n); }}
       , {typeid(ExtractValue),                 [](auto & s, auto & n){ s.AnalyzeExtractValue(n);          }}
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
    if (jive::is<PointerType>(node.output(n)->type()))
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

  auto & allocaOutputLocation = LocationSet_.FindOrInsertRegisterLocation(*node.output(0), false, false);
  auto & allocaLocation = LocationSet_.InsertAllocaLocation(node);
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
    allocaLocation.SetPointsToUnknownMemory(true);
  }
}

void
Steensgaard::AnalyzeMalloc(const jive::simple_node & node)
{
  JLM_ASSERT(is<malloc_op>(&node));

  auto & mallocOutputLocation = LocationSet_.FindOrInsertRegisterLocation(*node.output(0), false, false);
  auto & mallocLocation = LocationSet_.InsertMallocLocation(node);
  mallocOutputLocation.SetPointsTo(mallocLocation);
}

void
Steensgaard::AnalyzeLoad(const LoadNode & loadNode)
{
  if (!is<PointerType>(loadNode.GetValueOutput()->type()))
    return;

  auto & address = LocationSet_.Find(*loadNode.GetAddressInput()->origin());
  auto & result = LocationSet_.FindOrInsertRegisterLocation(
    *loadNode.GetValueOutput(),
    address.PointsToUnknownMemory(),
    address.PointsToExternalMemory());

  if (address.GetPointsTo() == nullptr) {
    address.SetPointsTo(result);
    return;
  }

  join(result, *address.GetPointsTo());
}

void
Steensgaard::AnalyzeStore(const StoreNode & storeNode)
{
  auto & address = *storeNode.GetAddressInput()->origin();
  auto & value = *storeNode.GetValueInput()->origin();

  if (!is<PointerType>(value.type()))
    return;

  auto & addressLocation = LocationSet_.Find(address);
  auto & valueLocation = LocationSet_.Find(value);

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
      auto & callArgument = *call.input(n)->origin();
      auto & lambdaArgument = *lambda.fctargument(n-1);

      if (!is<PointerType>(callArgument.type()))
        continue;

      auto & callArgumentLocation = LocationSet_.Find(callArgument);
      auto & lambdaArgumentLocation = LocationSet_.Contains(lambdaArgument)
                                      ? LocationSet_.Find(lambdaArgument)
                                      : LocationSet_.FindOrInsertRegisterLocation(lambdaArgument, false, false);

      join(callArgumentLocation, lambdaArgumentLocation);
    }

    /* handle call node results */
    auto subregion = lambda.subregion();
    JLM_ASSERT(subregion->nresults() == callNode.noutputs());
    for (size_t n = 0; n < call.noutputs(); n++) {
      auto & callResult = *call.output(n);
      auto & lambdaResult = *subregion->result(n)->origin();

      if (!is<PointerType>(callResult.type()))
        continue;

      auto & callResultLocation = LocationSet_.FindOrInsertRegisterLocation(callResult, false, false);
      auto & lambdaResultLocation = LocationSet_.Contains(lambdaResult)
                                    ? LocationSet_.Find(lambdaResult)
                                    : LocationSet_.FindOrInsertRegisterLocation(lambdaResult, false, false);

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
      auto & callResult = *call.output(n);
      if (!is<PointerType>(callResult.type()))
        continue;

      LocationSet_.FindOrInsertRegisterLocation(callResult, true, true);
    }
  };

  auto callTypeClassifier = CallNode::ClassifyCall(callNode);
    if (callTypeClassifier->GetCallType() == CallTypeClassifier::CallType::DirectCall) {
    handle_direct_call(callNode, *callTypeClassifier->GetLambdaOutput().node());
    return;
  }

  handle_indirect_call(callNode);
}

void
Steensgaard::AnalyzeGep(const jive::simple_node & node)
{
  JLM_ASSERT(is<getelementptr_op>(&node));

  auto & base = LocationSet_.Find(*node.input(0)->origin());
  auto & value = LocationSet_.FindOrInsertRegisterLocation(*node.output(0), false, false);

  join(base, value);
}

void
Steensgaard::AnalyzeBitcast(const jive::simple_node & node)
{
  JLM_ASSERT(is<bitcast_op>(&node));

  auto input = node.input(0);
  if (!is<PointerType>(input->type()))
    return;

  auto & operand = LocationSet_.Find(*input->origin());
  auto & result = LocationSet_.FindOrInsertRegisterLocation(*node.output(0), false, false);

  join(operand, result);
}

void
Steensgaard::AnalyzeBits2ptr(const jive::simple_node & node)
{
  JLM_ASSERT(is<bits2ptr_op>(&node));

  LocationSet_.FindOrInsertRegisterLocation(*node.output(0), true, true);
}

void
Steensgaard::AnalyzeExtractValue(const jive::simple_node & node)
{
  JLM_ASSERT(is<ExtractValue>(&node));

  auto & result = *node.output(0);
  if (!is<PointerType>(result.type()))
    return;

  LocationSet_.FindOrInsertRegisterLocation(result, true, true);
}

void
Steensgaard::AnalyzeConstantPointerNull(const jive::simple_node & node)
{
  JLM_ASSERT(is<ConstantPointerNullOperation>(&node));

  /*
   * ConstantPointerNull cannot point to any memory location. We therefore only insert a register node for it,
   * but let this node not point to anything.
   */
  LocationSet_.FindOrInsertRegisterLocation(*node.output(0), false, false);
}

void
Steensgaard::AnalyzeConstantAggregateZero(const jive::simple_node & node)
{
  JLM_ASSERT(is<ConstantAggregateZero>(&node));
  auto output = node.output(0);

  if (!IsOrContains<PointerType>(output->type()))
    return;

  /*
   * ConstantAggregateZero cannot point to any memory location. We therefore only insert a register node for it,
   * but let this node not point to anything.
   */
  LocationSet_.FindOrInsertRegisterLocation(*output, false, false);
}

void
Steensgaard::AnalyzeUndef(const jive::simple_node & node)
{
  JLM_ASSERT(is<UndefValueOperation>(&node));
  auto output = node.output(0);

  if (!is<PointerType>(output->type()))
    return;

  /*
   * UndefValue cannot point to any memory location. We therefore only insert a register node for it,
   * but let this node not point to anything.
   */
  LocationSet_.FindOrInsertRegisterLocation(*output, false, false);
}

void
Steensgaard::AnalyzeConstantArray(const jive::simple_node & node)
{
  JLM_ASSERT(is<ConstantArray>(&node));

  for (size_t n = 0; n < node.ninputs(); n++) {
    auto input = node.input(n);

    if (LocationSet_.Contains(*input->origin())) {
      auto & originLocation = LocationSet_.Find(*input->origin());
      auto & outputLocation = LocationSet_.FindOrInsertRegisterLocation(*node.output(0), false, false);
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

    if (LocationSet_.Contains(*input->origin())) {
      auto & originLocation = LocationSet_.Find(*input->origin());
      auto & outputLocation = LocationSet_.FindOrInsertRegisterLocation(*node.output(0), false, false);
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

  auto & dstAddress = LocationSet_.Find(*node.input(0)->origin());
  auto & srcAddress = LocationSet_.Find(*node.input(1)->origin());

  if (srcAddress.GetPointsTo() == nullptr) {
    /*
      If we do not know where the source address points to yet(!),
      insert a dummy location so we have something to work with.
    */
    auto & dummyLocation = LocationSet_.InsertDummyLocation();
    srcAddress.SetPointsTo(dummyLocation);
  }

  if (dstAddress.GetPointsTo() == nullptr) {
    /*
      If we do not know where the destination address points to yet(!),
      insert a dummy location so we have somehting to work with.
    */
    auto & dummyLocation = LocationSet_.InsertDummyLocation();
    dstAddress.SetPointsTo(dummyLocation);
  }

  auto & srcMemory = LocationSet_.GetRootLocation(*srcAddress.GetPointsTo());
  auto & dstMemory = LocationSet_.GetRootLocation(*dstAddress.GetPointsTo());

  if (srcMemory.GetPointsTo() == nullptr) {
    auto & dummyLocation = LocationSet_.InsertDummyLocation();
    srcMemory.SetPointsTo(dummyLocation);
  }

  if (dstMemory.GetPointsTo() == nullptr) {
    auto & dummyLocation = LocationSet_.InsertDummyLocation();
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
      if (!jive::is<PointerType>(cv.type()))
        continue;

      auto & origin = LocationSet_.Find(*cv.origin());
      auto & argument = LocationSet_.FindOrInsertRegisterLocation(*cv.argument(), false, false);
      join(origin, argument);
    }

    /* handle function arguments */
    for (auto & argument : lambda.fctarguments()) {
      if (!jive::is<PointerType>(argument.type()))
        continue;

      LocationSet_.FindOrInsertRegisterLocation(argument, false, false);
    }

    Analyze(*lambda.subregion());

    auto & lambdaOutputLocation = LocationSet_.FindOrInsertRegisterLocation(*lambda.output(), false, false);
    auto & lambdaLocation = LocationSet_.InsertLambdaLocation(lambda);
    lambdaOutputLocation.SetPointsTo(lambdaLocation);
  } else {
    /* handle context variables */
    for (auto & cv : lambda.ctxvars()) {
      if (!jive::is<PointerType>(cv.type()))
        continue;

      auto & origin = LocationSet_.Find(*cv.origin());
      auto & argument = LocationSet_.FindOrInsertRegisterLocation(*cv.argument(), false, false);
      join(origin, argument);
    }

    /* handle function arguments */
    for (auto & argument : lambda.fctarguments()) {
      if (!jive::is<PointerType>(argument.type()))
        continue;

      LocationSet_.FindOrInsertRegisterLocation(argument, true, true);
    }

    Analyze(*lambda.subregion());

    auto & lambdaOutputLocation = LocationSet_.FindOrInsertRegisterLocation(*lambda.output(), false, false);
    auto & lambdaLocation = LocationSet_.InsertLambdaLocation(lambda);
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
    if (!is<PointerType>(input.type()))
      continue;

    auto & origin = LocationSet_.Find(*input.origin());
    auto & argument = LocationSet_.FindOrInsertRegisterLocation(*input.arguments.first(), false, false);
    join(origin, argument);
  }

  Analyze(*delta.subregion());

  auto & deltaOutputLocation = LocationSet_.FindOrInsertRegisterLocation(*delta.output(), false, false);
  auto & deltaLocation = LocationSet_.InsertDeltaLocation(delta);
  deltaOutputLocation.SetPointsTo(deltaLocation);

  auto & origin = *delta.result()->origin();
  if (LocationSet_.Contains(origin)) {
    auto & resultLocation = LocationSet_.Find(origin);
    join(deltaLocation, resultLocation);
  }
}

void
Steensgaard::Analyze(const phi::node & phi)
{
  /* handle context variables */
  for (auto cv = phi.begin_cv(); cv != phi.end_cv(); cv++) {
    if (!is<PointerType>(cv->type()))
      continue;

    auto & origin = LocationSet_.Find(*cv->origin());
    auto & argument = LocationSet_.FindOrInsertRegisterLocation(*cv->argument(), false, false);
    join(origin, argument);
  }

  /* handle recursion variable arguments */
  for (auto rv = phi.begin_rv(); rv != phi.end_rv(); rv++) {
    if (!is<PointerType>(rv->type()))
      continue;

    LocationSet_.FindOrInsertRegisterLocation(*rv->argument(), false, false);
  }

  Analyze(*phi.subregion());

  /* handle recursion variable outputs */
  for (auto rv = phi.begin_rv(); rv != phi.end_rv(); rv++) {
    if (!is<PointerType>(rv->type()))
      continue;

    auto & origin = LocationSet_.Find(*rv->result()->origin());
    auto & argument = LocationSet_.Find(*rv->argument());
    join(origin, argument);

    auto & output = LocationSet_.FindOrInsertRegisterLocation(*rv.output(), false, false);
    join(argument, output);
  }
}

void
Steensgaard::Analyze(const jive::gamma_node & node)
{
  /* handle entry variables */
  for (auto ev = node.begin_entryvar(); ev != node.end_entryvar(); ev++) {
    if (!jive::is<PointerType>(ev->type()))
      continue;

    auto & originloc = LocationSet_.Find(*ev->origin());
    for (auto & argument : *ev) {
      auto & argumentloc = LocationSet_.FindOrInsertRegisterLocation(argument, false, false);
      join(argumentloc, originloc);
    }
  }

  /* handle subregions */
  for (size_t n = 0; n < node.nsubregions(); n++)
    Analyze(*node.subregion(n));

  /* handle exit variables */
  for (auto ex = node.begin_exitvar(); ex != node.end_exitvar(); ex++) {
    if (!jive::is<PointerType>(ex->type()))
      continue;

    auto & outputloc = LocationSet_.FindOrInsertRegisterLocation(*ex.output(), false, false);
    for (auto & result : *ex) {
      auto & resultloc = LocationSet_.Find(*result.origin());
      join(outputloc, resultloc);
    }
  }
}

void
Steensgaard::Analyze(const jive::theta_node & theta)
{
  for (auto thetaOutput : theta) {
    if (!jive::is<PointerType>(thetaOutput->type()))
      continue;

    auto & originLocation = LocationSet_.Find(*thetaOutput->input()->origin());
    auto & argumentLocation = LocationSet_.FindOrInsertRegisterLocation(*thetaOutput->argument(), false, false);

    join(argumentLocation, originLocation);
  }

  Analyze(*theta.subregion());

  for (auto thetaOutput : theta) {
    if (!jive::is<PointerType>(thetaOutput->type()))
      continue;

    auto & originLocation = LocationSet_.Find(*thetaOutput->result()->origin());
    auto & argumentLocation = LocationSet_.Find(*thetaOutput->argument());
    auto & outputLocation = LocationSet_.FindOrInsertRegisterLocation(*thetaOutput, false, false);

    join(originLocation, argumentLocation);
    join(originLocation, outputLocation);
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

    Analyze(*AssertedCast<const structural_node>(node));
  }
}

void
Steensgaard::Analyze(const jive::graph & graph)
{
  auto add_imports = [](const jive::graph & graph, LocationSet & lset)
  {
    auto region = graph.root();
    for (size_t n = 0; n < region->narguments(); n++) {
      auto & argument = *region->argument(n);
      if (!jive::is<PointerType>(argument.type()))
        continue;
      /* FIXME: we should not add function imports */
      auto & imploc = lset.InsertImportLocation(argument);
      auto & ptr = lset.FindOrInsertRegisterLocation(argument, false, false);
      ptr.SetPointsTo(imploc);
    }
  };

  add_imports(graph, LocationSet_);
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
  steensgardStatistics.Start(module.Rvsdg());
  Analyze(module.Rvsdg());
//	std::cout << LocationSet_.ToDot() << std::flush;
  steensgardStatistics.Stop();
  sd.PrintStatistics(steensgardStatistics);


  /**
   * Construct PointsTo graph
   */
  SteensgaardPointsToGraphConstructionStatistics ptgConstructionStatistics(module.SourceFileName());
  ptgConstructionStatistics.Start(LocationSet_);
  auto pointsToGraph = ConstructPointsToGraph(LocationSet_);
//	std::cout << PointsToGraph::ToDot(*pointsToGraph) << std::flush;
  ptgConstructionStatistics.Stop(*pointsToGraph);
  sd.PrintStatistics(ptgConstructionStatistics);

  return pointsToGraph;
}

std::unique_ptr<PointsToGraph>
Steensgaard::ConstructPointsToGraph(const LocationSet & locationSets)
{
  auto pointsToGraph = PointsToGraph::Create();

  /*
    Create points-to graph nodes
  */
  std::unordered_map<Location*, PointsToGraph::Node*> locationMap;
  std::unordered_map<const disjointset<Location*>::set*, std::vector<PointsToGraph::MemoryNode*>> memoryNodeMap;
  for (auto & locationSet : locationSets) {
    for (auto & location : locationSet) {
      if (auto registerLocation = dynamic_cast<RegisterLocation*>(location)) {
        locationMap[location] = &PointsToGraph::RegisterNode::Create(
          *pointsToGraph,
          registerLocation->GetOutput());
        continue;
      }

      if (auto allocaLocation = dynamic_cast<AllocaLocation*>(location)) {
        auto node = &PointsToGraph::AllocaNode::Create(
          *pointsToGraph,
          allocaLocation->GetNode());
        memoryNodeMap[&locationSet].push_back(node);
        locationMap[location] = node;
        continue;
      }

      if (auto mallocLocation = dynamic_cast<MallocLocation*>(location)) {
        auto node = &PointsToGraph::MallocNode::Create(
          *pointsToGraph,
          mallocLocation->GetNode());
        memoryNodeMap[&locationSet].push_back(node);
        locationMap[location] = node;
        continue;
      }

      if (auto lambdaLocation = dynamic_cast<LambdaLocation*>(location)) {
        auto node = &PointsToGraph::LambdaNode::Create(
          *pointsToGraph,
          lambdaLocation->GetNode());
        memoryNodeMap[&locationSet].push_back(node);
        locationMap[location] = node;
        continue;
      }

      if (auto deltaLocation = dynamic_cast<DeltaLocation*>(location)) {
        auto node = &PointsToGraph::DeltaNode::Create(
          *pointsToGraph,
          deltaLocation->GetNode());
        memoryNodeMap[&locationSet].push_back(node);
        locationMap[location] = node;
        continue;
      }

      if (auto importLocation = dynamic_cast<ImportLocation*>(location)) {
        auto node = &PointsToGraph::ImportNode::Create(
          *pointsToGraph,
          importLocation->GetArgument());
        memoryNodeMap[&locationSet].push_back(node);
        locationMap[location] = node;
        continue;
      }

      if (dynamic_cast<DummyLocation*>(location)) {
        continue;
      }

      JLM_UNREACHABLE("Unhandled location type.");
    }
  }

  /*
    Create points-to graph edges
  */
  for (auto & set : locationSets) {
    bool pointsToUnknown = locationSets.GetSet(**set.begin()).value()->PointsToUnknownMemory();
    bool pointsToExternalMemory = locationSets.GetSet(**set.begin()).value()->PointsToExternalMemory();

    for (auto & location : set) {
      if (dynamic_cast<DummyLocation*>(location))
        continue;

      if (pointsToUnknown) {
        locationMap[location]->AddEdge(pointsToGraph->GetUnknownMemoryNode());
      }

      if (pointsToExternalMemory) {
        locationMap[location]->AddEdge(pointsToGraph->GetExternalMemoryNode());
      }

      auto pt = set.value()->GetPointsTo();
      if (pt == nullptr)
        continue;

      auto & pointsToSet = locationSets.GetSet(*pt);
      auto & memoryNodes = memoryNodeMap[&pointsToSet];
      if (memoryNodes.empty()) {
        /*
          The location points to a pointsTo set that contains
          no memory nodes. Thus, we have no idea where this pointer
          points to. Let's be conservative and let it just point to
          unknown.
        */
        locationMap[location]->AddEdge(pointsToGraph->GetUnknownMemoryNode());
        continue;
      }

      for (auto & memoryNode : memoryNodeMap[&pointsToSet])
        locationMap[location]->AddEdge(*memoryNode);
    }
  }

  return pointsToGraph;
}

void
Steensgaard::ResetState()
{
  LocationSet_.Clear();
}

}
