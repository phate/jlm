/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>
#include <jlm/util/time.hpp>

/*
	FIXME: to be removed again
*/
#include <iostream>

namespace jlm::llvm::aa
{

/** \brief Collect statistics about Steensgaard alias analysis pass
 *
 */
class Steensgaard::Statistics final : public jlm::util::Statistics {
public:
  ~Statistics() override
  = default;

  explicit
  Statistics(jlm::util::filepath sourceFile)
    : jlm::util::Statistics(Statistics::Id::SteensgaardAnalysis)
    , NumRvsdgNodes_(0)
    , SourceFile_(std::move(sourceFile))
    , NumDisjointSets_(0)
    , NumLocations_(0)
    , NumPointsToGraphNodes_(0)
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
  StartSteensgaardStatistics(const jlm::rvsdg::graph & graph) noexcept
  {
    NumRvsdgNodes_ = jlm::rvsdg::nnodes(graph.root());
    AnalysisTimer_.start();
  }

  void
  StopSteensgaardStatistics() noexcept
  {
    AnalysisTimer_.stop();
  }

  void
  StartPointsToGraphConstructionStatistics(const LocationSet & locationSet)
  {
    NumDisjointSets_ = locationSet.NumDisjointSets();
    NumLocations_ = locationSet.NumLocations();
    PointsToGraphConstructionTimer_.start();
  }

  void
  StopPointsToGraphConstructionStatistics(const PointsToGraph & pointsToGraph)
  {
    PointsToGraphConstructionTimer_.stop();
    NumPointsToGraphNodes_ = pointsToGraph.NumNodes();
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
    return jlm::util::strfmt("SteensgaardAnalysis ",
                             SourceFile_.to_str(), " ",
                             "#RvsdgNodes:", NumRvsdgNodes_, " ",
                             "AliasAnalysisTime[ns]:", AnalysisTimer_.ns(), " ",
                             "#DisjointSets:", NumDisjointSets_, " ",
                             "#Locations:", NumLocations_, " ",
                             "#PointsToGraphNodes:", NumPointsToGraphNodes_, " ",
                             "#AllocaNodes:", NumAllocaNodes_, " ",
                             "#DeltaNodes:", NumDeltaNodes_, " ",
                             "#ImportNodes:", NumImportNodes_, " ",
                             "#LambdaNodes:", NumLambdaNodes_, " ",
                             "#MallocNodes:", NumMallocNodes_, " ",
                             "#MemoryNodes:", NumMemoryNodes_, " ",
                             "#RegisterNodes:", NumRegisterNodes_, " ",
                             "#UnknownMemorySources:", NumUnknownMemorySources_, " ",
                             "PointsToGraphConstructionTime[ns]:", PointsToGraphConstructionTimer_.ns());
  }

  static std::unique_ptr<Statistics>
  Create(const jlm::util::filepath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }

private:
  size_t NumRvsdgNodes_;
  jlm::util::filepath SourceFile_;

  size_t NumDisjointSets_;
  size_t NumLocations_;

  size_t NumPointsToGraphNodes_;
  size_t NumAllocaNodes_;
  size_t NumDeltaNodes_;
  size_t NumImportNodes_;
  size_t NumLambdaNodes_;
  size_t NumMallocNodes_;
  size_t NumMemoryNodes_;
  size_t NumRegisterNodes_;
  size_t NumUnknownMemorySources_;

  jlm::util::timer AnalysisTimer_;
  jlm::util::timer PointsToGraphConstructionTimer_;
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
  Location(PointsToFlags pointsToFlags)
    : PointsToFlags_(pointsToFlags)
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
    return (PointsToFlags_ & PointsToFlags::PointsToUnknownMemory) == PointsToFlags::PointsToUnknownMemory;
  }

  [[nodiscard]] bool
  PointsToExternalMemory() const noexcept
  {
    return (PointsToFlags_ & PointsToFlags::PointsToExternalMemory) == PointsToFlags::PointsToExternalMemory;
  }

  [[nodiscard]] bool
  PointsToEscapedMemory() const noexcept
  {
    return (PointsToFlags_ & PointsToFlags::PointsToEscapedMemory) == PointsToFlags::PointsToEscapedMemory;
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

  [[nodiscard]] PointsToFlags
  GetPointsToFlags() const noexcept
  {
    return PointsToFlags_;
  }

  void
  SetPointsToFlags(PointsToFlags pointsToFlags) noexcept
  {
    PointsToFlags_ = pointsToFlags;
  }

private:
  PointsToFlags PointsToFlags_;
  Location * PointsTo_;
};

class RegisterLocation final : public Location {
public:
  constexpr explicit
  RegisterLocation(
    const jlm::rvsdg::output & output,
    PointsToFlags pointsToFlags)
    : Location(pointsToFlags)
    , IsEscapingModule_(false)
    , Output_(&output)
  {}

  [[nodiscard]] const jlm::rvsdg::output &
  GetOutput() const noexcept
  {
    return *Output_;
  }

  /** Determines whether the register location escapes the module.
   *
   * @return True, if the location escapes the module, otherwise false.
   */
  [[nodiscard]] bool
  IsEscapingModule() const noexcept
  {
    return IsEscapingModule_;
  }

  void
  SetIsEscapingModule(bool isEscapingModule) noexcept
  {
    IsEscapingModule_ = isEscapingModule;
  }

  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    auto node = jlm::rvsdg::node_output::node(Output_);
    auto index = Output_->index();

    if (jlm::rvsdg::is<jlm::rvsdg::simple_op>(node)) {
      auto nodestr = node->operation().debug_string();
      auto outputstr = Output_->type().debug_string();
      return jlm::util::strfmt(nodestr, ":", index, "[" + outputstr + "]");
    }

    if (is<lambda::cvargument>(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":cv:", index);
    }

    if (is<lambda::fctargument>(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":arg:", index);
    }

    if (is<delta::cvargument>(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":cv:", index);
    }

    if (is_gamma_argument(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":arg", index);
    }

    if (is_theta_argument(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":arg", index);
    }

    if (is_theta_output(Output_)) {
      auto dbgstr = jlm::rvsdg::node_output::node(Output_)->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":out", index);
    }

    if (is_gamma_output(Output_)) {
      auto dbgstr = jlm::rvsdg::node_output::node(Output_)->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":out", index);
    }

    if (is_import(Output_)) {
      auto import = jlm::util::AssertedCast<const jlm::rvsdg::impport>(&Output_->port());
      return jlm::util::strfmt("imp:", import->name());
    }

    if (is<phi::rvargument>(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":rvarg", index);
    }

    if (is<phi::cvargument>(Output_)) {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":cvarg", index);
    }

    return jlm::util::strfmt(jlm::rvsdg::node_output::node(Output_)->operation().debug_string(), ":", index);
  }

  [[nodiscard]] static bool
  IsEscapingModule(const Location & location) noexcept
  {
    auto registerLocation = dynamic_cast<const RegisterLocation*>(&location);
    return registerLocation
           && registerLocation->IsEscapingModule();
  }

  static std::unique_ptr<RegisterLocation>
  Create(
    const jlm::rvsdg::output & output,
    PointsToFlags pointsToFlags)
  {
    return std::make_unique<RegisterLocation>(output, pointsToFlags);
  }

private:
  bool IsEscapingModule_;
  const jlm::rvsdg::output * Output_;
};

/** \brief MemoryLocation class
*
* This class represents an abstract memory location.
*/
class MemoryLocation : public Location {
public:
  constexpr
  MemoryLocation()
    : Location(PointsToFlags::PointsToNone)
  {}
};

/** \brief AllocaLocation class
 *
 * This class represents an abstract stack location allocated by a alloca operation.
 */
class AllocaLocation final : public MemoryLocation {

  ~AllocaLocation() override = default;

  constexpr explicit
  AllocaLocation(const jlm::rvsdg::node & node)
    : MemoryLocation()
    , Node_(node)
  {
    JLM_ASSERT(is<alloca_op>(&node));
  }

public:
  [[nodiscard]] const jlm::rvsdg::node &
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
  Create(const jlm::rvsdg::node & node)
  {
    return std::unique_ptr<Location>(new AllocaLocation(node));
  }

private:
  const jlm::rvsdg::node & Node_;
};

/** \brief MallocLocation class
 *
 * This class represents an abstract heap location allocated by a malloc operation.
 */
class MallocLocation final : public MemoryLocation {

  ~MallocLocation() override = default;

  constexpr explicit
  MallocLocation(const jlm::rvsdg::node & node)
    : MemoryLocation()
    , Node_(node)
  {
    JLM_ASSERT(is<malloc_op>(&node));
  }

public:
  [[nodiscard]] const jlm::rvsdg::node &
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
  Create(const jlm::rvsdg::node & node)
  {
    return std::unique_ptr<Location>(new MallocLocation(node));
  }

private:
  const jlm::rvsdg::node & Node_;
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
    const jlm::rvsdg::argument & argument,
    PointsToFlags pointsToFlags)
    : Location(pointsToFlags)
    , Argument_(argument)
  {
    JLM_ASSERT(dynamic_cast<const llvm::impport*>(&argument.port()));
  }

  [[nodiscard]] const jlm::rvsdg::argument &
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
  Create(const jlm::rvsdg::argument & argument)
  {
    auto & rvsdgImport = *jlm::util::AssertedCast<const impport>(&argument.port());
    bool pointsToUnknownMemory = is<PointerType>(rvsdgImport.GetValueType());
    /**
     * FIXME: We use pointsToUnknownMemory for pointsToExternalMemory
     */
    auto flags =
      PointsToFlags::PointsToUnknownMemory |
      PointsToFlags::PointsToExternalMemory |
      PointsToFlags::PointsToEscapedMemory;
    return std::unique_ptr<Location>(new ImportLocation(
      argument,
      pointsToUnknownMemory ? flags : PointsToFlags::PointsToNone));
  }

private:
  const jlm::rvsdg::argument & Argument_;
};

/** \brief FIXME: write documentation
*/
class DummyLocation final : public Location {
public:
  ~DummyLocation() override = default;

  DummyLocation()
    : Location(PointsToFlags::PointsToNone)
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

RegisterLocation &
LocationSet::InsertRegisterLocation(
  const jlm::rvsdg::output & output,
  PointsToFlags pointsToFlags)
{
  JLM_ASSERT(!Contains(output));

  auto registerLocation = RegisterLocation::Create(output, pointsToFlags);
  auto registerLocationPointer = registerLocation.get();

  LocationMap_[&output] = registerLocationPointer;
  DisjointLocationSet_.insert(registerLocationPointer);
  Locations_.push_back(std::move(registerLocation));

  return *registerLocationPointer;
}

Location &
LocationSet::InsertAllocaLocation(const jlm::rvsdg::node & node)
{
  Locations_.push_back(AllocaLocation::Create(node));
  auto location = Locations_.back().get();
  DisjointLocationSet_.insert(location);

  return *location;
}

Location &
LocationSet::InsertMallocLocation(const jlm::rvsdg::node & node)
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
LocationSet::InsertImportLocation(const jlm::rvsdg::argument & argument)
{
  Locations_.push_back(ImportLocation::Create(argument));
  auto location = Locations_.back().get();
  DisjointLocationSet_.insert(location);

  return *location;
}

RegisterLocation *
LocationSet::LookupRegisterLocation(const jlm::rvsdg::output & output)
{
  auto it = LocationMap_.find(&output);
  return it == LocationMap_.end() ? nullptr : it->second;
}

bool
LocationSet::Contains(const jlm::rvsdg::output & output) const noexcept
{
  return LocationMap_.find(&output) != LocationMap_.end();
}

Location &
LocationSet::FindOrInsertRegisterLocation(
  const jlm::rvsdg::output & output,
  PointsToFlags pointsToFlags)
{
  if (auto location = LookupRegisterLocation(output))
    return GetRootLocation(*location);

  return InsertRegisterLocation(output, pointsToFlags);
}

Location &
LocationSet::GetRootLocation(Location & l) const
{
  return *GetSet(l).value();
}

Location &
LocationSet::Find(const jlm::rvsdg::output & output)
{
  auto location = LookupRegisterLocation(output);
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
    auto rootLocation = set.value();

    std::string setLabel;
    for (auto & location : set) {
      auto unknownLabel = location->PointsToUnknownMemory() ? "{U}" : "";
      auto pointsToEscapedMemoryLabel = location->PointsToEscapedMemory() ? "{E}" : "";
      auto escapesModuleLabel = RegisterLocation::IsEscapingModule(*location) ? "{EscapesModule}" : "";
      auto pointsToLabel = jlm::util::strfmt("{pt:", (intptr_t) location->GetPointsTo(), "}");
      auto locationLabel = jlm::util::strfmt((intptr_t)location, " : ", location->DebugString());

      setLabel += location == rootLocation
                  ? jlm::util::strfmt("*",
                                      locationLabel,
                                      unknownLabel,
                                      pointsToEscapedMemoryLabel,
                                      escapesModuleLabel,
                                      pointsToLabel,
                                      "*\\n")
                  : jlm::util::strfmt(locationLabel, escapesModuleLabel, "\\n");
    }

    return jlm::util::strfmt("{ ", (intptr_t)&set, " [label = \"", setLabel, "\"]; }");
  };

  auto dot_edge = [&](const DisjointLocationSet::set & set, const DisjointLocationSet::set & pointsToSet)
  {
    return jlm::util::strfmt((intptr_t)&set, " -> ", (intptr_t)&pointsToSet);
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

    auto & rootx = LocationSet_->GetRootLocation(*x);
    auto & rooty = LocationSet_->GetRootLocation(*y);
    auto flags = rootx.GetPointsToFlags() | rooty.GetPointsToFlags();
    rootx.SetPointsToFlags(flags);
    rooty.SetPointsToFlags(flags);
    auto & tmp = LocationSet_->Merge(rootx, rooty);

    if (auto root = join(rootx.GetPointsTo(), rooty.GetPointsTo()))
      tmp.SetPointsTo(*root);

    return &tmp;
  };

  join(&x, &y);
}

void
Steensgaard::Analyze(const jlm::rvsdg::simple_node & node)
{
  auto AnalyzeCall  = [](auto & s, auto & n) { s.AnalyzeCall(*jlm::util::AssertedCast<const CallNode>(&n)); };
  auto AnalyzeLoad  = [](auto & s, auto & n) { s.AnalyzeLoad(*jlm::util::AssertedCast<const LoadNode>(&n)); };
  auto AnalyzeStore = [](auto & s, auto & n) { s.AnalyzeStore(*jlm::util::AssertedCast<const StoreNode>(&n)); };

  static std::unordered_map<
    std::type_index
    , std::function<void(Steensgaard&, const jlm::rvsdg::simple_node&)>> nodes
    ({
         {typeid(alloca_op),                    [](auto & s, auto & n){ s.AnalyzeAlloca(n);                }}
       , {typeid(malloc_op),                    [](auto & s, auto & n){ s.AnalyzeMalloc(n);                }}
       , {typeid(LoadOperation),                AnalyzeLoad                                                 }
       , {typeid(StoreOperation),               AnalyzeStore                                                }
       , {typeid(CallOperation),                AnalyzeCall                                                 }
       , {typeid(GetElementPtrOperation),       [](auto & s, auto & n){ s.AnalyzeGep(n);                   }}
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
    if (jlm::rvsdg::is<PointerType>(node.output(n)->type()))
      JLM_UNREACHABLE("We should have never reached this statement.");
  }
}

void
Steensgaard::AnalyzeAlloca(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<alloca_op>(&node));

  std::function<bool(const jlm::rvsdg::valuetype&)>
    IsVaListAlloca = [&](const jlm::rvsdg::valuetype & type)
  {
    auto structType = dynamic_cast<const StructType*>(&type);

    if (structType != nullptr
        && structType->GetName() == "struct.__va_list_tag")
      return true;

    if (structType != nullptr) {
      auto & declaration = structType->GetDeclaration();

      for (size_t n = 0; n < declaration.nelements(); n++) {
        if (IsVaListAlloca(declaration.element(n)))
          return true;
      }
    }

    if (auto arrayType = dynamic_cast<const arraytype*>(&type))
      return IsVaListAlloca(arrayType->element_type());

    return false;
  };

  auto & allocaOutputLocation = LocationSet_->FindOrInsertRegisterLocation(
    *node.output(0),
    PointsToFlags::PointsToNone);
  auto & allocaLocation = LocationSet_->InsertAllocaLocation(node);
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
    allocaLocation.SetPointsToFlags(PointsToFlags::PointsToUnknownMemory);
  }
}

void
Steensgaard::AnalyzeMalloc(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<malloc_op>(&node));

  auto & mallocOutputLocation = LocationSet_->FindOrInsertRegisterLocation(
    *node.output(0),
    PointsToFlags::PointsToNone);
  auto & mallocLocation = LocationSet_->InsertMallocLocation(node);
  mallocOutputLocation.SetPointsTo(mallocLocation);
}

void
Steensgaard::AnalyzeLoad(const LoadNode & loadNode)
{
  if (!is<PointerType>(loadNode.GetValueOutput()->type()))
    return;

  auto & address = LocationSet_->Find(*loadNode.GetAddressInput()->origin());
  auto & result = LocationSet_->FindOrInsertRegisterLocation(
    *loadNode.GetValueOutput(),
    address.GetPointsToFlags());

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

  auto & addressLocation = LocationSet_->Find(address);
  auto & valueLocation = LocationSet_->Find(value);

  if (addressLocation.GetPointsTo() == nullptr) {
    addressLocation.SetPointsTo(valueLocation);
    return;
  }

  join(*addressLocation.GetPointsTo(), valueLocation);
}

void
Steensgaard::AnalyzeCall(const CallNode & callNode)
{
  auto AnalyzeDirectCall = [&](const CallNode & callNode, const lambda::node & lambda)
  {
    /*
      FIXME: What about varargs
    */

    /* handle call node arguments */
    JLM_ASSERT(lambda.nfctarguments() == callNode.ninputs() - 1);
    for (size_t n = 1; n < callNode.ninputs(); n++) {
      auto & callArgument = *callNode.input(n)->origin();
      auto & lambdaArgument = *lambda.fctargument(n-1);

      if (!is<PointerType>(callArgument.type()))
        continue;

      auto & callArgumentLocation = LocationSet_->Find(callArgument);
      auto & lambdaArgumentLocation = LocationSet_->FindOrInsertRegisterLocation(
        lambdaArgument,
        PointsToFlags::PointsToNone);

      join(callArgumentLocation, lambdaArgumentLocation);
    }

    /* handle call node results */
    auto subregion = lambda.subregion();
    JLM_ASSERT(subregion->nresults() == callNode.noutputs());
    for (size_t n = 0; n < callNode.noutputs(); n++) {
      auto & callResult = *callNode.output(n);
      auto & lambdaResult = *subregion->result(n)->origin();

      if (!is<PointerType>(callResult.type()))
        continue;

      auto & callResultLocation = LocationSet_->FindOrInsertRegisterLocation(
        callResult,
        PointsToFlags::PointsToNone);
      auto & lambdaResultLocation = LocationSet_->FindOrInsertRegisterLocation(
        lambdaResult,
        PointsToFlags::PointsToNone);

      join(callResultLocation, lambdaResultLocation);
    }
  };

  auto AnalyzeExternalCall = [&](const CallNode & callNode)
  {
    /*
      FIXME: What about varargs
    */
    for (size_t n = 1; n < callNode.NumArguments(); n++)
    {
      auto & callArgument = *callNode.input(n)->origin();

      if (is<PointerType>(callArgument.type()))
      {
        auto registerLocation = LocationSet_->LookupRegisterLocation(callArgument);
        registerLocation->SetIsEscapingModule(true);
      }
    }

    for (size_t n = 0; n < callNode.NumResults(); n++) {
      auto & callResult = *callNode.Result(n);

      if (is<PointerType>(callResult.type()))
      {
        LocationSet_->FindOrInsertRegisterLocation(
          callResult,
          PointsToFlags::PointsToExternalMemory | PointsToFlags::PointsToEscapedMemory);
      }
    }
  };

  auto AnalyzeIndirectCall = [&](const CallNode & call)
  {
    /*
      Nothing can be done for the call/lambda arguments, as it is
      an indirect call and the lambda node cannot be retrieved.
    */

    /* handle call node results */
    for (size_t n = 0; n < call.noutputs(); n++) {
      auto & callResult = *call.output(n);

      if (is<PointerType>(callResult.type()))
      {
        LocationSet_->FindOrInsertRegisterLocation(
          callResult,
          PointsToFlags::PointsToUnknownMemory |
          PointsToFlags::PointsToExternalMemory |
          PointsToFlags::PointsToEscapedMemory);
      }
    }
  };

  auto callTypeClassifier = CallNode::ClassifyCall(callNode);
  switch (callTypeClassifier->GetCallType())
  {
    case CallTypeClassifier::CallType::NonRecursiveDirectCall:
    case CallTypeClassifier::CallType::RecursiveDirectCall:
      AnalyzeDirectCall(callNode, *callTypeClassifier->GetLambdaOutput().node());
      break;
    case CallTypeClassifier::CallType::ExternalCall:
      AnalyzeExternalCall(callNode);
      break;
    case CallTypeClassifier::CallType::IndirectCall:
      AnalyzeIndirectCall(callNode);
      break;
    default:
      JLM_UNREACHABLE("Unhandled call type.");
  }
}

void
Steensgaard::AnalyzeGep(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<GetElementPtrOperation>(&node));

  auto & base = LocationSet_->Find(*node.input(0)->origin());
  auto & value = LocationSet_->FindOrInsertRegisterLocation(
    *node.output(0),
    PointsToFlags::PointsToNone);

  join(base, value);
}

void
Steensgaard::AnalyzeBitcast(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<bitcast_op>(&node));

  auto input = node.input(0);
  if (!is<PointerType>(input->type()))
    return;

  auto & operand = LocationSet_->Find(*input->origin());
  auto & result = LocationSet_->FindOrInsertRegisterLocation(
    *node.output(0),
    PointsToFlags::PointsToNone);

  join(operand, result);
}

void
Steensgaard::AnalyzeBits2ptr(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<bits2ptr_op>(&node));

  LocationSet_->FindOrInsertRegisterLocation(
    *node.output(0),
    /*
     * The register location already points to unknown memory. Unknown memory is a superset of escaped memory and
     * therefore we can simply set escaped memory to false.
     */
    PointsToFlags::PointsToUnknownMemory | PointsToFlags::PointsToExternalMemory);
}

void
Steensgaard::AnalyzeExtractValue(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ExtractValue>(&node));

  auto & result = *node.output(0);
  if (!is<PointerType>(result.type()))
    return;

  /*
   * FIXME: Have a look at this operation again to ensure that the flags add up.
   */
  LocationSet_->FindOrInsertRegisterLocation(
    result,
    PointsToFlags::PointsToUnknownMemory | PointsToFlags::PointsToExternalMemory);
}

void
Steensgaard::AnalyzeConstantPointerNull(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantPointerNullOperation>(&node));

  /*
   * ConstantPointerNull cannot point to any memory location. We therefore only insert a register node for it,
   * but let this node not point to anything.
   */
  LocationSet_->FindOrInsertRegisterLocation(
    *node.output(0),
    PointsToFlags::PointsToNone);
}

void
Steensgaard::AnalyzeConstantAggregateZero(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantAggregateZero>(&node));
  auto output = node.output(0);

  if (!IsOrContains<PointerType>(output->type()))
    return;

  /*
   * ConstantAggregateZero cannot point to any memory location. We therefore only insert a register node for it,
   * but let this node not point to anything.
   */
  LocationSet_->FindOrInsertRegisterLocation(
    *output,
    PointsToFlags::PointsToNone);
}

void
Steensgaard::AnalyzeUndef(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<UndefValueOperation>(&node));
  auto output = node.output(0);

  if (!is<PointerType>(output->type()))
    return;

  /*
   * UndefValue cannot point to any memory location. We therefore only insert a register node for it,
   * but let this node not point to anything.
   */
  LocationSet_->FindOrInsertRegisterLocation(
    *output,
    PointsToFlags::PointsToNone);
}

void
Steensgaard::AnalyzeConstantArray(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantArray>(&node));

  for (size_t n = 0; n < node.ninputs(); n++) {
    auto input = node.input(n);

    if (LocationSet_->Contains(*input->origin())) {
      auto & originLocation = LocationSet_->Find(*input->origin());
      auto & outputLocation = LocationSet_->FindOrInsertRegisterLocation(
        *node.output(0),
        PointsToFlags::PointsToNone);
      join(outputLocation, originLocation);
    }
  }
}

void
Steensgaard::AnalyzeConstantStruct(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantStruct>(&node));

  for (size_t n = 0; n < node.ninputs(); n++) {
    auto input = node.input(n);

    if (LocationSet_->Contains(*input->origin())) {
      auto & originLocation = LocationSet_->Find(*input->origin());
      auto & outputLocation = LocationSet_->FindOrInsertRegisterLocation(
        *node.output(0),
        PointsToFlags::PointsToNone);
      join(outputLocation, originLocation);
    }
  }
}

void
Steensgaard::AnalyzeMemcpy(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<Memcpy>(&node));

  /*
    FIXME: handle unknown
  */

  /*
    FIXME: write some documentation about the implementation
  */

  auto & dstAddress = LocationSet_->Find(*node.input(0)->origin());
  auto & srcAddress = LocationSet_->Find(*node.input(1)->origin());

  if (srcAddress.GetPointsTo() == nullptr) {
    /*
      If we do not know where the source address points to yet(!),
      insert a dummy location so we have something to work with.
    */
    auto & dummyLocation = LocationSet_->InsertDummyLocation();
    srcAddress.SetPointsTo(dummyLocation);
  }

  if (dstAddress.GetPointsTo() == nullptr) {
    /*
      If we do not know where the destination address points to yet(!),
      insert a dummy location so we have somehting to work with.
    */
    auto & dummyLocation = LocationSet_->InsertDummyLocation();
    dstAddress.SetPointsTo(dummyLocation);
  }

  auto & srcMemory = LocationSet_->GetRootLocation(*srcAddress.GetPointsTo());
  auto & dstMemory = LocationSet_->GetRootLocation(*dstAddress.GetPointsTo());

  if (srcMemory.GetPointsTo() == nullptr) {
    auto & dummyLocation = LocationSet_->InsertDummyLocation();
    srcMemory.SetPointsTo(dummyLocation);
  }

  if (dstMemory.GetPointsTo() == nullptr) {
    auto & dummyLocation = LocationSet_->InsertDummyLocation();
    dstMemory.SetPointsTo(dummyLocation);
  }

  join(*srcMemory.GetPointsTo(), *dstMemory.GetPointsTo());
}

void
Steensgaard::Analyze(const lambda::node & lambda)
{
  /*
   * Handle context variables
   */
  for (auto & cv : lambda.ctxvars()) {
    if (!jlm::rvsdg::is<PointerType>(cv.type()))
      continue;

    auto & originLocation = LocationSet_->Find(*cv.origin());
    auto & argumentLocation = LocationSet_->FindOrInsertRegisterLocation(
      *cv.argument(),
      PointsToFlags::PointsToNone);
    join(originLocation, argumentLocation);
  }

  /*
   * Handle function arguments
   */
  if (lambda.direct_calls()) {
    for (auto & argument : lambda.fctarguments()) {
      if (jlm::rvsdg::is<PointerType>(argument.type())) {
        LocationSet_->FindOrInsertRegisterLocation(
          argument,
          PointsToFlags::PointsToNone);
      }
    }
  } else {
    /*
     * FIXME: We also end up in this case when the lambda has only direct calls, but is exported.
     */
    for (auto & argument : lambda.fctarguments()) {
      if (jlm::rvsdg::is<PointerType>(argument.type()))
        LocationSet_->FindOrInsertRegisterLocation(
          argument,
          PointsToFlags::PointsToExternalMemory | PointsToFlags::PointsToEscapedMemory);
    }
  }

  Analyze(*lambda.subregion());

  /*
   * Handle function results
   */
  for (auto & result : lambda.fctresults()) {
    if (jlm::rvsdg::is<PointerType>(result.type())) {
      auto registerLocation = LocationSet_->LookupRegisterLocation(*result.origin());

      if (is_exported(lambda))
        registerLocation->SetIsEscapingModule(true);
    }
  }

  /*
   * Handle function
   */
  auto & lambdaOutputLocation = LocationSet_->FindOrInsertRegisterLocation(
    *lambda.output(),
    PointsToFlags::PointsToNone);
  auto & lambdaLocation = LocationSet_->InsertLambdaLocation(lambda);
  lambdaOutputLocation.SetPointsTo(lambdaLocation);
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

    auto & origin = LocationSet_->Find(*input.origin());
    auto & argument = LocationSet_->FindOrInsertRegisterLocation(
      *input.arguments.first(),
      PointsToFlags::PointsToNone);
    join(origin, argument);
  }

  Analyze(*delta.subregion());

  auto & deltaOutputLocation = LocationSet_->FindOrInsertRegisterLocation(
    *delta.output(),
    PointsToFlags::PointsToNone);
  auto & deltaLocation = LocationSet_->InsertDeltaLocation(delta);
  deltaOutputLocation.SetPointsTo(deltaLocation);

  auto & origin = *delta.result()->origin();
  if (LocationSet_->Contains(origin)) {
    auto & resultLocation = LocationSet_->Find(origin);
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

    auto & origin = LocationSet_->Find(*cv->origin());
    auto & argument = LocationSet_->FindOrInsertRegisterLocation(
      *cv->argument(),
      PointsToFlags::PointsToNone);
    join(origin, argument);
  }

  /* handle recursion variable arguments */
  for (auto rv = phi.begin_rv(); rv != phi.end_rv(); rv++) {
    if (!is<PointerType>(rv->type()))
      continue;

    LocationSet_->FindOrInsertRegisterLocation(
      *rv->argument(),
      PointsToFlags::PointsToNone);
  }

  Analyze(*phi.subregion());

  /* handle recursion variable outputs */
  for (auto rv = phi.begin_rv(); rv != phi.end_rv(); rv++) {
    if (!is<PointerType>(rv->type()))
      continue;

    auto & origin = LocationSet_->Find(*rv->result()->origin());
    auto & argument = LocationSet_->Find(*rv->argument());
    join(origin, argument);

    auto & output = LocationSet_->FindOrInsertRegisterLocation(
      *rv.output(),
      PointsToFlags::PointsToNone);
    join(argument, output);
  }
}

void
Steensgaard::Analyze(const jlm::rvsdg::gamma_node & node)
{
  /* handle entry variables */
  for (auto ev = node.begin_entryvar(); ev != node.end_entryvar(); ev++) {
    if (!jlm::rvsdg::is<PointerType>(ev->type()))
      continue;

    auto & originLocation = LocationSet_->Find(*ev->origin());
    for (auto & argument : *ev) {
      auto & argumentLocation = LocationSet_->FindOrInsertRegisterLocation(
        argument,
        PointsToFlags::PointsToNone);
      join(argumentLocation, originLocation);
    }
  }

  /* handle subregions */
  for (size_t n = 0; n < node.nsubregions(); n++)
    Analyze(*node.subregion(n));

  /* handle exit variables */
  for (auto ex = node.begin_exitvar(); ex != node.end_exitvar(); ex++) {
    if (!jlm::rvsdg::is<PointerType>(ex->type()))
      continue;

    auto & outputLocation = LocationSet_->FindOrInsertRegisterLocation(
      *ex.output(),
      PointsToFlags::PointsToNone);
    for (auto & result : *ex) {
      auto & resultLocation = LocationSet_->Find(*result.origin());
      join(outputLocation, resultLocation);
    }
  }
}

void
Steensgaard::Analyze(const jlm::rvsdg::theta_node & theta)
{
  for (auto thetaOutput : theta) {
    if (!jlm::rvsdg::is<PointerType>(thetaOutput->type()))
      continue;

    auto & originLocation = LocationSet_->Find(*thetaOutput->input()->origin());
    auto & argumentLocation = LocationSet_->FindOrInsertRegisterLocation(
      *thetaOutput->argument(),
      PointsToFlags::PointsToNone);

    join(argumentLocation, originLocation);
  }

  Analyze(*theta.subregion());

  for (auto thetaOutput : theta) {
    if (!jlm::rvsdg::is<PointerType>(thetaOutput->type()))
      continue;

    auto & originLocation = LocationSet_->Find(*thetaOutput->result()->origin());
    auto & argumentLocation = LocationSet_->Find(*thetaOutput->argument());
    auto & outputLocation = LocationSet_->FindOrInsertRegisterLocation(
      *thetaOutput,
      PointsToFlags::PointsToNone);

    join(originLocation, argumentLocation);
    join(originLocation, outputLocation);
  }
}

void
Steensgaard::Analyze(const jlm::rvsdg::structural_node & node)
{
  auto analyzeLambda = [](auto& s, auto& n){s.Analyze(*static_cast<const lambda::node*>(&n));    };
  auto analyzeDelta  = [](auto& s, auto& n){s.Analyze(*static_cast<const delta::node*>(&n));     };
  auto analyzeGamma  = [](auto& s, auto& n){s.Analyze(*static_cast<const jlm::rvsdg::gamma_node*>(&n));};
  auto analyzeTheta  = [](auto& s, auto& n){s.Analyze(*static_cast<const jlm::rvsdg::theta_node*>(&n));};
  auto analyzePhi    = [](auto& s, auto& n){s.Analyze(*static_cast<const phi::node*>(&n));       };

  static std::unordered_map<
    std::type_index
    , std::function<void(Steensgaard&, const jlm::rvsdg::structural_node&)>> nodes
    ({
         {typeid(lambda::operation), analyzeLambda }
       , {typeid(delta::operation),  analyzeDelta  }
       , {typeid(jlm::rvsdg::gamma_op),    analyzeGamma  }
       , {typeid(jlm::rvsdg::theta_op),    analyzeTheta  }
       , {typeid(phi::operation),    analyzePhi    }
     });

  auto & op = node.operation();
  JLM_ASSERT(nodes.find(typeid(op)) != nodes.end());
  nodes[typeid(op)](*this, node);
}

void
Steensgaard::Analyze(jlm::rvsdg::region & region)
{
  using namespace jlm::rvsdg;

  topdown_traverser traverser(&region);
  for (auto & node : traverser) {
    if (auto simpleNode = dynamic_cast<const simple_node*>(node)) {
      Analyze(*simpleNode);
      continue;
    }

    Analyze(*jlm::util::AssertedCast<const structural_node>(node));
  }
}

void
Steensgaard::Analyze(const jlm::rvsdg::graph & graph)
{
  auto add_imports = [](const jlm::rvsdg::graph & graph, LocationSet & lset)
  {
    auto region = graph.root();
    for (size_t n = 0; n < region->narguments(); n++) {
      auto & argument = *region->argument(n);
      if (!jlm::rvsdg::is<PointerType>(argument.type()))
        continue;
      /* FIXME: we should not add function imports */
      auto & importLocation = lset.InsertImportLocation(argument);
      auto & importArgumentLocation = lset.FindOrInsertRegisterLocation(
        argument,
        PointsToFlags::PointsToNone);
      importArgumentLocation.SetPointsTo(importLocation);
    }
  };

  auto MarkExportsAsEscaping = [](const jlm::rvsdg::graph & graph, LocationSet & locationSet)
  {
    auto rootRegion = graph.root();

    for (size_t n = 0; n < rootRegion->nresults(); n++) {
      auto & result = *rootRegion->result(n);
      auto registerLocation = locationSet.LookupRegisterLocation(*result.origin());
      registerLocation->SetIsEscapingModule(true);
    }
  };

  add_imports(graph, *LocationSet_);
  Analyze(*graph.root());
  MarkExportsAsEscaping(graph, *LocationSet_);
}

std::unique_ptr<PointsToGraph>
Steensgaard::Analyze(
  const RvsdgModule & module,
  jlm::util::StatisticsCollector & statisticsCollector)
{
  LocationSet_ = LocationSet::Create();
  auto statistics = Statistics::Create(module.SourceFileName());

  // Perform Steensgaard analysis
  statistics->StartSteensgaardStatistics(module.Rvsdg());
  Analyze(module.Rvsdg());
  // std::cout << LocationSet_.ToDot() << std::flush;
  statistics->StopSteensgaardStatistics();

  // Construct PointsTo graph
  statistics->StartPointsToGraphConstructionStatistics(*LocationSet_);
  auto pointsToGraph = ConstructPointsToGraph(*LocationSet_);
  // std::cout << PointsToGraph::ToDot(*pointsToGraph) << std::flush;
  statistics->StopPointsToGraphConstructionStatistics(*pointsToGraph);

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  return pointsToGraph;
}

std::unique_ptr<PointsToGraph>
Steensgaard::ConstructPointsToGraph(const LocationSet & locationSets)
{
  auto pointsToGraph = PointsToGraph::Create();

  auto CreatePointsToGraphNode = [](const Location & location, PointsToGraph & pointsToGraph) -> PointsToGraph::Node&
  {
    if (auto registerLocation = dynamic_cast<const RegisterLocation*>(&location))
      return PointsToGraph::RegisterNode::Create(pointsToGraph, registerLocation->GetOutput());

    if (auto allocaLocation = dynamic_cast<const AllocaLocation*>(&location))
      return PointsToGraph::AllocaNode::Create(pointsToGraph, allocaLocation->GetNode());

    if (auto mallocLocation = dynamic_cast<const MallocLocation*>(&location))
      return PointsToGraph::MallocNode::Create(pointsToGraph, mallocLocation->GetNode());

    if (auto lambdaLocation = dynamic_cast<const LambdaLocation*>(&location))
      return PointsToGraph::LambdaNode::Create(pointsToGraph, lambdaLocation->GetNode());

    if (auto deltaLocation = dynamic_cast<const DeltaLocation*>(&location))
      return PointsToGraph::DeltaNode::Create(pointsToGraph, deltaLocation->GetNode());

    if (auto importLocation = dynamic_cast<const ImportLocation*>(&location))
      return PointsToGraph::ImportNode::Create(pointsToGraph, importLocation->GetArgument());

    JLM_UNREACHABLE("Unhandled location type.");
  };

  /*
   * We marked all register locations that escape the module throughout the analysis using
   * RegisterLocation::SetIsEscapingModule(). This function uses these as starting point for computing all module
   * escaping memory locations.
   */
  auto FindModuleEscapingMemoryNodes = [](
    std::unordered_set<RegisterLocation*> & moduleEscapingRegisterLocations,
    const LocationSet & locationSets,
    const std::unordered_map<const jlm::util::disjointset<Location*>::set*, std::vector<PointsToGraph::MemoryNode*>> & memoryNodeMap)
  {
    /*
     * Initialize our working set.
     */
    std::unordered_set<Location*> toVisit;
    while (!moduleEscapingRegisterLocations.empty()) {
      auto registerLocation = *moduleEscapingRegisterLocations.begin();
      moduleEscapingRegisterLocations.erase(registerLocation);

      auto & set = locationSets.GetSet(*registerLocation);
      auto pointsToLocation = set.value()->GetPointsTo();
      if (pointsToLocation)
        toVisit.insert(pointsToLocation);
    }

    /*
     * Collect escaping memory nodes.
     */
    jlm::util::HashSet<const LocationSet::DisjointLocationSet::set*> visitedSets;
    std::unordered_set<PointsToGraph::MemoryNode*> escapedMemoryNodes;
    while (!toVisit.empty()) {
      auto moduleEscapingLocation = *toVisit.begin();
      toVisit.erase(moduleEscapingLocation);

      auto & set = locationSets.GetSet(*moduleEscapingLocation);
      /*
       * Check if we already visited this set to avoid an endless loop.
       */
      if (visitedSets.Contains(&set))
      {
        continue;
      }
      visitedSets.Insert(&set);

      auto & memoryNodes = memoryNodeMap.at(&set);
      for (auto & memoryNode : memoryNodes)
      {
        memoryNode->MarkAsModuleEscaping();
        escapedMemoryNodes.insert(memoryNode);
      }


      auto pointsToLocation = set.value()->GetPointsTo();
      if (pointsToLocation)
        toVisit.insert(pointsToLocation);
    }

    return escapedMemoryNodes;
  };


  std::unordered_map<const Location*, PointsToGraph::Node*> locationMap;
  std::unordered_map<const jlm::util::disjointset<Location*>::set*, std::vector<PointsToGraph::MemoryNode*>> memoryNodeMap;
  std::unordered_set<RegisterLocation*> moduleEscapingRegisterLocations;

  /*
   * Create points-to graph nodes
   */
  for (auto & locationSet : locationSets)
  {
    memoryNodeMap[&locationSet] = {};
    for (auto & location : locationSet)
    {
      /*
       * We can ignore dummy nodes. They only exist for structural purposes and have no equivalent in the RVSDG.
       */
      if (dynamic_cast<const DummyLocation*>(location))
        continue;

      auto pointsToGraphNode = &CreatePointsToGraphNode(*location, *pointsToGraph);
      locationMap[location] = pointsToGraphNode;

      if (auto memoryNode = dynamic_cast<PointsToGraph::MemoryNode*>(pointsToGraphNode))
        memoryNodeMap[&locationSet].push_back(memoryNode);

      if (RegisterLocation::IsEscapingModule(*location))
        moduleEscapingRegisterLocations.insert(jlm::util::AssertedCast<RegisterLocation>(location));
    }
  }

  auto escapedMemoryNodes = FindModuleEscapingMemoryNodes(
    moduleEscapingRegisterLocations,
    locationSets,
    memoryNodeMap);

  /*
   * Create points-to graph edges
   */
  for (auto & set : locationSets) {
    bool pointsToUnknown = locationSets.GetSet(**set.begin()).value()->PointsToUnknownMemory();
    bool pointsToExternalMemory = locationSets.GetSet(**set.begin()).value()->PointsToExternalMemory();
    bool pointsToEscapedMemory = locationSets.GetSet(**set.begin()).value()->PointsToEscapedMemory();

    for (auto & location : set) {
      if (dynamic_cast<DummyLocation*>(location))
        continue;

      auto & pointsToGraphNode = *locationMap[location];

      if (pointsToUnknown)
        pointsToGraphNode.AddEdge(pointsToGraph->GetUnknownMemoryNode());

      if (pointsToExternalMemory)
        pointsToGraphNode.AddEdge(pointsToGraph->GetExternalMemoryNode());

      if (pointsToEscapedMemory) {
        for (auto & escapedMemoryNode : escapedMemoryNodes)
          pointsToGraphNode.AddEdge(*escapedMemoryNode);
      }

      auto pointsToLocation = set.value()->GetPointsTo();
      if (pointsToLocation == nullptr)
        continue;

      auto & pointsToSet = locationSets.GetSet(*pointsToLocation);
      auto & memoryNodes = memoryNodeMap[&pointsToSet];

      for (auto & memoryNode : memoryNodes)
        pointsToGraphNode.AddEdge(*memoryNode);
    }
  }

  return pointsToGraph;
}

}
