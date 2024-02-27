/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm::aa
{

/**
 * Determines whether \p output should be handled by the Steensgaard analysis.
 *
 * @param output An rvsdg::output.
 * @return True if \p output should handled, otherwise false.
 */
static bool
ShouldHandle(const rvsdg::output & output)
{
  return IsOrContains<PointerType>(output.type());
}

/**
 * Determines whether \p node should be handled by the Steensgaard analysis.
 *
 * @param node An rvsdg::simple_node.
 * @return True if \p node should be handled, otherwise false.
 */
static bool
ShouldHandle(const rvsdg::simple_node & node)
{
  for (size_t n = 0; n < node.ninputs(); n++)
  {
    auto & origin = *node.input(n)->origin();
    if (ShouldHandle(origin))
    {
      return true;
    }
  }

  return false;
}

enum class PointsToFlags
{
  PointsToNone = 1 << 0,
  PointsToUnknownMemory = 1 << 1,
  PointsToExternalMemory = 1 << 2,
  PointsToEscapedMemory = 1 << 3,
};

static inline PointsToFlags
operator|(PointsToFlags lhs, PointsToFlags rhs)
{
  typedef typename std::underlying_type<PointsToFlags>::type underlyingType;
  return static_cast<PointsToFlags>(
      static_cast<underlyingType>(lhs) | static_cast<underlyingType>(rhs));
}

static inline PointsToFlags
operator&(PointsToFlags lhs, PointsToFlags rhs)
{
  typedef typename std::underlying_type<PointsToFlags>::type underlyingType;
  return static_cast<PointsToFlags>(
      static_cast<underlyingType>(lhs) & static_cast<underlyingType>(rhs));
}

/** \brief Location class
 *
 * This class represents an abstract location in the program.
 */
class Location
{
public:
  virtual ~Location() = default;

  constexpr explicit Location(PointsToFlags pointsToFlags)
      : PointsToFlags_(pointsToFlags),
        PointsTo_(nullptr)
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
    return (PointsToFlags_ & PointsToFlags::PointsToUnknownMemory)
        == PointsToFlags::PointsToUnknownMemory;
  }

  [[nodiscard]] bool
  PointsToExternalMemory() const noexcept
  {
    return (PointsToFlags_ & PointsToFlags::PointsToExternalMemory)
        == PointsToFlags::PointsToExternalMemory;
  }

  [[nodiscard]] bool
  PointsToEscapedMemory() const noexcept
  {
    return (PointsToFlags_ & PointsToFlags::PointsToEscapedMemory)
        == PointsToFlags::PointsToEscapedMemory;
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

  template<typename L>
  static bool
  Is(const Location & location) noexcept
  {
    static_assert(
        std::is_base_of<Location, L>::value,
        "Template parameter L must be derived from Location.");

    return dynamic_cast<const L *>(&location) != nullptr;
  }

private:
  PointsToFlags PointsToFlags_;
  Location * PointsTo_;
};

/**
 * Represents a single RVSDG register location, i.e., an RVSDG output.
 */
class RegisterLocation final : public Location
{
public:
  constexpr explicit RegisterLocation(
      const jlm::rvsdg::output & output,
      PointsToFlags pointsToFlags)
      : Location(pointsToFlags),
        IsEscapingModule_(false),
        Output_(&output)
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

    if (jlm::rvsdg::is<jlm::rvsdg::simple_op>(node))
    {
      auto nodestr = node->operation().debug_string();
      auto outputstr = Output_->type().debug_string();
      return jlm::util::strfmt(nodestr, ":", index, "[" + outputstr + "]");
    }

    if (is<lambda::cvargument>(Output_))
    {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":cv:", index);
    }

    if (is<lambda::fctargument>(Output_))
    {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":arg:", index);
    }

    if (is<delta::cvargument>(Output_))
    {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":cv:", index);
    }

    if (is_gamma_argument(Output_))
    {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":arg", index);
    }

    if (is_theta_argument(Output_))
    {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":arg", index);
    }

    if (is_theta_output(Output_))
    {
      auto dbgstr = jlm::rvsdg::node_output::node(Output_)->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":out", index);
    }

    if (is_gamma_output(Output_))
    {
      auto dbgstr = jlm::rvsdg::node_output::node(Output_)->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":out", index);
    }

    if (is_import(Output_))
    {
      auto import = jlm::util::AssertedCast<const jlm::rvsdg::impport>(&Output_->port());
      return jlm::util::strfmt("imp:", import->name());
    }

    if (is<phi::rvargument>(Output_))
    {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":rvarg", index);
    }

    if (is<phi::cvargument>(Output_))
    {
      auto dbgstr = Output_->region()->node()->operation().debug_string();
      return jlm::util::strfmt(dbgstr, ":cvarg", index);
    }

    return jlm::util::strfmt(
        jlm::rvsdg::node_output::node(Output_)->operation().debug_string(),
        ":",
        index);
  }

  [[nodiscard]] static bool
  IsEscapingModule(const Location & location) noexcept
  {
    auto registerLocation = dynamic_cast<const RegisterLocation *>(&location);
    return registerLocation && registerLocation->IsEscapingModule();
  }

  static std::unique_ptr<RegisterLocation>
  Create(const jlm::rvsdg::output & output, PointsToFlags pointsToFlags)
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
class MemoryLocation : public Location
{
public:
  constexpr MemoryLocation()
      : Location(PointsToFlags::PointsToNone)
  {}
};

/** \brief AllocaLocation class
 *
 * This class represents an abstract stack location allocated by a alloca operation.
 */
class AllocaLocation final : public MemoryLocation
{

  ~AllocaLocation() override = default;

  explicit AllocaLocation(const jlm::rvsdg::node & node)
      : MemoryLocation(),
        Node_(node)
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
class MallocLocation final : public MemoryLocation
{
  ~MallocLocation() override = default;

  explicit MallocLocation(const jlm::rvsdg::node & node)
      : MemoryLocation(),
        Node_(node)
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
class LambdaLocation final : public MemoryLocation
{
  ~LambdaLocation() override = default;

  constexpr explicit LambdaLocation(const lambda::node & lambda)
      : MemoryLocation(),
        Lambda_(lambda)
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
 * This class represents an abstract global variable location, statically allocated by a delta
 * operation.
 */
class DeltaLocation final : public MemoryLocation
{
  ~DeltaLocation() override = default;

  constexpr explicit DeltaLocation(const delta::node & delta)
      : MemoryLocation(),
        Delta_(delta)
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

/**
 * This class represents all global variable and function locations that are imported to the
 * translation unit.
 *
 * FIXME: We should be able to further distinguish imported locations between function and data
 * locations. Function locations cannot point to any other memory locations, helping us to
 * potentially improve analysis precision.
 */
class ImportLocation final : public MemoryLocation
{
  ~ImportLocation() override = default;

  ImportLocation(const rvsdg::argument & argument, PointsToFlags pointsToFlags)
      : MemoryLocation(),
        Argument_(argument)
  {
    JLM_ASSERT(dynamic_cast<const llvm::impport *>(&argument.port()));
    SetPointsToFlags(pointsToFlags);
  }

public:
  [[nodiscard]] const rvsdg::argument &
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
  Create(const rvsdg::argument & argument)
  {
    JLM_ASSERT(is<PointerType>(argument.type()));

    // If the imported memory location is a pointer type or contains a pointer type, then these
    // pointers can point to values that escaped this module.
    auto & rvsdgImport = *util::AssertedCast<const impport>(&argument.port());
    bool isOrContainsPointerType = IsOrContains<PointerType>(rvsdgImport.GetValueType());

    return std::unique_ptr<Location>(new ImportLocation(
        argument,
        isOrContainsPointerType
            ? PointsToFlags::PointsToExternalMemory | PointsToFlags::PointsToEscapedMemory
            : PointsToFlags::PointsToNone));
  }

private:
  const rvsdg::argument & Argument_;
};

/**
 * This class represents a location that only exists for structural purposes of the algorithm. It
 * has no equivalent in the RVSDG.
 */
class DummyLocation final : public Location
{
  ~DummyLocation() override = default;

  DummyLocation()
      : Location(PointsToFlags::PointsToNone)
  {}

public:
  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    return "UNNAMED";
  }

  static std::unique_ptr<Location>
  Create()
  {
    return std::unique_ptr<Location>(new DummyLocation());
  }
};

/** \brief LocationSet class
 */
class LocationSet final
{
public:
  using DisjointLocationSet = typename jlm::util::disjointset<Location *>;

  using const_iterator = std::unordered_map<const jlm::rvsdg::output *, Location *>::const_iterator;

  ~LocationSet() = default;

  LocationSet() = default;

  LocationSet(const LocationSet &) = delete;

  LocationSet(LocationSet &&) = delete;

  LocationSet &
  operator=(const LocationSet &) = delete;

  LocationSet &
  operator=(LocationSet &&) = delete;

  DisjointLocationSet::set_iterator
  begin() const
  {
    return DisjointLocationSet_.begin();
  }

  DisjointLocationSet::set_iterator
  end() const
  {
    return DisjointLocationSet_.end();
  }

  Location &
  InsertAllocaLocation(const jlm::rvsdg::node & node)
  {
    Locations_.push_back(AllocaLocation::Create(node));
    auto location = Locations_.back().get();
    DisjointLocationSet_.insert(location);

    return *location;
  }

  Location &
  InsertMallocLocation(const jlm::rvsdg::node & node)
  {
    Locations_.push_back(MallocLocation::Create(node));
    auto location = Locations_.back().get();
    DisjointLocationSet_.insert(location);

    return *location;
  }

  Location &
  InsertLambdaLocation(const lambda::node & lambda)
  {
    Locations_.push_back(LambdaLocation::Create(lambda));
    auto location = Locations_.back().get();
    DisjointLocationSet_.insert(location);

    return *location;
  }

  Location &
  InsertDeltaLocation(const delta::node & delta)
  {
    Locations_.push_back(DeltaLocation::Create(delta));
    auto location = Locations_.back().get();
    DisjointLocationSet_.insert(location);

    return *location;
  }

  Location &
  InsertImportLocation(const jlm::rvsdg::argument & argument)
  {
    Locations_.push_back(ImportLocation::Create(argument));
    auto location = Locations_.back().get();
    DisjointLocationSet_.insert(location);

    return *location;
  }

  Location &
  InsertDummyLocation()
  {
    Locations_.push_back(DummyLocation::Create());
    auto location = Locations_.back().get();
    DisjointLocationSet_.insert(location);

    return *location;
  }

  bool
  Contains(const jlm::rvsdg::output & output) const noexcept
  {
    return LocationMap_.find(&output) != LocationMap_.end();
  }

  Location &
  FindOrInsertRegisterLocation(const jlm::rvsdg::output & output, PointsToFlags pointsToFlags)
  {
    if (auto location = LookupRegisterLocation(output))
      return GetRootLocation(*location);

    return InsertRegisterLocation(output, pointsToFlags);
  }

  const DisjointLocationSet::set &
  GetSet(Location & location) const
  {
    return *DisjointLocationSet_.find(&location);
  }

  size_t
  NumDisjointSets() const noexcept
  {
    return DisjointLocationSet_.nsets();
  }

  size_t
  NumLocations() const noexcept
  {
    return DisjointLocationSet_.nvalues();
  }

  Location &
  GetRootLocation(Location & location) const
  {
    return *GetSet(location).value();
  }

  Location &
  Find(const jlm::rvsdg::output & output)
  {
    auto location = LookupRegisterLocation(output);
    JLM_ASSERT(location != nullptr);

    return GetRootLocation(*location);
  }

  RegisterLocation *
  LookupRegisterLocation(const jlm::rvsdg::output & output)
  {
    auto it = LocationMap_.find(&output);
    return it == LocationMap_.end() ? nullptr : it->second;
  }

  bool
  ContainsRegisterLocation(const rvsdg::output & output)
  {
    auto it = LocationMap_.find(&output);
    return it != LocationMap_.end();
  }

  std::string
  ToDot() const
  {
    auto toDotNode = [](const DisjointLocationSet::set & set)
    {
      auto rootLocation = set.value();

      std::string setLabel;
      for (auto & location : set)
      {
        auto unknownLabel = location->PointsToUnknownMemory() ? "{U}" : "";
        auto pointsToEscapedMemoryLabel = location->PointsToEscapedMemory() ? "{E}" : "";
        auto escapesModuleLabel =
            RegisterLocation::IsEscapingModule(*location) ? "{EscapesModule}" : "";
        auto pointsToLabel = jlm::util::strfmt("{pt:", (intptr_t)location->GetPointsTo(), "}");
        auto locationLabel = jlm::util::strfmt((intptr_t)location, " : ", location->DebugString());

        setLabel += location == rootLocation
                      ? jlm::util::strfmt(
                          "*",
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

    auto toDotEdge =
        [](const DisjointLocationSet::set & set, const DisjointLocationSet::set & pointsToSet)
    {
      return jlm::util::strfmt((intptr_t)&set, " -> ", (intptr_t)&pointsToSet);
    };

    std::string str;
    str.append("digraph DisjointLocationSetGraph {\n");

    for (auto & set : DisjointLocationSet_)
    {
      str += toDotNode(set) + "\n";

      auto pointsTo = set.value()->GetPointsTo();
      if (pointsTo != nullptr)
      {
        auto pointsToSet = DisjointLocationSet_.find(pointsTo);
        str += toDotEdge(set, *pointsToSet) + "\n";
      }
    }

    str.append("}\n");

    return str;
  }

  /** \brief Perform a recursive union of Location \p x and \p y.
   */
  void
  Join(Location & x, Location & y)
  {
    std::function<Location *(Location *, Location *)> join = [&](Location * x, Location * y)
    {
      if (x == nullptr)
        return y;

      if (y == nullptr)
        return x;

      if (x == y)
        return x;

      auto & rootx = GetRootLocation(*x);
      auto & rooty = GetRootLocation(*y);
      auto flags = rootx.GetPointsToFlags() | rooty.GetPointsToFlags();
      rootx.SetPointsToFlags(flags);
      rooty.SetPointsToFlags(flags);
      auto & tmp = Merge(rootx, rooty);

      if (auto root = join(rootx.GetPointsTo(), rooty.GetPointsTo()))
        tmp.SetPointsTo(*root);

      return &tmp;
    };

    join(&x, &y);
  }

  static std::unique_ptr<LocationSet>
  Create()
  {
    return std::make_unique<LocationSet>();
  }

private:
  Location &
  Merge(Location & location1, Location & location2)
  {
    return *DisjointLocationSet_.merge(&location1, &location2)->value();
  }

  RegisterLocation &
  InsertRegisterLocation(const jlm::rvsdg::output & output, PointsToFlags pointsToFlags)
  {
    JLM_ASSERT(!Contains(output));

    auto registerLocation = RegisterLocation::Create(output, pointsToFlags);
    auto registerLocationPointer = registerLocation.get();

    LocationMap_[&output] = registerLocationPointer;
    DisjointLocationSet_.insert(registerLocationPointer);
    Locations_.push_back(std::move(registerLocation));

    return *registerLocationPointer;
  }

  DisjointLocationSet DisjointLocationSet_;
  std::vector<std::unique_ptr<Location>> Locations_;
  std::unordered_map<const jlm::rvsdg::output *, RegisterLocation *> LocationMap_;
};

/** \brief Collect statistics about Steensgaard alias analysis pass
 *
 */
class Steensgaard::Statistics final : public util::Statistics
{
  const char * AnalysisTimerLabel_ = "AliasAnalysisTime";
  const char * PointsToFlagsPropagationTimerLabel_ = "PointsToFlagsPropagationTime";
  const char * PointsToGraphConstructionTimerLabel_ = "PointsToGraphConstructionTime";
  const char * UnknownMemoryNodeSourcesRedirectionTimerLabel_ =
      "UnknownMemoryNodeSourcesRedirectionTime";

  const char * NumDisjointSetsLabel_ = "#DisjointSets";
  const char * NumLocationsLabel_ = "#Locations";

  const char * NumPointsToGraphNodesLabel_ = "#PointsToGraphNodes";
  const char * NumAllocaNodesLabel_ = "#AllocaNodes";
  const char * NuMDeltaNodesLabel_ = "#DeltaNodes";
  const char * NumImportNodesLabel_ = "#ImportNodes";
  const char * NumLambdaNodesLabel_ = "#LambdaNodes";
  const char * NumMallocNodesLabel_ = "#MallocNodes";
  const char * NumMemoryNodesLabel_ = "#MemoryNodes";
  const char * NumRegisterNodesLabel_ = "#RegisterNodes";
  const char * NumUnknownMemorySourcesLabel_ = "#UnknownMemorySources";

public:
  ~Statistics() override = default;

  explicit Statistics(const util::filepath & sourceFile)
      : util::Statistics(Statistics::Id::SteensgaardAnalysis, sourceFile)
  {}

  void
  StartSteensgaardStatistics(const jlm::rvsdg::graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodes, rvsdg::nnodes(graph.root()));
    AddTimer(AnalysisTimerLabel_).start();
  }

  void
  StopSteensgaardStatistics() noexcept
  {
    GetTimer(AnalysisTimerLabel_).stop();
  }

  void
  StartPointsToFlagsPropagationStatistics(const LocationSet & disjointLocationSet) noexcept
  {
    AddMeasurement(NumDisjointSetsLabel_, disjointLocationSet.NumDisjointSets());
    AddMeasurement(NumLocationsLabel_, disjointLocationSet.NumLocations());
    AddTimer(PointsToFlagsPropagationTimerLabel_).start();
  }

  void
  StopPointsToFlagsPropagationStatistics() noexcept
  {
    GetTimer(PointsToFlagsPropagationTimerLabel_).stop();
  }

  void
  StartPointsToGraphConstructionStatistics(const LocationSet & locationSet)
  {
    AddTimer(PointsToGraphConstructionTimerLabel_).start();
  }

  void
  StopPointsToGraphConstructionStatistics(const PointsToGraph & pointsToGraph)
  {
    GetTimer(PointsToGraphConstructionTimerLabel_).stop();
    AddMeasurement(NumPointsToGraphNodesLabel_, pointsToGraph.NumNodes());
    AddMeasurement(NumAllocaNodesLabel_, pointsToGraph.NumAllocaNodes());
    AddMeasurement(NuMDeltaNodesLabel_, pointsToGraph.NumDeltaNodes());
    AddMeasurement(NumImportNodesLabel_, pointsToGraph.NumImportNodes());
    AddMeasurement(NumLambdaNodesLabel_, pointsToGraph.NumLambdaNodes());
    AddMeasurement(NumMallocNodesLabel_, pointsToGraph.NumMallocNodes());
    AddMeasurement(NumMemoryNodesLabel_, pointsToGraph.NumMemoryNodes());
    AddMeasurement(NumRegisterNodesLabel_, pointsToGraph.NumRegisterNodes());
    AddMeasurement(
        NumUnknownMemorySourcesLabel_,
        pointsToGraph.GetUnknownMemoryNode().NumSources());
  }

  void
  StartUnknownMemoryNodeSourcesRedirectionStatistics()
  {
    AddTimer(UnknownMemoryNodeSourcesRedirectionTimerLabel_).start();
  }

  void
  StopUnknownMemoryNodeSourcesRedirectionStatistics()
  {
    GetTimer(UnknownMemoryNodeSourcesRedirectionTimerLabel_).stop();
  }

  static std::unique_ptr<Statistics>
  Create(const util::filepath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

Steensgaard::~Steensgaard() = default;

Steensgaard::Steensgaard() = default;

void
Steensgaard::AnalyzeSimpleNode(const jlm::rvsdg::simple_node & node)
{
  if (is<alloca_op>(&node))
  {
    AnalyzeAlloca(node);
  }
  else if (is<malloc_op>(&node))
  {
    AnalyzeMalloc(node);
  }
  else if (auto loadNode = dynamic_cast<const LoadNode *>(&node))
  {
    AnalyzeLoad(*loadNode);
  }
  else if (auto storeNode = dynamic_cast<const StoreNode *>(&node))
  {
    AnalyzeStore(*storeNode);
  }
  else if (auto callNode = dynamic_cast<const CallNode *>(&node))
  {
    AnalyzeCall(*callNode);
  }
  else if (is<GetElementPtrOperation>(&node))
  {
    AnalyzeGep(node);
  }
  else if (is<bitcast_op>(&node))
  {
    AnalyzeBitcast(node);
  }
  else if (is<bits2ptr_op>(&node))
  {
    AnalyzeBits2ptr(node);
  }
  else if (is<ConstantPointerNullOperation>(&node))
  {
    AnalyzeConstantPointerNull(node);
  }
  else if (is<UndefValueOperation>(&node))
  {
    AnalyzeUndef(node);
  }
  else if (is<Memcpy>(&node))
  {
    AnalyzeMemcpy(node);
  }
  else if (is<ConstantArray>(&node))
  {
    AnalyzeConstantArray(node);
  }
  else if (is<ConstantStruct>(&node))
  {
    AnalyzeConstantStruct(node);
  }
  else if (is<ConstantAggregateZero>(&node))
  {
    AnalyzeConstantAggregateZero(node);
  }
  else if (is<ExtractValue>(&node))
  {
    AnalyzeExtractValue(node);
  }
  else if (is<FreeOperation>(&node) || is<ptrcmp_op>(&node))
  {
    // Nothing needs to be done as these operations do not affect points-to sets
  }
  else
  {
    // Ensure that we took care of all pointer consuming nodes.
    JLM_ASSERT(!ShouldHandle(node));
  }
}

void
Steensgaard::AnalyzeAlloca(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<alloca_op>(&node));

  std::function<bool(const jlm::rvsdg::valuetype &)> IsVaListAlloca =
      [&](const jlm::rvsdg::valuetype & type)
  {
    auto structType = dynamic_cast<const StructType *>(&type);

    if (structType != nullptr && structType->GetName() == "struct.__va_list_tag")
      return true;

    if (structType != nullptr)
    {
      auto & declaration = structType->GetDeclaration();

      for (size_t n = 0; n < declaration.NumElements(); n++)
      {
        if (IsVaListAlloca(declaration.GetElement(n)))
          return true;
      }
    }

    if (auto arrayType = dynamic_cast<const arraytype *>(&type))
      return IsVaListAlloca(arrayType->element_type());

    return false;
  };

  auto & allocaOutputLocation =
      LocationSet_->FindOrInsertRegisterLocation(*node.output(0), PointsToFlags::PointsToNone);
  auto & allocaLocation = LocationSet_->InsertAllocaLocation(node);
  allocaOutputLocation.SetPointsTo(allocaLocation);

  auto & op = *dynamic_cast<const alloca_op *>(&node.operation());

  // FIXME: We should discover such an alloca already at construction time and not by traversing the
  // type here.
  if (IsVaListAlloca(op.value_type()))
  {
    // FIXME: We should be able to do better than just pointing to unknown.
    allocaLocation.SetPointsToFlags(PointsToFlags::PointsToUnknownMemory);
  }
}

void
Steensgaard::AnalyzeMalloc(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<malloc_op>(&node));

  auto & mallocOutputLocation =
      LocationSet_->FindOrInsertRegisterLocation(*node.output(0), PointsToFlags::PointsToNone);
  auto & mallocLocation = LocationSet_->InsertMallocLocation(node);
  mallocOutputLocation.SetPointsTo(mallocLocation);
}

void
Steensgaard::AnalyzeLoad(const LoadNode & loadNode)
{
  auto & result = *loadNode.GetValueOutput();
  auto & address = *loadNode.GetAddressInput()->origin();

  if (!ShouldHandle(result))
    return;

  auto & addressLocation = LocationSet_->Find(address);
  auto & resultLocation =
      LocationSet_->FindOrInsertRegisterLocation(result, addressLocation.GetPointsToFlags());

  if (addressLocation.GetPointsTo() == nullptr)
  {
    addressLocation.SetPointsTo(resultLocation);
  }
  else
  {
    LocationSet_->Join(resultLocation, *addressLocation.GetPointsTo());
  }
}

void
Steensgaard::AnalyzeStore(const StoreNode & storeNode)
{
  auto & address = *storeNode.GetAddressInput()->origin();
  auto & value = *storeNode.GetValueInput()->origin();

  if (!ShouldHandle(value))
    return;

  auto & addressLocation = LocationSet_->Find(address);
  auto & valueLocation = LocationSet_->Find(value);

  if (addressLocation.GetPointsTo() == nullptr)
  {
    addressLocation.SetPointsTo(valueLocation);
  }
  else
  {
    LocationSet_->Join(*addressLocation.GetPointsTo(), valueLocation);
  }
}

void
Steensgaard::AnalyzeCall(const CallNode & callNode)
{
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
Steensgaard::AnalyzeDirectCall(const CallNode & callNode, const lambda::node & lambdaNode)
{
  auto & lambdaFunctionType = lambdaNode.operation().type();
  auto & callFunctionType = callNode.GetOperation().GetFunctionType();
  if (callFunctionType != lambdaFunctionType)
  {
    // LLVM permits code where it can happen that the number and type of the arguments handed in to
    // the call node do not agree with the number and type of lambda parameters, even though it is a
    // direct call. See jlm::tests::LambdaCallArgumentMismatch for an example. We handle this case
    // the same as an indirect call, as it is impossible to join the call argument/result locations
    // with the corresponding lambda argument/result locations.
    AnalyzeIndirectCall(callNode);
    return;
  }

  // FIXME: What about varargs
  // Handle call node operands
  for (size_t n = 1; n < callNode.ninputs(); n++)
  {
    auto & callArgument = *callNode.input(n)->origin();
    auto & lambdaArgument = *lambdaNode.fctargument(n - 1);

    if (ShouldHandle(callArgument))
    {
      auto & callArgumentLocation = LocationSet_->Find(callArgument);
      auto & lambdaArgumentLocation =
          LocationSet_->FindOrInsertRegisterLocation(lambdaArgument, PointsToFlags::PointsToNone);

      LocationSet_->Join(callArgumentLocation, lambdaArgumentLocation);
    }
  }

  // Handle call node results
  auto subregion = lambdaNode.subregion();
  JLM_ASSERT(subregion->nresults() == callNode.noutputs());
  for (size_t n = 0; n < callNode.noutputs(); n++)
  {
    auto & callResult = *callNode.output(n);
    auto & lambdaResult = *subregion->result(n)->origin();

    if (ShouldHandle(callResult))
    {
      auto & callResultLocation =
          LocationSet_->FindOrInsertRegisterLocation(callResult, PointsToFlags::PointsToNone);
      auto & lambdaResultLocation =
          LocationSet_->FindOrInsertRegisterLocation(lambdaResult, PointsToFlags::PointsToNone);

      LocationSet_->Join(callResultLocation, lambdaResultLocation);
    }
  }
}

void
Steensgaard::AnalyzeExternalCall(const CallNode & callNode)
{
  // FIXME: What about varargs
  for (size_t n = 1; n < callNode.NumArguments(); n++)
  {
    auto & callArgument = *callNode.input(n)->origin();

    if (ShouldHandle(callArgument))
    {
      auto registerLocation = LocationSet_->LookupRegisterLocation(callArgument);
      registerLocation->SetIsEscapingModule(true);
    }
  }

  for (size_t n = 0; n < callNode.NumResults(); n++)
  {
    auto & callResult = *callNode.Result(n);

    if (ShouldHandle(callResult))
    {
      LocationSet_->FindOrInsertRegisterLocation(
          callResult,
          PointsToFlags::PointsToExternalMemory | PointsToFlags::PointsToEscapedMemory);
    }
  }
}

void
Steensgaard::AnalyzeIndirectCall(const CallNode & callNode)
{
  // Nothing can be done for the call/lambda arguments, as it is
  // an indirect call and the lambda node cannot be retrieved.

  // Handle call node results
  for (size_t n = 0; n < callNode.noutputs(); n++)
  {
    auto & callResult = *callNode.output(n);

    if (ShouldHandle(callResult))
    {
      LocationSet_->FindOrInsertRegisterLocation(
          callResult,
          PointsToFlags::PointsToUnknownMemory | PointsToFlags::PointsToExternalMemory
              | PointsToFlags::PointsToEscapedMemory);
    }
  }
}

void
Steensgaard::AnalyzeGep(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<GetElementPtrOperation>(&node));

  auto & base = LocationSet_->Find(*node.input(0)->origin());
  auto & value =
      LocationSet_->FindOrInsertRegisterLocation(*node.output(0), PointsToFlags::PointsToNone);

  LocationSet_->Join(base, value);
}

void
Steensgaard::AnalyzeBitcast(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<bitcast_op>(&node));

  auto & operand = *node.input(0)->origin();
  auto & result = *node.output(0);

  if (ShouldHandle(operand))
  {
    auto & operandLocation = LocationSet_->Find(operand);
    auto & resultLocation =
        LocationSet_->FindOrInsertRegisterLocation(result, PointsToFlags::PointsToNone);

    LocationSet_->Join(operandLocation, resultLocation);
  }
}

void
Steensgaard::AnalyzeBits2ptr(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<bits2ptr_op>(&node));

  LocationSet_->FindOrInsertRegisterLocation(
      *node.output(0),
      // The register location already points to unknown memory. Unknown memory is a superset of
      // escaped memory, and therefore we can simply set escaped memory to false.
      PointsToFlags::PointsToUnknownMemory | PointsToFlags::PointsToExternalMemory);
}

void
Steensgaard::AnalyzeExtractValue(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ExtractValue>(&node));

  auto & result = *node.output(0);

  if (ShouldHandle(result))
  {
    // FIXME: Have a look at this operation again to ensure that the flags add up.
    LocationSet_->FindOrInsertRegisterLocation(
        result,
        PointsToFlags::PointsToUnknownMemory | PointsToFlags::PointsToExternalMemory);
  }
}

void
Steensgaard::AnalyzeConstantPointerNull(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantPointerNullOperation>(&node));

  // ConstantPointerNull cannot point to any memory location. We therefore only insert a register
  // node for it, but let this node not point to anything.
  LocationSet_->FindOrInsertRegisterLocation(*node.output(0), PointsToFlags::PointsToNone);
}

void
Steensgaard::AnalyzeConstantAggregateZero(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantAggregateZero>(&node));
  auto & output = *node.output(0);

  if (ShouldHandle(output))
  {
    // ConstantAggregateZero cannot point to any memory location. We therefore only insert a
    // register node for it, but let this node not point to anything.
    LocationSet_->FindOrInsertRegisterLocation(output, PointsToFlags::PointsToNone);
  }
}

void
Steensgaard::AnalyzeUndef(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<UndefValueOperation>(&node));
  auto & output = *node.output(0);

  if (ShouldHandle(output))
  {
    // UndefValue cannot point to any memory location. We therefore only insert a register node for
    // it, but let this node not point to anything.
    LocationSet_->FindOrInsertRegisterLocation(output, PointsToFlags::PointsToNone);
  }
}

void
Steensgaard::AnalyzeConstantArray(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantArray>(&node));

  auto & output = *node.output(0);
  for (size_t n = 0; n < node.ninputs(); n++)
  {
    auto & operand = *node.input(n)->origin();

    if (LocationSet_->Contains(operand))
    {
      auto & originLocation = LocationSet_->Find(operand);
      auto & outputLocation =
          LocationSet_->FindOrInsertRegisterLocation(output, PointsToFlags::PointsToNone);
      LocationSet_->Join(outputLocation, originLocation);
    }
  }
}

void
Steensgaard::AnalyzeConstantStruct(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<ConstantStruct>(&node));

  auto & output = *node.output(0);
  for (size_t n = 0; n < node.ninputs(); n++)
  {
    auto & operand = *node.input(n)->origin();

    if (LocationSet_->Contains(operand))
    {
      auto & originLocation = LocationSet_->Find(operand);
      auto & outputLocation =
          LocationSet_->FindOrInsertRegisterLocation(output, PointsToFlags::PointsToNone);
      LocationSet_->Join(outputLocation, originLocation);
    }
  }
}

void
Steensgaard::AnalyzeMemcpy(const jlm::rvsdg::simple_node & node)
{
  JLM_ASSERT(is<Memcpy>(&node));

  auto & dstAddress = LocationSet_->Find(*node.input(0)->origin());
  auto & srcAddress = LocationSet_->Find(*node.input(1)->origin());

  // We implement memcpy by pointing srcAddress and dstAddress to the same underlying memory:
  //
  // srcAddress -> underlyingMemory
  // dstAddress -> underlyingMemory
  //
  // Preferably, I would have liked to implement it as follows:
  //
  // srcAddress -> srcMemory
  // dstAddress -> dstMemory (which is a copy of srcMemory)
  // srcMemory and dstMemory -> underlyingMemory
  //
  // However, this was not possible due to points-to flags propagation. We have no guarantee that
  // srcAddress and dstAddress are annotated with the right flags BEFORE PropagatePointsToFlags()
  // ran. In this scheme, dstMemory is a copy of srcMemory, which means that it should also get a
  // copy of its flags (which again are the same as the points-to flags of srcAddress).
  if (srcAddress.GetPointsTo() == nullptr)
  {
    auto & underlyingMemory = LocationSet_->InsertDummyLocation();
    srcAddress.SetPointsTo(underlyingMemory);
  }

  if (dstAddress.GetPointsTo() == nullptr)
  {
    dstAddress.SetPointsTo(*srcAddress.GetPointsTo());
  }
  else
  {
    // Unifies the underlying memory of srcMemory and dstMemory
    LocationSet_->Join(*srcAddress.GetPointsTo(), *dstAddress.GetPointsTo());
  }
}

void
Steensgaard::AnalyzeLambda(const lambda::node & lambda)
{
  // Handle context variables
  for (auto & cv : lambda.ctxvars())
  {
    auto & origin = *cv.origin();

    if (ShouldHandle(origin))
    {
      auto & originLocation = LocationSet_->Find(origin);
      auto & argumentLocation =
          LocationSet_->FindOrInsertRegisterLocation(*cv.argument(), PointsToFlags::PointsToNone);
      LocationSet_->Join(originLocation, argumentLocation);
    }
  }

  // Handle function arguments
  auto callSummary = lambda.ComputeCallSummary();
  if (callSummary->HasOnlyDirectCalls())
  {
    for (auto & argument : lambda.fctarguments())
    {
      if (ShouldHandle(argument))
      {
        LocationSet_->FindOrInsertRegisterLocation(argument, PointsToFlags::PointsToNone);
      }
    }
  }
  else
  {
    // FIXME: We also end up in this case when the lambda has only direct calls, but is exported.
    for (auto & argument : lambda.fctarguments())
    {
      if (rvsdg::is<PointerType>(argument.type()))
        LocationSet_->FindOrInsertRegisterLocation(
            argument,
            PointsToFlags::PointsToExternalMemory | PointsToFlags::PointsToEscapedMemory);
    }
  }

  AnalyzeRegion(*lambda.subregion());

  // Handle function results
  for (auto & result : lambda.fctresults())
  {
    auto & operand = *result.origin();

    if (ShouldHandle(operand))
    {
      auto registerLocation = LocationSet_->LookupRegisterLocation(operand);

      if (is_exported(lambda))
        registerLocation->SetIsEscapingModule(true);
    }
  }

  // Handle function
  auto & lambdaOutputLocation =
      LocationSet_->FindOrInsertRegisterLocation(*lambda.output(), PointsToFlags::PointsToNone);
  auto & lambdaLocation = LocationSet_->InsertLambdaLocation(lambda);
  lambdaOutputLocation.SetPointsTo(lambdaLocation);
}

void
Steensgaard::AnalyzeDelta(const delta::node & delta)
{
  // Handle context variables
  for (auto & input : delta.ctxvars())
  {
    auto & origin = *input.origin();

    if (ShouldHandle(origin))
    {
      auto & originLocation = LocationSet_->Find(origin);
      auto & argumentLocation = LocationSet_->FindOrInsertRegisterLocation(
          *input.arguments.first(),
          PointsToFlags::PointsToNone);
      LocationSet_->Join(originLocation, argumentLocation);
    }
  }

  AnalyzeRegion(*delta.subregion());

  auto & deltaOutputLocation =
      LocationSet_->FindOrInsertRegisterLocation(*delta.output(), PointsToFlags::PointsToNone);
  auto & deltaLocation = LocationSet_->InsertDeltaLocation(delta);
  deltaOutputLocation.SetPointsTo(deltaLocation);

  auto & origin = *delta.result()->origin();
  if (LocationSet_->Contains(origin))
  {
    auto & resultLocation = LocationSet_->Find(origin);
    LocationSet_->Join(deltaLocation, resultLocation);
  }
}

void
Steensgaard::AnalyzePhi(const phi::node & phi)
{
  // Handle context variables
  for (auto cv = phi.begin_cv(); cv != phi.end_cv(); cv++)
  {
    auto & origin = *cv->origin();

    if (ShouldHandle(origin))
    {
      auto & originLocation = LocationSet_->Find(origin);
      auto & argumentLocation =
          LocationSet_->FindOrInsertRegisterLocation(*cv->argument(), PointsToFlags::PointsToNone);
      LocationSet_->Join(originLocation, argumentLocation);
    }
  }

  // Handle recursion variable arguments
  for (auto rv = phi.begin_rv(); rv != phi.end_rv(); rv++)
  {
    auto & argument = *rv->argument();

    if (ShouldHandle(argument))
    {
      LocationSet_->FindOrInsertRegisterLocation(argument, PointsToFlags::PointsToNone);
    }
  }

  AnalyzeRegion(*phi.subregion());

  // Handle recursion variable outputs
  for (auto rv = phi.begin_rv(); rv != phi.end_rv(); rv++)
  {
    auto & argument = *rv->argument();
    auto & output = *rv.output();
    auto & result = *rv->result();

    if (ShouldHandle(argument))
    {
      auto & originLocation = LocationSet_->Find(*result.origin());
      auto & argumentLocation = LocationSet_->Find(argument);
      auto & outputLocation =
          LocationSet_->FindOrInsertRegisterLocation(output, PointsToFlags::PointsToNone);

      LocationSet_->Join(originLocation, argumentLocation);
      LocationSet_->Join(argumentLocation, outputLocation);
    }
  }
}

void
Steensgaard::AnalyzeGamma(const jlm::rvsdg::gamma_node & node)
{
  // Handle entry variables
  for (auto ev = node.begin_entryvar(); ev != node.end_entryvar(); ev++)
  {
    auto & origin = *ev->origin();

    if (ShouldHandle(origin))
    {
      auto & originLocation = LocationSet_->Find(*ev->origin());
      for (auto & argument : *ev)
      {
        auto & argumentLocation =
            LocationSet_->FindOrInsertRegisterLocation(argument, PointsToFlags::PointsToNone);
        LocationSet_->Join(argumentLocation, originLocation);
      }
    }
  }

  // Handle subregions
  for (size_t n = 0; n < node.nsubregions(); n++)
    AnalyzeRegion(*node.subregion(n));

  // Handle exit variables
  for (auto ex = node.begin_exitvar(); ex != node.end_exitvar(); ex++)
  {
    auto & output = *ex.output();

    if (ShouldHandle(output))
    {
      auto & outputLocation =
          LocationSet_->FindOrInsertRegisterLocation(output, PointsToFlags::PointsToNone);
      for (auto & result : *ex)
      {
        auto & resultLocation = LocationSet_->Find(*result.origin());
        LocationSet_->Join(outputLocation, resultLocation);
      }
    }
  }
}

void
Steensgaard::AnalyzeTheta(const jlm::rvsdg::theta_node & theta)
{
  for (auto thetaOutput : theta)
  {
    if (ShouldHandle(*thetaOutput))
    {
      auto & originLocation = LocationSet_->Find(*thetaOutput->input()->origin());
      auto & argumentLocation = LocationSet_->FindOrInsertRegisterLocation(
          *thetaOutput->argument(),
          PointsToFlags::PointsToNone);

      LocationSet_->Join(argumentLocation, originLocation);
    }
  }

  AnalyzeRegion(*theta.subregion());

  for (auto thetaOutput : theta)
  {
    if (ShouldHandle(*thetaOutput))
    {
      auto & originLocation = LocationSet_->Find(*thetaOutput->result()->origin());
      auto & argumentLocation = LocationSet_->Find(*thetaOutput->argument());
      auto & outputLocation =
          LocationSet_->FindOrInsertRegisterLocation(*thetaOutput, PointsToFlags::PointsToNone);

      LocationSet_->Join(originLocation, argumentLocation);
      LocationSet_->Join(originLocation, outputLocation);
    }
  }
}

void
Steensgaard::AnalyzeStructuralNode(const jlm::rvsdg::structural_node & node)
{
  if (auto lambdaNode = dynamic_cast<const lambda::node *>(&node))
  {
    AnalyzeLambda(*lambdaNode);
  }
  else if (auto deltaNode = dynamic_cast<const delta::node *>(&node))
  {
    AnalyzeDelta(*deltaNode);
  }
  else if (auto gammaNode = dynamic_cast<const rvsdg::gamma_node *>(&node))
  {
    AnalyzeGamma(*gammaNode);
  }
  else if (auto thetaNode = dynamic_cast<const rvsdg::theta_node *>(&node))
  {
    AnalyzeTheta(*thetaNode);
  }
  else if (auto phiNode = dynamic_cast<const phi::node *>(&node))
  {
    AnalyzePhi(*phiNode);
  }
  else
  {
    JLM_UNREACHABLE("Unhandled node type!");
  }
}

void
Steensgaard::AnalyzeRegion(jlm::rvsdg::region & region)
{
  // Check that we added a RegisterLocation for each required argument
  for (size_t n = 0; n < region.narguments(); n++)
  {
    auto & argument = *region.argument(n);
    if (ShouldHandle(argument))
    {
      JLM_ASSERT(LocationSet_->ContainsRegisterLocation(argument));
    }
  }

  using namespace jlm::rvsdg;

  topdown_traverser traverser(&region);
  for (auto & node : traverser)
  {
    if (auto simpleNode = dynamic_cast<const simple_node *>(node))
    {
      AnalyzeSimpleNode(*simpleNode);
    }
    else if (auto structuralNode = dynamic_cast<const structural_node *>(node))
    {
      AnalyzeStructuralNode(*structuralNode);
    }
    else
    {
      JLM_UNREACHABLE("Unhandled node type.");
    }
  }
}

void
Steensgaard::AnalyzeRvsdg(const jlm::rvsdg::graph & graph)
{
  AnalyzeImports(graph);
  AnalyzeRegion(*graph.root());
  AnalyzeExports(graph);
}

void
Steensgaard::AnalyzeImports(const rvsdg::graph & graph)
{
  auto rootRegion = graph.root();
  for (size_t n = 0; n < rootRegion->narguments(); n++)
  {
    auto & argument = *rootRegion->argument(n);

    if (ShouldHandle(argument))
    {
      auto & importLocation = LocationSet_->InsertImportLocation(argument);
      auto & registerLocation =
          LocationSet_->FindOrInsertRegisterLocation(argument, PointsToFlags::PointsToNone);
      registerLocation.SetPointsTo(importLocation);
    }
  }
}

void
Steensgaard::AnalyzeExports(const rvsdg::graph & graph)
{
  auto rootRegion = graph.root();

  for (size_t n = 0; n < rootRegion->nresults(); n++)
  {
    auto & result = *rootRegion->result(n);
    auto registerLocation = LocationSet_->LookupRegisterLocation(*result.origin());
    registerLocation->SetIsEscapingModule(true);
  }
}

std::unique_ptr<PointsToGraph>
Steensgaard::Analyze(const RvsdgModule & rvsdgModule)
{
  util::StatisticsCollector statisticsCollector;
  return Analyze(rvsdgModule, statisticsCollector);
}

std::unique_ptr<PointsToGraph>
Steensgaard::Analyze(
    const RvsdgModule & module,
    jlm::util::StatisticsCollector & statisticsCollector)
{
  // std::unordered_map<const rvsdg::output *, std::string> outputMap;
  // std::cout << jlm::rvsdg::view(module.Rvsdg().root(), outputMap) << std::flush;

  LocationSet_ = LocationSet::Create();
  auto statistics = Statistics::Create(module.SourceFileName());

  // Perform Steensgaard analysis
  statistics->StartSteensgaardStatistics(module.Rvsdg());
  AnalyzeRvsdg(module.Rvsdg());
  // std::cout << LocationSet_->ToDot() << std::flush;
  statistics->StopSteensgaardStatistics();

  // Propagate points-to flags in disjoint location set graph
  statistics->StartPointsToFlagsPropagationStatistics(*LocationSet_);
  PropagatePointsToFlags();
  statistics->StopPointsToFlagsPropagationStatistics();

  // Construct PointsTo graph
  statistics->StartPointsToGraphConstructionStatistics(*LocationSet_);
  auto pointsToGraph = ConstructPointsToGraph();
  // std::cout << PointsToGraph::ToDot(*pointsToGraph, outputMap) << std::flush;
  statistics->StopPointsToGraphConstructionStatistics(*pointsToGraph);

  // Redirect unknown memory node sources
  statistics->StartUnknownMemoryNodeSourcesRedirectionStatistics();
  RedirectUnknownMemoryNodeSources(*pointsToGraph);
  statistics->StopUnknownMemoryNodeSourcesRedirectionStatistics();

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done with the analysis
  LocationSet_.reset();

  return pointsToGraph;
}

void
Steensgaard::PropagatePointsToFlags()
{
  bool pointsToFlagsChanged;
  do
  {
    pointsToFlagsChanged = false;

    for (auto & set : *LocationSet_)
    {
      auto location = set.value();

      // Nothing needs to be done if this set does not point to another set
      if (!location->GetPointsTo())
      {
        continue;
      }
      auto & pointsToLocation = LocationSet_->GetRootLocation(*location->GetPointsTo());

      auto locationFlags = location->GetPointsToFlags();
      auto pointsToLocationFlags = pointsToLocation.GetPointsToFlags();
      auto combinedFlags = locationFlags | pointsToLocationFlags;
      if (pointsToLocationFlags != combinedFlags)
      {
        pointsToLocation.SetPointsToFlags(combinedFlags);
        pointsToFlagsChanged = true;
      }
    }
  } while (pointsToFlagsChanged);
}

PointsToGraph::MemoryNode &
Steensgaard::CreatePointsToGraphMemoryNode(const Location & location, PointsToGraph & pointsToGraph)
{
  if (auto allocaLocation = dynamic_cast<const AllocaLocation *>(&location))
    return PointsToGraph::AllocaNode::Create(pointsToGraph, allocaLocation->GetNode());

  if (auto mallocLocation = dynamic_cast<const MallocLocation *>(&location))
    return PointsToGraph::MallocNode::Create(pointsToGraph, mallocLocation->GetNode());

  if (auto lambdaLocation = dynamic_cast<const LambdaLocation *>(&location))
    return PointsToGraph::LambdaNode::Create(pointsToGraph, lambdaLocation->GetNode());

  if (auto deltaLocation = dynamic_cast<const DeltaLocation *>(&location))
    return PointsToGraph::DeltaNode::Create(pointsToGraph, deltaLocation->GetNode());

  if (auto importLocation = dynamic_cast<const ImportLocation *>(&location))
    return PointsToGraph::ImportNode::Create(pointsToGraph, importLocation->GetArgument());

  JLM_UNREACHABLE("Unhandled location type.");
}

util::HashSet<PointsToGraph::MemoryNode *>
Steensgaard::CollectEscapedMemoryNodes(
    const util::HashSet<RegisterLocation *> & escapingRegisterLocations,
    const std::unordered_map<
        const util::disjointset<Location *>::set *,
        std::vector<PointsToGraph::MemoryNode *>> & memoryNodesInSet) const
{
  // Initialize working set
  util::HashSet<Location *> toVisit;
  for (auto registerLocation : escapingRegisterLocations.Items())
  {
    auto & set = LocationSet_->GetSet(*registerLocation);
    if (auto pointsToLocation = set.value()->GetPointsTo())
    {
      toVisit.Insert(pointsToLocation);
    }
  }

  // Collect escaped memory nodes
  util::HashSet<PointsToGraph::MemoryNode *> escapedMemoryNodes;
  util::HashSet<const LocationSet::DisjointLocationSet::set *> visited;
  while (!toVisit.IsEmpty())
  {
    auto moduleEscapingLocation = *toVisit.Items().begin();
    toVisit.Remove(moduleEscapingLocation);

    auto & set = LocationSet_->GetSet(*moduleEscapingLocation);

    // Check if we already visited this set to avoid an endless loop
    if (visited.Contains(&set))
    {
      continue;
    }
    visited.Insert(&set);

    auto & memoryNodes = memoryNodesInSet.at(&set);
    for (auto & memoryNode : memoryNodes)
    {
      memoryNode->MarkAsModuleEscaping();
      escapedMemoryNodes.Insert(memoryNode);
    }

    if (auto pointsToLocation = set.value()->GetPointsTo())
    {
      toVisit.Insert(pointsToLocation);
    }
  }

  return escapedMemoryNodes;
}

std::unique_ptr<PointsToGraph>
Steensgaard::ConstructPointsToGraph() const
{
  auto pointsToGraph = PointsToGraph::Create();

  // All the memory nodes within a LocationSet
  std::unordered_map<
      const util::disjointset<Location *>::set *,
      std::vector<PointsToGraph::MemoryNode *>>
      memoryNodesInSet;

  // All register locations that are marked as RegisterLocation::IsEscapingModule()
  util::HashSet<RegisterLocation *> escapingRegisterLocations;

  // Mapping between locations and points-to graph nodes
  std::unordered_map<const Location *, PointsToGraph::Node *> locationMap;

  // Create points-to graph nodes
  for (auto & locationSet : *LocationSet_)
  {
    memoryNodesInSet[&locationSet] = {};

    util::HashSet<const rvsdg::output *> registers;
    util::HashSet<RegisterLocation *> registerLocations;
    for (auto & location : locationSet)
    {
      if (auto registerLocation = dynamic_cast<RegisterLocation *>(location))
      {
        registers.Insert(&registerLocation->GetOutput());
        registerLocations.Insert(registerLocation);

        if (registerLocation->IsEscapingModule())
          escapingRegisterLocations.Insert(registerLocation);
      }
      else if (Location::Is<MemoryLocation>(*location))
      {
        auto & pointsToGraphNode = CreatePointsToGraphMemoryNode(*location, *pointsToGraph);
        memoryNodesInSet[&locationSet].push_back(&pointsToGraphNode);
        locationMap[location] = &pointsToGraphNode;
      }
      else if (Location::Is<DummyLocation>(*location))
      {
        // We can ignore dummy nodes. They only exist for structural purposes in the Steensgaard
        // analysis and have no equivalent in the points-to graph.
      }
      else
      {
        JLM_UNREACHABLE("Unhandled location type.");
      }
    }

    // We found register locations in this set.
    // Create a single points-to graph register node for all of them.
    if (!registerLocations.IsEmpty())
    {
      auto & pointsToGraphNode = PointsToGraph::RegisterNode::Create(*pointsToGraph, registers);
      for (auto registerLocation : registerLocations.Items())
        locationMap[registerLocation] = &pointsToGraphNode;
    }
  }

  auto escapedMemoryNodes = CollectEscapedMemoryNodes(escapingRegisterLocations, memoryNodesInSet);

  // Create points-to graph edges
  for (auto & set : *LocationSet_)
  {
    bool pointsToUnknown = LocationSet_->GetSet(**set.begin()).value()->PointsToUnknownMemory();
    bool pointsToExternalMemory =
        LocationSet_->GetSet(**set.begin()).value()->PointsToExternalMemory();
    bool pointsToEscapedMemory =
        LocationSet_->GetSet(**set.begin()).value()->PointsToEscapedMemory();

    bool handledRegisterLocations = false;
    for (auto & location : set)
    {
      // We can ignore dummy nodes. They only exist for structural purposes in the Steensgaard
      // analysis and have no equivalent in the points-to graph.
      if (Location::Is<DummyLocation>(*location))
        continue;

      if (Location::Is<RegisterLocation>(*location))
      {
        // All register locations in a set are mapped to a single points-to graph register-set node.
        // For the sake of performance, we would like to ony insert points-graph edges for this
        // register-set node once, rather than redo the work for each register (location) in the
        // set.
        if (handledRegisterLocations)
          continue;
        handledRegisterLocations = true;
      }

      auto & pointsToGraphNode = *locationMap[location];

      if (pointsToUnknown)
        pointsToGraphNode.AddEdge(pointsToGraph->GetUnknownMemoryNode());

      if (pointsToExternalMemory)
        pointsToGraphNode.AddEdge(pointsToGraph->GetExternalMemoryNode());

      if (pointsToEscapedMemory)
      {
        for (auto & escapedMemoryNode : escapedMemoryNodes.Items())
          pointsToGraphNode.AddEdge(*escapedMemoryNode);
      }

      // Add edges to all memory nodes the location points to
      if (auto pointsToLocation = set.value()->GetPointsTo())
      {
        auto & pointsToSet = LocationSet_->GetSet(*pointsToLocation);
        auto & memoryNodes = memoryNodesInSet[&pointsToSet];

        for (auto & memoryNode : memoryNodes)
          pointsToGraphNode.AddEdge(*memoryNode);
      }
    }
  }

  return pointsToGraph;
}

void
Steensgaard::RedirectUnknownMemoryNodeSources(PointsToGraph & pointsToGraph)
{
  auto collectMemoryNodes = [](PointsToGraph & pointsToGraph)
  {
    std::vector<PointsToGraph::MemoryNode *> memoryNodes;
    for (auto & allocaNode : pointsToGraph.AllocaNodes())
      memoryNodes.push_back(&allocaNode);

    for (auto & deltaNode : pointsToGraph.DeltaNodes())
      memoryNodes.push_back(&deltaNode);

    for (auto & lambdaNode : pointsToGraph.LambdaNodes())
      memoryNodes.push_back(&lambdaNode);

    for (auto & mallocNode : pointsToGraph.MallocNodes())
      memoryNodes.push_back(&mallocNode);

    for (auto & node : pointsToGraph.ImportNodes())
      memoryNodes.push_back(&node);

    return memoryNodes;
  };

  auto & unknownMemoryNode = pointsToGraph.GetUnknownMemoryNode();
  if (unknownMemoryNode.NumSources() == 0)
  {
    return;
  }

  auto memoryNodes = collectMemoryNodes(pointsToGraph);
  while (unknownMemoryNode.NumSources())
  {
    auto & source = *unknownMemoryNode.Sources().begin();
    for (auto & memoryNode : memoryNodes)
    {
      source.AddEdge(*memoryNode);
    }
    source.RemoveEdge(unknownMemoryNode);
  }
}

}
