/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/CallSummary.hpp>
#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/operators/IOBarrier.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/alias-analyses/PointsToGraph.hpp>
#include <jlm/llvm/opt/alias-analyses/Steensgaard.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/Statistics.hpp>

namespace jlm::llvm::aa
{

/**
 * Determines whether \p output%s type is or contains a pointer type.
 *
 * @param output An rvsdg::output.
 * @return True if \p output%s type is or contains a pointer type, otherwise false.
 */
static bool
HasOrContainsPointerType(const rvsdg::Output & output)
{
  return IsOrContains<PointerType>(*output.Type()) || is<rvsdg::FunctionType>(output.Type());
}

/**
 * Determines whether \p node should be handled by the Steensgaard analysis.
 *
 * @param node An rvsdg::SimpleNode.
 * @return True if \p node should be handled, otherwise false.
 */
static bool
ShouldHandle(const rvsdg::SimpleNode & node)
{
  for (size_t n = 0; n < node.ninputs(); n++)
  {
    auto & origin = *node.input(n)->origin();
    if (HasOrContainsPointerType(origin))
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
      const jlm::rvsdg::Output & output,
      PointsToFlags pointsToFlags)
      : Location(pointsToFlags),
        HasEscaped_(false),
        Output_(&output)
  {}

  [[nodiscard]] const jlm::rvsdg::Output &
  GetOutput() const noexcept
  {
    return *Output_;
  }

  /** Determines whether the register location escapes the module.
   *
   * @return True, if the location escapes the module, otherwise false.
   */
  [[nodiscard]] bool
  HasEscaped() const noexcept
  {
    return HasEscaped_;
  }

  void
  MarkAsEscaped() noexcept
  {
    HasEscaped_ = true;
  }

  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    auto node = rvsdg::TryGetOwnerNode<rvsdg::Node>(*Output_);
    auto index = Output_->index();

    if (jlm::rvsdg::is<rvsdg::SimpleOperation>(node))
    {
      auto nodestr = node->DebugString();
      auto outputstr = Output_->Type()->debug_string();
      return jlm::util::strfmt(nodestr, ":", index, "[" + outputstr + "]");
    }

    if (auto node = rvsdg::TryGetRegionParentNode<rvsdg::LambdaNode>(*Output_))
    {
      auto dbgstr = node->DebugString();
      if (auto ctxvar = node->MapBinderContextVar(*Output_))
      {
        // Bound context variable.
        return jlm::util::strfmt(dbgstr, ":cv:", index);
      }
      else
      {
        // Formal function argument.
        return jlm::util::strfmt(dbgstr, ":arg:", index);
      }
    }

    if (rvsdg::TryGetRegionParentNode<DeltaNode>(*Output_))
    {
      auto dbgstr = Output_->region()->node()->DebugString();
      return jlm::util::strfmt(dbgstr, ":cv:", index);
    }

    if (rvsdg::TryGetRegionParentNode<rvsdg::GammaNode>(*Output_))
    {
      auto dbgstr = Output_->region()->node()->DebugString();
      return jlm::util::strfmt(dbgstr, ":arg", index);
    }

    if (rvsdg::TryGetRegionParentNode<rvsdg::ThetaNode>(*Output_))
    {
      auto dbgstr = Output_->region()->node()->DebugString();
      return jlm::util::strfmt(dbgstr, ":arg", index);
    }

    if (const auto thetaNode = rvsdg::TryGetOwnerNode<rvsdg::ThetaNode>(*Output_))
    {
      auto dbgstr = thetaNode->DebugString();
      return jlm::util::strfmt(dbgstr, ":out", index);
    }

    if (auto node = rvsdg::TryGetOwnerNode<rvsdg::GammaNode>(*Output_))
    {
      auto dbgstr = node->DebugString();
      return jlm::util::strfmt(dbgstr, ":out", index);
    }

    if (auto graphImport = dynamic_cast<const GraphImport *>(Output_))
    {
      return jlm::util::strfmt("imp:", graphImport->Name());
    }

    if (auto phi = rvsdg::TryGetRegionParentNode<rvsdg::PhiNode>(*Output_))
    {
      auto dbgstr = phi->DebugString();
      auto var = phi->MapArgument(*Output_);
      if (auto fix = std::get_if<rvsdg::PhiNode::FixVar>(&var))
      {
        return jlm::util::strfmt(dbgstr, ":rvarg", fix->output->index());
      }
      else if (auto ctx = std::get_if<rvsdg::PhiNode::ContextVar>(&var))
      {
        return jlm::util::strfmt(dbgstr, ":cvarg", ctx->input->index());
      }
    }

    return jlm::util::strfmt(
        rvsdg::TryGetOwnerNode<rvsdg::Node>(*Output_)->DebugString(),
        ":",
        index);
  }

  [[nodiscard]] static bool
  HasEscaped(const Location & location) noexcept
  {
    auto registerLocation = dynamic_cast<const RegisterLocation *>(&location);
    return registerLocation && registerLocation->HasEscaped();
  }

  static std::unique_ptr<RegisterLocation>
  Create(const jlm::rvsdg::Output & output, PointsToFlags pointsToFlags)
  {
    return std::make_unique<RegisterLocation>(output, pointsToFlags);
  }

private:
  bool HasEscaped_;
  const jlm::rvsdg::Output * Output_;
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

  explicit AllocaLocation(const rvsdg::Node & node)
      : MemoryLocation(),
        Node_(node)
  {
    JLM_ASSERT(is<AllocaOperation>(&node));
  }

public:
  [[nodiscard]] const rvsdg::Node &
  GetNode() const noexcept
  {
    return Node_;
  }

  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    return Node_.DebugString();
  }

  static std::unique_ptr<Location>
  Create(const rvsdg::Node & node)
  {
    return std::unique_ptr<Location>(new AllocaLocation(node));
  }

private:
  const rvsdg::Node & Node_;
};

/** \brief MallocLocation class
 *
 * This class represents an abstract heap location allocated by a malloc operation.
 */
class MallocLocation final : public MemoryLocation
{
  ~MallocLocation() override = default;

  explicit MallocLocation(const rvsdg::Node & node)
      : MemoryLocation(),
        Node_(node)
  {
    JLM_ASSERT(is<MallocOperation>(&node));
  }

public:
  [[nodiscard]] const rvsdg::Node &
  GetNode() const noexcept
  {
    return Node_;
  }

  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    return Node_.DebugString();
  }

  static std::unique_ptr<Location>
  Create(const rvsdg::Node & node)
  {
    return std::unique_ptr<Location>(new MallocLocation(node));
  }

private:
  const rvsdg::Node & Node_;
};

/** \brief LambdaLocation class
 *
 * This class represents an abstract function location, statically allocated by a lambda operation.
 */
class LambdaLocation final : public MemoryLocation
{
  ~LambdaLocation() override = default;

  constexpr explicit LambdaLocation(const rvsdg::LambdaNode & lambda)
      : MemoryLocation(),
        Lambda_(lambda)
  {}

public:
  [[nodiscard]] const rvsdg::LambdaNode &
  GetNode() const noexcept
  {
    return Lambda_;
  }

  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    return Lambda_.DebugString();
  }

  static std::unique_ptr<Location>
  Create(const rvsdg::LambdaNode & node)
  {
    return std::unique_ptr<Location>(new LambdaLocation(node));
  }

private:
  const rvsdg::LambdaNode & Lambda_;
};

/** \brief DeltaLocation class
 *
 * This class represents an abstract global variable location, statically allocated by a delta
 * operation.
 */
class DeltaLocation final : public MemoryLocation
{
  ~DeltaLocation() override = default;

  constexpr explicit DeltaLocation(const DeltaNode & delta)
      : MemoryLocation(),
        Delta_(delta)
  {}

public:
  [[nodiscard]] const DeltaNode &
  GetNode() const noexcept
  {
    return Delta_;
  }

  [[nodiscard]] std::string
  DebugString() const noexcept override
  {
    return Delta_.DebugString();
  }

  static std::unique_ptr<Location>
  Create(const DeltaNode & node)
  {
    return std::unique_ptr<Location>(new DeltaLocation(node));
  }

private:
  const DeltaNode & Delta_;
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

  ImportLocation(const GraphImport & graphImport, PointsToFlags pointsToFlags)
      : MemoryLocation(),
        Argument_(graphImport)
  {
    SetPointsToFlags(pointsToFlags);
  }

public:
  [[nodiscard]] const GraphImport &
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
  Create(const GraphImport & graphImport)
  {
    JLM_ASSERT(is<PointerType>(graphImport.Type()) || is<rvsdg::FunctionType>(graphImport.Type()));

    // If the imported memory location is a pointer type or contains a pointer type, then these
    // pointers can point to values that escaped this module.
    bool isOrContainsPointerType = IsOrContains<PointerType>(*graphImport.ValueType());

    return std::unique_ptr<Location>(new ImportLocation(
        graphImport,
        isOrContainsPointerType
            ? PointsToFlags::PointsToExternalMemory | PointsToFlags::PointsToEscapedMemory
            : PointsToFlags::PointsToNone));
  }

private:
  const GraphImport & Argument_;
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

using DisjointLocationSet = util::DisjointSet<Location *>;

/** \brief Context class
 */
class Steensgaard::Context final
{
public:
  using DisjointLocationSetConstRange =
      util::IteratorRange<const DisjointLocationSet::set_iterator>;

  ~Context() = default;

  Context() = default;

  Context(const Context &) = delete;

  Context(Context &&) = delete;

  Context &
  operator=(const Context &) = delete;

  Context &
  operator=(Context &&) = delete;

  [[nodiscard]] DisjointLocationSetConstRange
  Sets() const
  {
    return { DisjointLocationSet_.begin(), DisjointLocationSet_.end() };
  }

  Location &
  InsertAllocaLocation(const rvsdg::Node & node)
  {
    Locations_.push_back(AllocaLocation::Create(node));
    auto location = Locations_.back().get();
    DisjointLocationSet_.insert(location);

    return *location;
  }

  Location &
  InsertMallocLocation(const rvsdg::Node & node)
  {
    Locations_.push_back(MallocLocation::Create(node));
    auto location = Locations_.back().get();
    DisjointLocationSet_.insert(location);

    return *location;
  }

  Location &
  InsertLambdaLocation(const rvsdg::LambdaNode & lambda)
  {
    Locations_.push_back(LambdaLocation::Create(lambda));
    auto location = Locations_.back().get();
    DisjointLocationSet_.insert(location);

    return *location;
  }

  Location &
  InsertDeltaLocation(const DeltaNode & delta)
  {
    Locations_.push_back(DeltaLocation::Create(delta));
    auto location = Locations_.back().get();
    DisjointLocationSet_.insert(location);

    return *location;
  }

  Location &
  InsertImportLocation(const GraphImport & graphImport)
  {
    Locations_.push_back(ImportLocation::Create(graphImport));
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

  /**
   * Returns the set's root location for register \p output if output's location exists.
   * Otherwise creates a new location for register \p output with PointsToFlags::PointsToNone.
   *
   * @param output A register.
   * @return A Location.
   *
   * \note It is deliberate that only GetOrInsertRegisterLocation() instead of also
   * InsertRegisterLocation() is exposed for inserting register locations into the context. The
   * reason is that at a lot of places in the analysis it would only be correct to use
   * GetOrInsertRegisterLocation(). The problem originates from direct call nodes in combination
   * with recursive functions. For direct call nodes, we merge the locations of the call node
   * results with the respective origin of the lambda result in the disjoint set. Now, for the call
   * result in a single recursive functions, it potentially could be that we need to merge it with
   * the respective origin of the lambda result even though we have not handled this origin in
   * the analysis yet. In other words, we would need to create the respective location for the
   * origin of the lambda result when handling the call node result instead of creating it when we
   * would later handle the origin of the lambda result. As the origin of the lambda result could
   * originate from "any" node, i.e., any pointer output of a simple node, gamma outputs, theta
   * outputs, etc., it is safer to simply just use GetOrInsertRegisterLocation() everywhere
   * and be done with the problem.
   *
   * Here is a contrived example to further illustrate the issue:
   *
   * ... = phi [a1]
   *   o4 = lambda[o1 <= a1]
   *     ...
   *     o2 ... = call a1 ...
   *     o3 ... = load o2 ...
   *   [o3]
   * [o4]
   *
   * The analysis handles nodes top-down. This means that we would handle the call node before the
   * load node. For the direct recursive call node, the analysis traces the function input to the
   * lambda and tries to merge the call result \a o2 with the origin of the respective lambda
   * result \a o3. However, the location for \a o3 has not been created yet (as the load node was
   * not handled yet) and it would need to create the location in order to be able to merge. After
   * the call node, the analysis would continue with the load node. Handling the load node requires
   * to create a location for its output, but this location was already created and we should just
   * return the already created location. In other cases, the output of the load node might not have
   * been created yet and we would need to do so. By always utilizing GetOrInsertRegisterLocation()
   * instead of exposing InsertRegisterLocation(), we take care of this created-before-handled
   * problem automatically.
   */
  Location &
  GetOrInsertRegisterLocation(const rvsdg::Output & output)
  {
    if (auto it = LocationMap_.find(&output); it != LocationMap_.end())
      return GetRootLocation(*it->second);

    return InsertRegisterLocation(output, PointsToFlags::PointsToNone);
  }

  /**
   * Returns the disjoint set for Location \p location.
   *
   * @param location A Location.
   * @return The disjoint set the location is part of.
   */
  const DisjointLocationSet::Set &
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

  /**
   * Returns the set's root location for Location \p location.
   *
   * @param location A Location.
   * @return The set's root location.
   */
  Location &
  GetRootLocation(Location & location) const
  {
    return *GetSet(location).value();
  }

  /**
   * Returns the set's root location of which the \p output register location is part of.
   *
   * @param output A register.
   * @return The root Location of the set.
   */
  Location &
  GetLocation(const rvsdg::Output & output)
  {
    return GetRootLocation(GetRegisterLocation(output));
  }

  /**
   * Returns the register location for register \p output.
   *
   * @param output A register.
   * @return A RegisterLocation.
   */
  RegisterLocation &
  GetRegisterLocation(const rvsdg::Output & output)
  {
    auto it = LocationMap_.find(&output);
    JLM_ASSERT(it != LocationMap_.end());
    return *it->second;
  }

  /**
   * Checks whether a register location exists for register \p output.
   *
   * @param output A register.
   * @return True if the location exists, otherwise false.
   */
  bool
  HasRegisterLocation(const rvsdg::Output & output)
  {
    auto it = LocationMap_.find(&output);
    return it != LocationMap_.end();
  }

  std::string
  ToDot() const
  {
    auto toDotNode = [](const DisjointLocationSet::Set & set)
    {
      auto rootLocation = set.value();

      std::string setLabel;
      for (auto & location : set)
      {
        auto unknownLabel = location->PointsToUnknownMemory() ? "{U}" : "";
        auto pointsToEscapedMemoryLabel = location->PointsToEscapedMemory() ? "{E}" : "";
        auto escapesModuleLabel = RegisterLocation::HasEscaped(*location) ? "{EscapesModule}" : "";
        auto pointsToLabel = jlm::util::strfmt("{pt:", (intptr_t)location->GetPointsTo(), "}");
        auto locationLabel = jlm::util::strfmt((intptr_t)location, " : ", location->DebugString());

        if (location == rootLocation)
        {
          setLabel += jlm::util::strfmt(
              "*",
              locationLabel,
              unknownLabel,
              pointsToEscapedMemoryLabel,
              escapesModuleLabel,
              pointsToLabel,
              "*\\n");
        }
        else
        {
          setLabel += jlm::util::strfmt(
              locationLabel,
              unknownLabel,
              pointsToEscapedMemoryLabel,
              escapesModuleLabel,
              "\\n");
        }
      }

      return jlm::util::strfmt("{ ", (intptr_t)&set, " [label = \"", setLabel, "\"]; }");
    };

    auto toDotEdge =
        [](const DisjointLocationSet::Set & set, const DisjointLocationSet::Set & pointsToSet)
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

  static std::unique_ptr<Context>
  Create()
  {
    return std::make_unique<Context>();
  }

private:
  RegisterLocation &
  InsertRegisterLocation(const jlm::rvsdg::Output & output, PointsToFlags pointsToFlags)
  {
    JLM_ASSERT(!HasRegisterLocation(output));

    auto registerLocation = RegisterLocation::Create(output, pointsToFlags);
    auto registerLocationPointer = registerLocation.get();

    LocationMap_[&output] = registerLocationPointer;
    DisjointLocationSet_.insert(registerLocationPointer);
    Locations_.push_back(std::move(registerLocation));

    return *registerLocationPointer;
  }

  Location &
  Merge(Location & location1, Location & location2)
  {
    return *DisjointLocationSet_.merge(&location1, &location2)->value();
  }

  DisjointLocationSet DisjointLocationSet_;
  std::vector<std::unique_ptr<Location>> Locations_;
  std::unordered_map<const jlm::rvsdg::Output *, RegisterLocation *> LocationMap_;
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

  explicit Statistics(const util::FilePath & sourceFile)
      : util::Statistics(Statistics::Id::SteensgaardAnalysis, sourceFile)
  {}

  void
  StartSteensgaardStatistics(const rvsdg::Graph & graph) noexcept
  {
    AddMeasurement(Label::NumRvsdgNodes, rvsdg::nnodes(&graph.GetRootRegion()));
    AddTimer(AnalysisTimerLabel_).start();
  }

  void
  StopSteensgaardStatistics() noexcept
  {
    GetTimer(AnalysisTimerLabel_).stop();
  }

  void
  StartPointsToFlagsPropagationStatistics(const Context & context) noexcept
  {
    AddMeasurement(NumDisjointSetsLabel_, context.NumDisjointSets());
    AddMeasurement(NumLocationsLabel_, context.NumLocations());
    AddTimer(PointsToFlagsPropagationTimerLabel_).start();
  }

  void
  StopPointsToFlagsPropagationStatistics() noexcept
  {
    GetTimer(PointsToFlagsPropagationTimerLabel_).stop();
  }

  void
  StartPointsToGraphConstructionStatistics()
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
  Create(const util::FilePath & sourceFile)
  {
    return std::make_unique<Statistics>(sourceFile);
  }
};

Steensgaard::~Steensgaard() = default;

Steensgaard::Steensgaard() = default;

void
Steensgaard::AnalyzeSimpleNode(const jlm::rvsdg::SimpleNode & node)
{
  if (is<AllocaOperation>(&node))
  {
    AnalyzeAlloca(node);
  }
  else if (is<MallocOperation>(&node))
  {
    AnalyzeMalloc(node);
  }
  else if (is<LoadOperation>(&node))
  {
    AnalyzeLoad(node);
  }
  else if (is<StoreOperation>(&node))
  {
    AnalyzeStore(node);
  }
  else if (is<CallOperation>(&node))
  {
    AnalyzeCall(node);
  }
  else if (is<GetElementPtrOperation>(&node))
  {
    AnalyzeGep(node);
  }
  else if (is<BitCastOperation>(&node))
  {
    AnalyzeBitcast(node);
  }
  else if (is<IntegerToPointerOperation>(&node))
  {
    AnalyzeBits2ptr(node);
  }
  else if (is<PtrToIntOperation>(&node))
  {
    AnalyzePtrToInt(node);
  }
  else if (is<ConstantPointerNullOperation>(&node))
  {
    AnalyzeConstantPointerNull(node);
  }
  else if (is<UndefValueOperation>(&node))
  {
    AnalyzeUndef(node);
  }
  else if (is<MemCpyOperation>(&node))
  {
    AnalyzeMemcpy(node);
  }
  else if (is<ConstantArrayOperation>(&node))
  {
    AnalyzeConstantArray(node);
  }
  else if (is<ConstantStruct>(&node))
  {
    AnalyzeConstantStruct(node);
  }
  else if (is<ConstantAggregateZeroOperation>(&node))
  {
    AnalyzeConstantAggregateZero(node);
  }
  else if (is<ExtractValueOperation>(&node))
  {
    AnalyzeExtractValue(node);
  }
  else if (is<VariadicArgumentListOperation>(&node))
  {
    AnalyzeVaList(node);
  }
  else if (is<PointerToFunctionOperation>(&node))
  {
    AnalyzePointerToFunction(node);
  }
  else if (is<FunctionToPointerOperation>(&node))
  {
    AnalyzeFunctionToPointer(node);
  }
  else if (is<IOBarrierOperation>(&node))
  {
    AnalyzeIOBarrier(node);
  }
  else if (is<FreeOperation>(&node) || is<PtrCmpOperation>(&node))
  {
    // Nothing needs to be done as FreeOperation and PtrCmpOperation do not affect points-to sets
  }
  else
  {
    // Ensure that we took care of all pointer consuming nodes.
    JLM_ASSERT(!ShouldHandle(node));
  }
}

void
Steensgaard::AnalyzeAlloca(const jlm::rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<AllocaOperation>(&node));

  auto & allocaOutputLocation = Context_->GetOrInsertRegisterLocation(*node.output(0));
  auto & allocaLocation = Context_->InsertAllocaLocation(node);
  allocaOutputLocation.SetPointsTo(allocaLocation);
}

void
Steensgaard::AnalyzeMalloc(const jlm::rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<MallocOperation>(&node));

  auto & mallocOutputLocation = Context_->GetOrInsertRegisterLocation(*node.output(0));
  auto & mallocLocation = Context_->InsertMallocLocation(node);
  mallocOutputLocation.SetPointsTo(mallocLocation);
}

void
Steensgaard::AnalyzeLoad(const rvsdg::SimpleNode & node)
{
  auto & result = LoadOperation::LoadedValueOutput(node);
  auto & address = *LoadOperation::AddressInput(node).origin();

  if (!HasOrContainsPointerType(result))
    return;

  auto & addressLocation = Context_->GetLocation(address);
  auto & resultLocation = Context_->GetOrInsertRegisterLocation(result);
  resultLocation.SetPointsToFlags(
      resultLocation.GetPointsToFlags() | addressLocation.GetPointsToFlags());

  if (addressLocation.GetPointsTo() == nullptr)
  {
    addressLocation.SetPointsTo(resultLocation);
  }
  else
  {
    Context_->Join(resultLocation, *addressLocation.GetPointsTo());
  }
}

void
Steensgaard::AnalyzeStore(const rvsdg::SimpleNode & node)
{
  auto & address = *StoreOperation::AddressInput(node).origin();
  auto & value = *StoreOperation::StoredValueInput(node).origin();

  if (!HasOrContainsPointerType(value))
    return;

  auto & addressLocation = Context_->GetLocation(address);
  auto & valueLocation = Context_->GetLocation(value);

  if (addressLocation.GetPointsTo() == nullptr)
  {
    addressLocation.SetPointsTo(valueLocation);
  }
  else
  {
    Context_->Join(*addressLocation.GetPointsTo(), valueLocation);
  }
}

void
Steensgaard::AnalyzeCall(const rvsdg::SimpleNode & callNode)
{
  JLM_ASSERT(is<CallOperation>(&callNode));

  auto callTypeClassifier = CallOperation::ClassifyCall(callNode);
  switch (callTypeClassifier->GetCallType())
  {
  case CallTypeClassifier::CallType::NonRecursiveDirectCall:
  case CallTypeClassifier::CallType::RecursiveDirectCall:
    AnalyzeDirectCall(
        callNode,
        rvsdg::AssertGetOwnerNode<rvsdg::LambdaNode>(callTypeClassifier->GetLambdaOutput()));
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
Steensgaard::AnalyzeDirectCall(
    const rvsdg::SimpleNode & callNode,
    const rvsdg::LambdaNode & lambdaNode)
{
  JLM_ASSERT(is<CallOperation>(&callNode));

  auto & lambdaFunctionType = lambdaNode.GetOperation().type();
  auto & callFunctionType = *CallOperation::GetFunctionInput(callNode).Type();
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

  // Handle call node operands
  //
  // Variadic arguments are taken care of in AnalyzeVaList().
  auto arguments = lambdaNode.GetFunctionArguments();
  for (size_t n = 1; n < callNode.ninputs(); n++)
  {
    auto & callArgument = *callNode.input(n)->origin();
    auto & lambdaArgument = *arguments[n - 1];

    if (HasOrContainsPointerType(callArgument))
    {
      auto & callArgumentLocation = Context_->GetLocation(callArgument);
      auto & lambdaArgumentLocation = Context_->GetOrInsertRegisterLocation(lambdaArgument);

      Context_->Join(callArgumentLocation, lambdaArgumentLocation);
    }
  }

  // Handle call node results
  auto subregion = lambdaNode.subregion();
  JLM_ASSERT(subregion->nresults() == callNode.noutputs());
  for (size_t n = 0; n < callNode.noutputs(); n++)
  {
    auto & callResult = *callNode.output(n);
    auto & lambdaResult = *subregion->result(n)->origin();

    if (HasOrContainsPointerType(callResult))
    {
      auto & callResultLocation = Context_->GetOrInsertRegisterLocation(callResult);
      auto & lambdaResultLocation = Context_->GetOrInsertRegisterLocation(lambdaResult);

      Context_->Join(callResultLocation, lambdaResultLocation);
    }
  }
}

void
Steensgaard::AnalyzeExternalCall(const rvsdg::SimpleNode & callNode)
{
  JLM_ASSERT(is<CallOperation>(&callNode));

  // Mark arguments of external function call as escaped
  //
  // Variadic arguments are taken care of in AnalyzeVaList().
  for (size_t n = 1; n < CallOperation::NumArguments(callNode); n++)
  {
    auto & callArgument = *callNode.input(n)->origin();

    if (HasOrContainsPointerType(callArgument))
    {
      MarkAsEscaped(callArgument);
    }
  }

  // Mark results of external function call as pointing to escaped and external
  for (size_t n = 0; n < callNode.noutputs(); n++)
  {
    auto & callResult = *callNode.output(n);

    if (HasOrContainsPointerType(callResult))
    {
      auto & callResultLocation = Context_->GetOrInsertRegisterLocation(callResult);
      callResultLocation.SetPointsToFlags(
          callResultLocation.GetPointsToFlags() | PointsToFlags::PointsToExternalMemory
          | PointsToFlags::PointsToEscapedMemory);
    }
  }
}

void
Steensgaard::AnalyzeIndirectCall(const rvsdg::SimpleNode & callNode)
{
  JLM_ASSERT(is<CallOperation>(&callNode));

  // Nothing can be done for the call/lambda arguments, as it is
  // an indirect call and the lambda node cannot be retrieved.
  //
  // Variadic arguments are taken care of in AnalyzeVaList().

  // Handle call node results
  for (size_t n = 0; n < callNode.noutputs(); n++)
  {
    auto & callResult = *callNode.output(n);

    if (HasOrContainsPointerType(callResult))
    {
      auto & callResultLocation = Context_->GetOrInsertRegisterLocation(callResult);
      callResultLocation.SetPointsToFlags(
          callResultLocation.GetPointsToFlags() | PointsToFlags::PointsToUnknownMemory
          | PointsToFlags::PointsToExternalMemory | PointsToFlags::PointsToEscapedMemory);
    }
  }
}

void
Steensgaard::AnalyzeGep(const jlm::rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<GetElementPtrOperation>(&node));

  auto & base = Context_->GetLocation(*node.input(0)->origin());
  auto & value = Context_->GetOrInsertRegisterLocation(*node.output(0));

  Context_->Join(base, value);
}

void
Steensgaard::AnalyzeBitcast(const jlm::rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<BitCastOperation>(&node));

  auto & operand = *node.input(0)->origin();
  auto & result = *node.output(0);

  if (HasOrContainsPointerType(operand))
  {
    auto & operandLocation = Context_->GetLocation(operand);
    auto & resultLocation = Context_->GetOrInsertRegisterLocation(result);

    Context_->Join(operandLocation, resultLocation);
  }
}

void
Steensgaard::AnalyzeBits2ptr(const jlm::rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<IntegerToPointerOperation>(&node));

  auto & registerLocation = Context_->GetOrInsertRegisterLocation(*node.output(0));
  registerLocation.SetPointsToFlags(
      registerLocation.GetPointsToFlags()
      // The register location already points to unknown memory. Unknown memory is a superset of
      // escaped memory, and therefore we can simply set escaped memory to false.
      | PointsToFlags::PointsToUnknownMemory | PointsToFlags::PointsToExternalMemory);
}

void
Steensgaard::AnalyzePtrToInt(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<PtrToIntOperation>(&node));

  MarkAsEscaped(*node.input(0)->origin());
}

void
Steensgaard::AnalyzeExtractValue(const jlm::rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<ExtractValueOperation>(&node));

  auto & result = *node.output(0);

  if (HasOrContainsPointerType(result))
  {
    // FIXME: Have a look at this operation again to ensure that the flags add up.
    auto & registerLocation = Context_->GetOrInsertRegisterLocation(result);
    registerLocation.SetPointsToFlags(
        registerLocation.GetPointsToFlags() | PointsToFlags::PointsToUnknownMemory
        | PointsToFlags::PointsToExternalMemory);
  }
}

void
Steensgaard::AnalyzeConstantPointerNull(const jlm::rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<ConstantPointerNullOperation>(&node));

  // ConstantPointerNull cannot point to any memory location. We therefore only insert a register
  // node for it, but let this node not point to anything.
  Context_->GetOrInsertRegisterLocation(*node.output(0));
}

void
Steensgaard::AnalyzeConstantAggregateZero(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<ConstantAggregateZeroOperation>(&node));
  auto & output = *node.output(0);

  if (HasOrContainsPointerType(output))
  {
    // ConstantAggregateZero cannot point to any memory location. We therefore only insert a
    // register node for it, but let this node not point to anything.
    Context_->GetOrInsertRegisterLocation(output);
  }
}

void
Steensgaard::AnalyzeUndef(const jlm::rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<UndefValueOperation>(&node));
  auto & output = *node.output(0);

  if (HasOrContainsPointerType(output))
  {
    // UndefValue cannot point to any memory location. We therefore only insert a register node for
    // it, but let this node not point to anything.
    Context_->GetOrInsertRegisterLocation(output);
  }
}

void
Steensgaard::AnalyzeConstantArray(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<ConstantArrayOperation>(&node));

  auto & output = *node.output(0);
  if (!HasOrContainsPointerType(output))
    return;

  auto & outputLocation = Context_->GetOrInsertRegisterLocation(output);

  for (size_t n = 0; n < node.ninputs(); n++)
  {
    auto & operand = *node.input(n)->origin();
    JLM_ASSERT(HasOrContainsPointerType(operand));

    auto & operandLocation = Context_->GetLocation(operand);
    Context_->Join(outputLocation, operandLocation);
  }
}

void
Steensgaard::AnalyzeConstantStruct(const jlm::rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<ConstantStruct>(&node));

  auto & output = *node.output(0);
  if (!HasOrContainsPointerType(output))
    return;

  auto & outputLocation = Context_->GetOrInsertRegisterLocation(output);

  for (size_t n = 0; n < node.ninputs(); n++)
  {
    auto & operand = *node.input(n)->origin();
    if (HasOrContainsPointerType(operand))
    {
      auto & operandLocation = Context_->GetLocation(operand);
      Context_->Join(outputLocation, operandLocation);
    }
  }
}

void
Steensgaard::AnalyzeMemcpy(const jlm::rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<MemCpyOperation>(&node));

  auto & dstAddress = Context_->GetLocation(*node.input(0)->origin());
  auto & srcAddress = Context_->GetLocation(*node.input(1)->origin());

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
    auto & underlyingMemory = Context_->InsertDummyLocation();
    srcAddress.SetPointsTo(underlyingMemory);
  }

  if (dstAddress.GetPointsTo() == nullptr)
  {
    dstAddress.SetPointsTo(*srcAddress.GetPointsTo());
  }
  else
  {
    // Unifies the underlying memory of srcMemory and dstMemory
    Context_->Join(*srcAddress.GetPointsTo(), *dstAddress.GetPointsTo());
  }
}

void
Steensgaard::AnalyzeVaList(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<VariadicArgumentListOperation>(&node));

  // Members of the valist are extracted using the va_arg macro, which loads from the va_list struct
  // on the stack. This struct will be marked as escaped from the call to va_start, and thus point
  // to external. All we need to do is mark all pointees of pointer varargs as escaping. When the
  // pointers are re-created inside the function, they will be marked as pointing to external.

  for (size_t n = 0; n < node.ninputs(); n++)
  {
    auto & origin = *node.input(n)->origin();

    if (HasOrContainsPointerType(origin))
    {
      MarkAsEscaped(origin);
    }
  }
}

void
Steensgaard::AnalyzeFunctionToPointer(const rvsdg::SimpleNode & node)
{
  auto & outputLocation = Context_->GetOrInsertRegisterLocation(*node.output(0));
  auto & originLocation = Context_->GetOrInsertRegisterLocation(*node.input(0)->origin());
  Context_->Join(outputLocation, originLocation);
}

void
Steensgaard::AnalyzeIOBarrier(const rvsdg::SimpleNode & node)
{
  JLM_ASSERT(is<IOBarrierOperation>(&node));
  const auto & origin = *node.input(0)->origin();
  const auto & output = *node.output(0);

  if (!HasOrContainsPointerType(origin))
    return;

  auto & originLocation = Context_->GetOrInsertRegisterLocation(origin);
  auto & outputLocation = Context_->GetOrInsertRegisterLocation(output);
  Context_->Join(originLocation, outputLocation);
}

void
Steensgaard::AnalyzePointerToFunction(const rvsdg::SimpleNode & node)
{
  auto & outputLocation = Context_->GetOrInsertRegisterLocation(*node.output(0));
  auto & originLocation = Context_->GetOrInsertRegisterLocation(*node.input(0)->origin());
  Context_->Join(outputLocation, originLocation);
}

void
Steensgaard::AnalyzeLambda(const rvsdg::LambdaNode & lambda)
{
  // Handle context variables
  for (const auto & cv : lambda.GetContextVars())
  {
    auto & origin = *cv.input->origin();

    if (HasOrContainsPointerType(origin))
    {
      auto & originLocation = Context_->GetLocation(origin);
      auto & argumentLocation = Context_->GetOrInsertRegisterLocation(*cv.inner);
      Context_->Join(originLocation, argumentLocation);
    }
  }

  // Handle function arguments
  auto callSummary = ComputeCallSummary(lambda);
  if (callSummary.HasOnlyDirectCalls())
  {
    for (auto & argument : lambda.GetFunctionArguments())
    {
      if (HasOrContainsPointerType(*argument))
      {
        Context_->GetOrInsertRegisterLocation(*argument);
      }
    }
  }
  else
  {
    // FIXME: We also end up in this case when the lambda has only direct calls, but is exported.
    for (auto argument : lambda.GetFunctionArguments())
    {
      if (HasOrContainsPointerType(*argument))
      {
        auto & argumentLocation = Context_->GetOrInsertRegisterLocation(*argument);
        argumentLocation.SetPointsToFlags(
            argumentLocation.GetPointsToFlags() | PointsToFlags::PointsToExternalMemory
            | PointsToFlags::PointsToEscapedMemory);
      }
    }
  }

  AnalyzeRegion(*lambda.subregion());

  // Handle function results
  if (callSummary.IsExported())
  {
    for (auto result : lambda.GetFunctionResults())
    {
      auto & operand = *result->origin();

      if (HasOrContainsPointerType(operand))
      {
        MarkAsEscaped(operand);
      }
    }
  }

  // Handle function
  auto & lambdaOutputLocation = Context_->GetOrInsertRegisterLocation(*lambda.output());
  auto & lambdaLocation = Context_->InsertLambdaLocation(lambda);
  lambdaOutputLocation.SetPointsTo(lambdaLocation);
}

void
Steensgaard::AnalyzeDelta(const DeltaNode & delta)
{
  // Handle context variables
  for (auto & ctxVar : delta.GetContextVars())
  {
    auto & origin = *ctxVar.input->origin();

    if (HasOrContainsPointerType(origin))
    {
      auto & originLocation = Context_->GetLocation(origin);
      auto & argumentLocation = Context_->GetOrInsertRegisterLocation(*ctxVar.inner);
      Context_->Join(originLocation, argumentLocation);
    }
  }

  AnalyzeRegion(*delta.subregion());

  auto & deltaOutputLocation = Context_->GetOrInsertRegisterLocation(delta.output());
  auto & deltaLocation = Context_->InsertDeltaLocation(delta);
  deltaOutputLocation.SetPointsTo(deltaLocation);

  auto & origin = *delta.result().origin();
  if (HasOrContainsPointerType(origin))
  {
    auto & originLocation = Context_->GetLocation(origin);
    Context_->Join(deltaLocation, originLocation);
  }
}

void
Steensgaard::AnalyzePhi(const rvsdg::PhiNode & phi)
{
  // Handle context variables
  for (auto cv : phi.GetContextVars())
  {
    auto & origin = *cv.input->origin();

    if (HasOrContainsPointerType(origin))
    {
      auto & originLocation = Context_->GetLocation(origin);
      auto & argumentLocation = Context_->GetOrInsertRegisterLocation(*cv.inner);
      Context_->Join(originLocation, argumentLocation);
    }
  }

  // Create Register PointerObjects for each fixpoint variable argument
  for (auto var : phi.GetFixVars())
  {
    auto & argument = *var.recref;

    if (HasOrContainsPointerType(argument))
    {
      Context_->GetOrInsertRegisterLocation(argument);
    }
  }

  AnalyzeRegion(*phi.subregion());

  // Handle recursion variable outputs
  for (auto var : phi.GetFixVars())
  {
    auto & argument = *var.recref;
    auto & output = *var.output;
    auto & result = *var.result;

    if (HasOrContainsPointerType(argument))
    {
      auto & originLocation = Context_->GetLocation(*result.origin());
      auto & argumentLocation = Context_->GetLocation(argument);
      auto & outputLocation = Context_->GetOrInsertRegisterLocation(output);

      Context_->Join(originLocation, argumentLocation);
      Context_->Join(argumentLocation, outputLocation);
    }
  }
}

void
Steensgaard::AnalyzeGamma(const rvsdg::GammaNode & node)
{
  // Handle entry variables
  for (const auto & ev : node.GetEntryVars())
  {
    auto & origin = *ev.input->origin();

    if (HasOrContainsPointerType(origin))
    {
      auto & originLocation = Context_->GetLocation(*ev.input->origin());
      for (auto argument : ev.branchArgument)
      {
        auto & argumentLocation = Context_->GetOrInsertRegisterLocation(*argument);
        Context_->Join(argumentLocation, originLocation);
      }
    }
  }

  // Handle subregions
  for (size_t n = 0; n < node.nsubregions(); n++)
    AnalyzeRegion(*node.subregion(n));

  // Handle exit variables
  for (auto ex : node.GetExitVars())
  {
    if (HasOrContainsPointerType(*ex.output))
    {
      auto & outputLocation = Context_->GetOrInsertRegisterLocation(*ex.output);
      for (auto result : ex.branchResult)
      {
        auto & resultLocation = Context_->GetLocation(*result->origin());
        Context_->Join(outputLocation, resultLocation);
      }
    }
  }
}

void
Steensgaard::AnalyzeTheta(const rvsdg::ThetaNode & theta)
{
  for (const auto & loopVar : theta.GetLoopVars())
  {
    if (HasOrContainsPointerType(*loopVar.output))
    {
      auto & originLocation = Context_->GetLocation(*loopVar.input->origin());
      auto & argumentLocation = Context_->GetOrInsertRegisterLocation(*loopVar.pre);

      Context_->Join(argumentLocation, originLocation);
    }
  }

  AnalyzeRegion(*theta.subregion());

  for (const auto & loopVar : theta.GetLoopVars())
  {
    if (HasOrContainsPointerType(*loopVar.output))
    {
      auto & originLocation = Context_->GetLocation(*loopVar.post->origin());
      auto & argumentLocation = Context_->GetLocation(*loopVar.pre);
      auto & outputLocation = Context_->GetOrInsertRegisterLocation(*loopVar.output);

      Context_->Join(originLocation, argumentLocation);
      Context_->Join(originLocation, outputLocation);
    }
  }
}

void
Steensgaard::AnalyzeStructuralNode(const rvsdg::StructuralNode & node)
{
  if (auto lambdaNode = dynamic_cast<const rvsdg::LambdaNode *>(&node))
  {
    AnalyzeLambda(*lambdaNode);
  }
  else if (auto deltaNode = dynamic_cast<const DeltaNode *>(&node))
  {
    AnalyzeDelta(*deltaNode);
  }
  else if (auto gammaNode = dynamic_cast<const rvsdg::GammaNode *>(&node))
  {
    AnalyzeGamma(*gammaNode);
  }
  else if (auto thetaNode = dynamic_cast<const rvsdg::ThetaNode *>(&node))
  {
    AnalyzeTheta(*thetaNode);
  }
  else if (auto phiNode = dynamic_cast<const rvsdg::PhiNode *>(&node))
  {
    AnalyzePhi(*phiNode);
  }
  else
  {
    JLM_UNREACHABLE("Unhandled node type!");
  }
}

void
Steensgaard::AnalyzeRegion(rvsdg::Region & region)
{
  // Check that we added a RegisterLocation for each required argument
  for (size_t n = 0; n < region.narguments(); n++)
  {
    auto & argument = *region.argument(n);
    if (HasOrContainsPointerType(argument))
    {
      JLM_ASSERT(Context_->HasRegisterLocation(argument));
    }
  }

  using namespace jlm::rvsdg;

  TopDownTraverser traverser(&region);
  for (auto & node : traverser)
  {
    if (auto simpleNode = dynamic_cast<const SimpleNode *>(node))
    {
      AnalyzeSimpleNode(*simpleNode);
    }
    else if (auto structuralNode = dynamic_cast<const StructuralNode *>(node))
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
Steensgaard::AnalyzeRvsdg(const rvsdg::Graph & graph)
{
  AnalyzeImports(graph);
  AnalyzeRegion(graph.GetRootRegion());
  AnalyzeExports(graph);
}

void
Steensgaard::AnalyzeImports(const rvsdg::Graph & graph)
{
  auto rootRegion = &graph.GetRootRegion();
  for (size_t n = 0; n < rootRegion->narguments(); n++)
  {
    auto & graphImport = *util::AssertedCast<const GraphImport>(rootRegion->argument(n));

    if (HasOrContainsPointerType(graphImport))
    {
      auto & importLocation = Context_->InsertImportLocation(graphImport);
      auto & registerLocation = Context_->GetOrInsertRegisterLocation(graphImport);
      registerLocation.SetPointsTo(importLocation);
    }
  }
}

void
Steensgaard::AnalyzeExports(const rvsdg::Graph & graph)
{
  auto rootRegion = &graph.GetRootRegion();

  for (size_t n = 0; n < rootRegion->nresults(); n++)
  {
    auto & result = *rootRegion->result(n);
    MarkAsEscaped(*result.origin());
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
    const rvsdg::RvsdgModule & module,
    util::StatisticsCollector & statisticsCollector)
{
  // std::unordered_map<const rvsdg::output *, std::string> outputMap;
  // std::cout << jlm::rvsdg::view(module.Rvsdg().root(), outputMap) << std::flush;

  Context_ = Context::Create();
  auto statistics = Statistics::Create(module.SourceFilePath().value());

  // Perform Steensgaard analysis
  statistics->StartSteensgaardStatistics(module.Rvsdg());
  AnalyzeRvsdg(module.Rvsdg());
  // std::cout << Context_->ToDot() << std::flush;
  statistics->StopSteensgaardStatistics();

  // Propagate points-to flags in disjoint location set graph
  statistics->StartPointsToFlagsPropagationStatistics(*Context_);
  PropagatePointsToFlags();
  statistics->StopPointsToFlagsPropagationStatistics();

  // Construct PointsTo graph
  statistics->StartPointsToGraphConstructionStatistics();
  auto pointsToGraph = ConstructPointsToGraph();
  // std::cout << PointsToGraph::ToDot(*pointsToGraph, outputMap) << std::flush;
  statistics->StopPointsToGraphConstructionStatistics(*pointsToGraph);

  // Redirect unknown memory node sources
  statistics->StartUnknownMemoryNodeSourcesRedirectionStatistics();
  RedirectUnknownMemoryNodeSources(*pointsToGraph);
  statistics->StopUnknownMemoryNodeSourcesRedirectionStatistics();

  statisticsCollector.CollectDemandedStatistics(std::move(statistics));

  // Discard internal state to free up memory after we are done with the analysis
  Context_.reset();

  return pointsToGraph;
}

void
Steensgaard::MarkAsEscaped(const rvsdg::Output & output)
{
  auto & outputLocation = Context_->GetRegisterLocation(output);
  outputLocation.MarkAsEscaped();

  auto & rootLocation = Context_->GetRootLocation(outputLocation);
  if (!rootLocation.GetPointsTo())
  {
    auto & dummyLocation = Context_->InsertDummyLocation();
    rootLocation.SetPointsTo(dummyLocation);
  }

  rootLocation.GetPointsTo()->SetPointsToFlags(
      PointsToFlags::PointsToEscapedMemory | PointsToFlags::PointsToExternalMemory);
}

void
Steensgaard::PropagatePointsToFlags()
{
  bool pointsToFlagsChanged = false;
  do
  {
    pointsToFlagsChanged = false;

    for (auto & set : Context_->Sets())
    {
      auto location = set.value();

      // Nothing needs to be done if this set does not point to another set
      if (!location->GetPointsTo())
      {
        continue;
      }
      auto & pointsToLocation = Context_->GetRootLocation(*location->GetPointsTo());

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
        const util::DisjointSet<Location *>::Set *,
        std::vector<PointsToGraph::MemoryNode *>> & memoryNodesInSet) const
{
  // Initialize working set
  util::HashSet<Location *> toVisit;
  for (auto registerLocation : escapingRegisterLocations.Items())
  {
    auto & set = Context_->GetSet(*registerLocation);
    if (auto pointsToLocation = set.value()->GetPointsTo())
    {
      toVisit.Insert(pointsToLocation);
    }
  }

  // Collect escaped memory nodes
  util::HashSet<PointsToGraph::MemoryNode *> escapedMemoryNodes;
  util::HashSet<const DisjointLocationSet::Set *> visited;
  while (!toVisit.IsEmpty())
  {
    auto moduleEscapingLocation = *toVisit.Items().begin();
    toVisit.Remove(moduleEscapingLocation);

    auto & set = Context_->GetSet(*moduleEscapingLocation);

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

  // All the memory nodes within a set
  std::unordered_map<const DisjointLocationSet::Set *, std::vector<PointsToGraph::MemoryNode *>>
      memoryNodesInSet;

  // All register locations that are marked as RegisterLocation::HasEscaped()
  util::HashSet<RegisterLocation *> escapingRegisterLocations;

  // Mapping between locations and points-to graph nodes
  std::unordered_map<const Location *, PointsToGraph::Node *> locationMap;

  // Create points-to graph nodes
  for (auto & locationSet : Context_->Sets())
  {
    memoryNodesInSet[&locationSet] = {};

    util::HashSet<const rvsdg::Output *> registers;
    util::HashSet<RegisterLocation *> registerLocations;
    for (auto & location : locationSet)
    {
      if (auto registerLocation = dynamic_cast<RegisterLocation *>(location))
      {
        registers.Insert(&registerLocation->GetOutput());
        registerLocations.Insert(registerLocation);

        if (registerLocation->HasEscaped())
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
  for (auto & set : Context_->Sets())
  {
    bool pointsToUnknown = set.value()->PointsToUnknownMemory();
    bool pointsToExternalMemory = set.value()->PointsToExternalMemory();
    bool pointsToEscapedMemory = set.value()->PointsToEscapedMemory();

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

      if (pointsToUnknown && !Location::Is<LambdaLocation>(*location))
        pointsToGraphNode.AddEdge(pointsToGraph->GetUnknownMemoryNode());

      if (pointsToExternalMemory && !Location::Is<LambdaLocation>(*location))
        pointsToGraphNode.AddEdge(pointsToGraph->GetExternalMemoryNode());

      if (pointsToEscapedMemory && !Location::Is<LambdaLocation>(*location))
      {
        for (auto & escapedMemoryNode : escapedMemoryNodes.Items())
          pointsToGraphNode.AddEdge(*escapedMemoryNode);
      }

      // Add edges to all memory nodes the location points to
      if (auto pointsToLocation = set.value()->GetPointsTo())
      {
        auto & pointsToSet = Context_->GetSet(*pointsToLocation);
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
