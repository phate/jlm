/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/DotWriter.hpp>
#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/AnnotationMap.hpp>
#include <jlm/util/file.hpp>
#include <jlm/util/Program.hpp>
#include <jlm/util/strfmt.hpp>

#include <fstream>

namespace jlm::rvsdg
{

RegionArgument::~RegionArgument() noexcept
{
  if (input())
    input()->arguments.erase(this);
}

RegionArgument::RegionArgument(
    rvsdg::Region * region,
    StructuralInput * input,
    std::shared_ptr<const rvsdg::Type> type)
    : Output(region, std::move(type)),
      input_(input)
{
  if (input)
  {
    if (input->node() != region->node())
      throw util::Error("Argument cannot be added to input.");

    if (*input->Type() != *Type())
    {
      throw util::TypeError(Type()->debug_string(), input->Type()->debug_string());
    }

    input->arguments.push_back(this);
  }
}

std::string
RegionArgument::debug_string() const
{
  return util::strfmt("a", index());
}

RegionArgument &
RegionArgument::Copy(Region & region, StructuralInput * input)
{
  return RegionArgument::Create(region, input, Type());
}

RegionArgument &
RegionArgument::Create(
    rvsdg::Region & region,
    StructuralInput * input,
    std::shared_ptr<const rvsdg::Type> type)
{
  return region.addArgument(std::make_unique<RegionArgument>(&region, input, std::move(type)));
}

RegionResult::~RegionResult() noexcept
{
  if (output())
    output()->results.erase(this);
}

RegionResult::RegionResult(
    rvsdg::Region * region,
    jlm::rvsdg::Output * origin,
    StructuralOutput * output,
    std::shared_ptr<const rvsdg::Type> type)
    : Input(*region, *origin, std::move(type)),
      output_(output)
{
  if (output)
  {
    if (output->node() != region->node())
      throw util::Error("Result cannot be added to output.");

    if (*Type() != *output->Type())
    {
      throw jlm::util::TypeError(Type()->debug_string(), output->Type()->debug_string());
    }

    output->results.push_back(this);
  }
}

std::string
RegionResult::debug_string() const
{
  return util::strfmt("r", index());
}

RegionResult &
RegionResult::Copy(rvsdg::Output & origin, StructuralOutput * output)
{
  return RegionResult::Create(*origin.region(), origin, output, origin.Type());
}

RegionResult &
RegionResult::Create(
    rvsdg::Region & region,
    rvsdg::Output & origin,
    StructuralOutput * output,
    std::shared_ptr<const rvsdg::Type> type)
{
  return region.addResult(
      std::make_unique<RegionResult>(&region, &origin, output, std::move(type)));
}

Region::~Region() noexcept
{
  while (results_.size())
    RemoveResult(results_.size() - 1);

  prune(false);
  JLM_ASSERT(numNodes() == 0);
  JLM_ASSERT(numTopNodes() == 0);
  JLM_ASSERT(numBottomNodes() == 0);

  while (arguments_.size())
    RemoveArgument(arguments_.size() - 1);

  // Disconnect observers
  while (observers_)
  {
    RegionObserver * head = observers_;
    observers_ = head->next_;
    head->pprev_ = &head->next_;
    head->next_ = nullptr;
  }
}

Region::Region(Region *, Graph * graph)
    : index_(0),
      graph_(graph),
      nextNodeId_(0),
      node_(nullptr),
      numTopNodes_(0),
      numBottomNodes_(0),
      numNodes_(0)
{}

Region::Region(rvsdg::StructuralNode * node, size_t index)
    : index_(index),
      graph_(node->graph()),
      nextNodeId_(0),
      node_(node),
      numTopNodes_(0),
      numBottomNodes_(0),
      numNodes_(0)
{}

bool
Region::IsRootRegion() const noexcept
{
  return &this->graph()->GetRootRegion() == this;
}

RegionArgument &
Region::addArgument(std::unique_ptr<RegionArgument> argument)
{
  if (argument->region() != this)
    throw util::Error("Appending argument to wrong region.");

  argument->index_ = narguments();
  arguments_.push_back(argument.release());
  return *arguments_.back();
}

RegionArgument &
Region::insertArgument(size_t index, std::unique_ptr<RegionArgument> argument)
{
  if (argument->region() != this)
    throw util::Error("Inserting argument to wrong region.");

  if (index > narguments())
    throw util::Error("Inserting argument after end of region.");

  arguments_.push_back(nullptr);

  // Move everything at index or above one index up
  for (size_t n = narguments() - 1; n > index; n--)
  {
    arguments_[n] = arguments_[n - 1];
    arguments_[n]->index_ = n;
  }
  arguments_[index] = argument.release();
  arguments_[index]->index_ = index;

  return *arguments_[index];
}

void
Region::RemoveArgument(size_t index)
{
  JLM_ASSERT(index < narguments());
  RegionArgument * argument = arguments_[index];

  delete argument;
  for (size_t n = index; n < arguments_.size() - 1; n++)
  {
    arguments_[n] = arguments_[n + 1];
    arguments_[n]->index_ = n;
  }
  arguments_.pop_back();
}

RegionResult &
Region::addResult(std::unique_ptr<RegionResult> result)
{
  const auto resultPtr = result.release();

  if (resultPtr->region() != this)
    throw util::Error("Appending result to wrong region.");

  resultPtr->index_ = nresults();
  results_.push_back(resultPtr);

  notifyInputCreate(resultPtr);

  return *resultPtr;
}

void
Region::RemoveResult(size_t index)
{
  JLM_ASSERT(index < results_.size());
  RegionResult * result = results_[index];

  notifyInputDestory(result);

  delete result;
  for (size_t n = index; n < results_.size() - 1; n++)
  {
    results_[n] = results_[n + 1];
    results_[n]->index_ = n;
  }
  results_.pop_back();
}

void
Region::removeNode(Node * node)
{
  JLM_ASSERT(node->region() == this);
  // The node's destructor handles informing the region about removal
  delete node;
}

void
Region::copy(Region * target, SubstitutionMap & smap, bool copy_arguments, bool copy_results) const
{
  smap.insert(this, target);

  if (copy_arguments)
  {
    for (const auto oldArgument : Arguments())
    {
      const auto input = smap.lookup(oldArgument->input());
      auto & newArgument = oldArgument->Copy(*target, input);
      smap.insert(oldArgument, &newArgument);
    }
  }

  // copy nodes
  for (const auto node : TopDownConstTraverser(this))
  {
    JLM_ASSERT(target == smap.lookup(node->region()));
    node->copy(target, smap);
  }

  if (copy_results)
  {
    for (const auto oldResult : Results())
    {
      const auto newOrigin = smap.lookup(oldResult->origin());
      JLM_ASSERT(newOrigin != nullptr);
      const auto newOutput = dynamic_cast<StructuralOutput *>(smap.lookup(oldResult->output()));
      oldResult->Copy(*newOrigin, newOutput);
    }
  }
}

void
Region::prune(bool recursive)
{
  while (bottomNodes_.first())
    removeNode(bottomNodes_.first());

  if (!recursive)
    return;

  for (const auto & node : Nodes())
  {
    if (auto snode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (size_t n = 0; n < snode->nsubregions(); n++)
        snode->subregion(n)->prune(recursive);
    }
  }
}

void
Region::view() const
{
  DotWriter dotWriter;
  util::graph::Writer graphWriter;
  dotWriter.WriteGraph(graphWriter, *this);

  const util::FilePath outputFilePath =
      util::FilePath::createUniqueFileName(util::FilePath::TempDirectoryPath(), "region-", ".dot");

  std::ofstream outputFile(outputFilePath.to_str());
  graphWriter.outputAllGraphs(outputFile, util::graph::OutputFormat::Dot);

  util::executeProgramAndWait(util::getDotViewer(), { outputFilePath.to_str() });
}

void
Region::onTopNodeAdded(Node & node)
{
  JLM_ASSERT(node.region() == this);
  JLM_ASSERT(node.ninputs() == 0);
  topNodes_.push_back(&node);
  numTopNodes_++;
}

void
Region::onTopNodeRemoved(Node & node)
{
  JLM_ASSERT(node.region() == this);
  topNodes_.erase(&node);
  numTopNodes_--;
}

void
Region::onBottomNodeAdded(Node & node)
{
  JLM_ASSERT(node.region() == this);
  JLM_ASSERT(node.IsDead());
  bottomNodes_.push_back(&node);
  numBottomNodes_++;
}

void
Region::onBottomNodeRemoved(Node & node)
{
  JLM_ASSERT(node.region() == this);
  bottomNodes_.erase(&node);
  numBottomNodes_--;
}

void
Region::onNodeAdded(Node & node)
{
  JLM_ASSERT(node.region() == this);
  nodes_.push_back(&node);
  numNodes_++;
}

void
Region::onNodeRemoved(Node & node)
{
  JLM_ASSERT(node.region() == this);
  nodes_.erase(&node);
  numNodes_--;
}

void
Region::notifyNodeCreate(Node * node)
{
  for (auto observer = observers_; observer; observer = observer->next_)
  {
    observer->onNodeCreate(node);
  }
}

void
Region::notifyNodeDestroy(Node * node)
{
  for (auto observer = observers_; observer; observer = observer->next_)
  {
    observer->onNodeDestroy(node);
  }
}

void
Region::notifyInputCreate(Input * input)
{
  for (auto observer = observers_; observer; observer = observer->next_)
  {
    observer->onInputCreate(input);
  }
}

void
Region::notifyInputChange(Input * input, Output * old_origin, Output * new_origin)
{
  for (auto observer = observers_; observer; observer = observer->next_)
  {
    observer->onInputChange(input, old_origin, new_origin);
  }
}

void
Region::notifyInputDestory(Input * input)
{
  for (auto observer = observers_; observer; observer = observer->next_)
  {
    observer->onInputDestroy(input);
  }
}

size_t
Region::NumRegions(const rvsdg::Region & region) noexcept
{
  size_t numRegions = 1;
  for (auto & node : region.Nodes())
  {
    if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (size_t n = 0; n < structuralNode->nsubregions(); n++)
      {
        numRegions += NumRegions(*structuralNode->subregion(n));
      }
    }
  }

  return numRegions;
}

std::string
Region::ToTree(const rvsdg::Region & region, const util::AnnotationMap & annotationMap) noexcept
{
  std::stringstream stream;
  ToTree(region, annotationMap, 0, stream);
  return stream.str();
}

std::string
Region::ToTree(const rvsdg::Region & region) noexcept
{
  std::stringstream stream;
  util::AnnotationMap annotationMap;
  ToTree(region, annotationMap, 0, stream);
  return stream.str();
}

void
Region::ToTree(
    const rvsdg::Region & region,
    const util::AnnotationMap & annotationMap,
    size_t indentationDepth,
    std::stringstream & stream) noexcept
{
  static const char indentationChar = '-';
  static const char annotationSeparator = ' ';
  static const char labelValueSeparator = ':';

  // Convert current region to a string
  auto indentationString = std::string(indentationDepth, indentationChar);
  auto regionString =
      region.IsRootRegion() ? "RootRegion" : util::strfmt("Region[", region.index(), "]");
  auto regionAnnotationString =
      GetAnnotationString(&region, annotationMap, annotationSeparator, labelValueSeparator);

  stream << indentationString << regionString << regionAnnotationString << '\n';

  // Convert the region's structural nodes with their subregions to a string
  indentationDepth++;
  indentationString = std::string(indentationDepth, indentationChar);
  for (auto & node : region.Nodes())
  {
    if (auto structuralNode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      auto nodeString = structuralNode->DebugString();
      auto annotationString = GetAnnotationString(
          structuralNode,
          annotationMap,
          annotationSeparator,
          labelValueSeparator);
      stream << indentationString << nodeString << annotationString << '\n';

      for (size_t n = 0; n < structuralNode->nsubregions(); n++)
      {
        ToTree(*structuralNode->subregion(n), annotationMap, indentationDepth + 1, stream);
      }
    }
  }
}

std::string
Region::GetAnnotationString(
    const void * key,
    const util::AnnotationMap & annotationMap,
    char annotationSeparator,
    char labelValueSeparator)
{
  if (!annotationMap.HasAnnotations(key))
    return "";

  auto & annotations = annotationMap.GetAnnotations(key);
  return ToString(annotations, annotationSeparator, labelValueSeparator);
}

std::string
Region::ToString(
    const std::vector<util::Annotation> & annotations,
    char annotationSeparator,
    char labelValueSeparator)
{
  std::stringstream stream;
  for (auto & annotation : annotations)
  {
    auto annotationString = ToString(annotation, labelValueSeparator);
    stream << annotationSeparator << annotationString;
  }

  return stream.str();
}

std::string
Region::ToString(const util::Annotation & annotation, char labelValueSeparator)
{
  std::string value;
  if (annotation.HasValueType<std::string>())
  {
    value = annotation.Value<std::string>();
  }
  else if (annotation.HasValueType<int64_t>())
  {
    value = util::strfmt(annotation.Value<int64_t>());
  }
  else if (annotation.HasValueType<uint64_t>())
  {
    value = util::strfmt(annotation.Value<uint64_t>());
  }
  else if (annotation.HasValueType<double>())
  {
    value = util::strfmt(annotation.Value<double>());
  }
  else
  {
    JLM_UNREACHABLE("Unhandled annotation type.");
  }

  return util::strfmt(annotation.Label(), labelValueSeparator, value);
}

RegionObserver::~RegionObserver() noexcept
{
  *pprev_ = next_;
  if (next_)
  {
    next_->pprev_ = pprev_;
  }
}

RegionObserver::RegionObserver(const Region & region)
{
  next_ = region.observers_;
  if (next_)
  {
    next_->pprev_ = &next_;
  }
  pprev_ = &region.observers_;
  region.observers_ = this;
}

std::unordered_map<const Node *, size_t>
computeDepthMap(const Region & region)
{
  std::unordered_map<const Node *, size_t> depthMap;
  for (const auto node : TopDownConstTraverser(&region))
  {
    size_t depth = 0;
    for (auto & input : node->Inputs())
    {
      if (const auto owner = TryGetOwnerNode<Node>(*input.origin()))
      {
        depth = std::max(depth, depthMap[owner] + 1);
      }
    }
    depthMap[node] = depth;
  }

  return depthMap;
}

size_t
nnodes(const jlm::rvsdg::Region * region) noexcept
{
  size_t n = region->numNodes();
  for (const auto & node : region->Nodes())
  {
    if (auto snode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (size_t r = 0; r < snode->nsubregions(); r++)
        n += nnodes(snode->subregion(r));
    }
  }

  return n;
}

size_t
nstructnodes(const rvsdg::Region * region) noexcept
{
  size_t n = 0;
  for (const auto & node : region->Nodes())
  {
    if (auto snode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (size_t r = 0; r < snode->nsubregions(); r++)
        n += nstructnodes(snode->subregion(r));
      n += 1;
    }
  }

  return n;
}

size_t
nsimpnodes(const rvsdg::Region * region) noexcept
{
  size_t n = 0;
  for (const auto & node : region->Nodes())
  {
    if (auto snode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (size_t r = 0; r < snode->nsubregions(); r++)
        n += nsimpnodes(snode->subregion(r));
    }
    else
    {
      n += 1;
    }
  }

  return n;
}

size_t
ninputs(const rvsdg::Region * region) noexcept
{
  size_t n = region->nresults();
  for (const auto & node : region->Nodes())
  {
    if (auto snode = dynamic_cast<const rvsdg::StructuralNode *>(&node))
    {
      for (size_t r = 0; r < snode->nsubregions(); r++)
        n += ninputs(snode->subregion(r));
    }
    n += node.ninputs();
  }

  return n;
}

} // namespace
