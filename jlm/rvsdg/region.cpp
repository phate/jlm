/*
 * Copyright 2010 2011 2012 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2012 2013 2014 2015 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/notifiers.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/util/AnnotationMap.hpp>

namespace jlm::rvsdg
{

RegionArgument::~RegionArgument() noexcept
{
  on_output_destroy(this);

  if (input())
    input()->arguments.erase(this);
}

RegionArgument::RegionArgument(
    rvsdg::Region * region,
    StructuralInput * input,
    std::shared_ptr<const rvsdg::Type> type)
    : output(region, std::move(type)),
      input_(input)
{
  if (input)
  {
    if (input->node() != region->node())
      throw jlm::util::error("Argument cannot be added to input.");

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

[[nodiscard]] std::variant<Node *, Region *>
RegionArgument::GetOwner() const noexcept
{
  return region();
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
  auto argument = new RegionArgument(&region, input, std::move(type));
  region.append_argument(argument);
  return *argument;
}

RegionResult::~RegionResult() noexcept
{
  on_input_destroy(this);

  if (output())
    output()->results.erase(this);
}

RegionResult::RegionResult(
    rvsdg::Region * region,
    jlm::rvsdg::output * origin,
    StructuralOutput * output,
    std::shared_ptr<const rvsdg::Type> type)
    : input(origin, region, std::move(type)),
      output_(output)
{
  if (output)
  {
    if (output->node() != region->node())
      throw jlm::util::error("Result cannot be added to output.");

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

[[nodiscard]] std::variant<Node *, Region *>
RegionResult::GetOwner() const noexcept
{
  return region();
}

RegionResult &
RegionResult::Copy(rvsdg::output & origin, StructuralOutput * output)
{
  return RegionResult::Create(*origin.region(), origin, output, origin.Type());
}

RegionResult &
RegionResult::Create(
    rvsdg::Region & region,
    rvsdg::output & origin,
    StructuralOutput * output,
    std::shared_ptr<const rvsdg::Type> type)
{
  JLM_ASSERT(origin.region() == &region);
  auto result = new RegionResult(&region, &origin, output, std::move(type));
  region.append_result(result);
  return *result;
}

Region::~Region() noexcept
{
  on_region_destroy(this);

  while (results_.size())
    RemoveResult(results_.size() - 1);

  prune(false);
  JLM_ASSERT(nnodes() == 0);
  JLM_ASSERT(NumTopNodes() == 0);
  JLM_ASSERT(NumBottomNodes() == 0);

  while (arguments_.size())
    RemoveArgument(arguments_.size() - 1);
}

Region::Region(Region *, Graph * graph)
    : index_(0),
      graph_(graph),
      node_(nullptr)
{
  on_region_create(this);
}

Region::Region(rvsdg::StructuralNode * node, size_t index)
    : index_(index),
      graph_(node->graph()),
      node_(node)
{
  on_region_create(this);
}

void
Region::append_argument(RegionArgument * argument)
{
  if (argument->region() != this)
    throw jlm::util::error("Appending argument to wrong region.");

  auto index = argument->index();
  JLM_ASSERT(index == 0);
  if (index != 0 || (index == 0 && narguments() > 0 && this->argument(0) == argument))
    return;

  argument->index_ = narguments();
  arguments_.push_back(argument);
  on_output_create(argument);
}

void
Region::insert_argument(size_t index, RegionArgument * argument)
{
  if (argument->region() != this)
    throw jlm::util::error("Inserting argument to wrong region.");

  JLM_ASSERT(argument->index() == 0);

  argument->index_ = index;
  arguments_.insert(arguments_.begin() + index, argument);
  for (size_t n = index + 1; n < arguments_.size(); ++n)
    arguments_[n]->index_ = n;
  on_output_create(argument);
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

void
Region::append_result(RegionResult * result)
{
  if (result->region() != this)
    throw jlm::util::error("Appending result to wrong region.");

  /*
    Check if result was already appended to this region. This check
    relies on the fact that an unappended result has an index of zero.
  */
  auto index = result->index();
  JLM_ASSERT(index == 0);
  if (index != 0 || (index == 0 && nresults() > 0 && this->result(0) == result))
    return;

  result->index_ = nresults();
  results_.push_back(result);
  on_input_create(result);
}

void
Region::RemoveResult(size_t index)
{
  JLM_ASSERT(index < results_.size());
  RegionResult * result = results_[index];

  delete result;
  for (size_t n = index; n < results_.size() - 1; n++)
  {
    results_[n] = results_[n + 1];
    results_[n]->index_ = n;
  }
  results_.pop_back();
}

void
Region::remove_node(Node * node)
{
  delete node;
}

bool
Region::AddTopNode(Node & node)
{
  if (node.region() != this)
    return false;

  if (node.ninputs() != 0)
    return false;

  // FIXME: We should check that a node is not already part of the top nodes before adding it.
  TopNodes_.push_back(&node);

  return true;
}

bool
Region::AddBottomNode(Node & node)
{
  if (node.region() != this)
    return false;

  if (!node.IsDead())
    return false;

  // FIXME: We should check that a node is not already part of the bottom nodes before adding it.
  BottomNodes_.push_back(&node);

  return true;
}

bool
Region::AddNode(Node & node)
{
  if (node.region() != this)
    return false;

  Nodes_.push_back(&node);

  return true;
}

bool
Region::RemoveBottomNode(Node & node)
{
  auto numBottomNodes = NumBottomNodes();
  BottomNodes_.erase(&node);
  return numBottomNodes != NumBottomNodes();
}

bool
Region::RemoveTopNode(Node & node)
{
  auto numTopNodes = NumTopNodes();
  TopNodes_.erase(&node);
  return numTopNodes != NumTopNodes();
}

bool
Region::RemoveNode(Node & node)
{
  auto numNodes = nnodes();
  Nodes_.erase(&node);
  return numNodes != nnodes();
}

void
Region::copy(Region * target, SubstitutionMap & smap, bool copy_arguments, bool copy_results) const
{
  smap.insert(this, target);

  // order nodes top-down
  std::vector<std::vector<const Node *>> context(nnodes());
  for (const auto & node : Nodes())
  {
    JLM_ASSERT(node.depth() < context.size());
    context[node.depth()].push_back(&node);
  }

  if (copy_arguments)
  {
    for (size_t n = 0; n < narguments(); n++)
    {
      auto oldArgument = argument(n);
      auto input = smap.lookup(oldArgument->input());
      auto & newArgument = oldArgument->Copy(*target, input);
      smap.insert(oldArgument, &newArgument);
    }
  }

  // copy nodes
  for (size_t n = 0; n < context.size(); n++)
  {
    for (const auto node : context[n])
    {
      JLM_ASSERT(target == smap.lookup(node->region()));
      node->copy(target, smap);
    }
  }

  if (copy_results)
  {
    for (size_t n = 0; n < nresults(); n++)
    {
      auto oldResult = result(n);
      auto newOrigin = smap.lookup(oldResult->origin());
      JLM_ASSERT(newOrigin != nullptr);
      auto newOutput = dynamic_cast<StructuralOutput *>(smap.lookup(oldResult->output()));
      oldResult->Copy(*newOrigin, newOutput);
    }
  }
}

void
Region::prune(bool recursive)
{
  while (BottomNodes_.first())
    remove_node(BottomNodes_.first());

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

bool
Region::IsRootRegion() const noexcept
{
  return &this->graph()->GetRootRegion() == this;
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

size_t
nnodes(const jlm::rvsdg::Region * region) noexcept
{
  size_t n = region->nnodes();
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
