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
    jlm::rvsdg::structural_input * input,
    std::shared_ptr<const rvsdg::Type> type)
    : output(region, std::move(type)),
      input_(input)
{
  if (input)
  {
    if (input->node() != region->node())
      throw jlm::util::error("Argument cannot be added to input.");

    if (input->type() != *Type())
    {
      throw util::type_error(Type()->debug_string(), input->type().debug_string());
    }

    input->arguments.push_back(this);
  }
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
    jlm::rvsdg::structural_output * output,
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
      throw jlm::util::type_error(Type()->debug_string(), output->Type()->debug_string());
    }

    output->results.push_back(this);
  }
}

Region::~Region() noexcept
{
  on_region_destroy(this);

  while (results_.size())
    RemoveResult(results_.size() - 1);

  prune(false);
  JLM_ASSERT(nodes.empty());
  JLM_ASSERT(top_nodes.empty());
  JLM_ASSERT(bottom_nodes.empty());

  while (arguments_.size())
    RemoveArgument(arguments_.size() - 1);
}

Region::Region(rvsdg::Region * parent, jlm::rvsdg::graph * graph)
    : index_(0),
      graph_(graph),
      node_(nullptr)
{
  on_region_create(this);
}

Region::Region(jlm::rvsdg::structural_node * node, size_t index)
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
Region::remove_node(jlm::rvsdg::node * node)
{
  delete node;
}

void
Region::copy(Region * target, SubstitutionMap & smap, bool copy_arguments, bool copy_results) const
{
  smap.insert(this, target);

  // order nodes top-down
  std::vector<std::vector<const jlm::rvsdg::node *>> context(nnodes());
  for (const auto & node : nodes)
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
      auto newOutput = dynamic_cast<structural_output *>(smap.lookup(oldResult->output()));
      oldResult->Copy(*newOrigin, newOutput);
    }
  }
}

void
Region::prune(bool recursive)
{
  while (bottom_nodes.first())
    remove_node(bottom_nodes.first());

  if (!recursive)
    return;

  for (const auto & node : nodes)
  {
    if (auto snode = dynamic_cast<const jlm::rvsdg::structural_node *>(&node))
    {
      for (size_t n = 0; n < snode->nsubregions(); n++)
        snode->subregion(n)->prune(recursive);
    }
  }
}

void
Region::normalize(bool recursive)
{
  for (auto node : jlm::rvsdg::topdown_traverser(this))
  {
    if (auto structnode = dynamic_cast<const jlm::rvsdg::structural_node *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
        structnode->subregion(n)->normalize(recursive);
    }

    const auto & op = node->operation();
    graph()->node_normal_form(typeid(op))->normalize_node(node);
  }
}

bool
Region::IsRootRegion() const noexcept
{
  return this->graph()->root() == this;
}

size_t
Region::NumRegions(const rvsdg::Region & region) noexcept
{
  size_t numRegions = 1;
  for (auto & node : region.nodes)
  {
    if (auto structuralNode = dynamic_cast<const jlm::rvsdg::structural_node *>(&node))
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
  for (auto & node : region.nodes)
  {
    if (auto structuralNode = dynamic_cast<const rvsdg::structural_node *>(&node))
    {
      auto nodeString = structuralNode->operation().debug_string();
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
  for (const auto & node : region->nodes)
  {
    if (auto snode = dynamic_cast<const jlm::rvsdg::structural_node *>(&node))
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
  for (const auto & node : region->nodes)
  {
    if (auto snode = dynamic_cast<const jlm::rvsdg::structural_node *>(&node))
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
  for (const auto & node : region->nodes)
  {
    if (auto snode = dynamic_cast<const jlm::rvsdg::structural_node *>(&node))
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
  for (const auto & node : region->nodes)
  {
    if (auto snode = dynamic_cast<const jlm::rvsdg::structural_node *>(&node))
    {
      for (size_t r = 0; r < snode->nsubregions(); r++)
        n += ninputs(snode->subregion(r));
    }
    n += node.ninputs();
  }

  return n;
}

} // namespace
