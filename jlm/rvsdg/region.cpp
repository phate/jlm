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

/* argument */

argument::~argument() noexcept
{
  on_output_destroy(this);

  if (input())
    input()->arguments.erase(this);
}

argument::argument(
    jlm::rvsdg::region * region,
    jlm::rvsdg::structural_input * input,
    const jlm::rvsdg::port & port)
    : output(region, port),
      input_(input)
{
  if (input)
  {
    if (input->node() != region->node())
      throw jlm::util::error("Argument cannot be added to input.");

    input->arguments.push_back(this);
  }
}

argument::argument(
    jlm::rvsdg::region * region,
    jlm::rvsdg::structural_input * input,
    std::shared_ptr<const rvsdg::type> type)
    : output(region, std::move(type)),
      input_(input)
{
  if (input)
  {
    if (input->node() != region->node())
      throw jlm::util::error("Argument cannot be added to input.");

    input->arguments.push_back(this);
  }
}

jlm::rvsdg::argument *
argument::create(
    jlm::rvsdg::region * region,
    structural_input * input,
    const jlm::rvsdg::port & port)
{
  auto argument = new jlm::rvsdg::argument(region, input, port);
  region->append_argument(argument);
  return argument;
}

jlm::rvsdg::argument *
argument::create(
    jlm::rvsdg::region * region,
    structural_input * input,
    std::shared_ptr<const jlm::rvsdg::type> type)
{
  auto argument = new jlm::rvsdg::argument(region, input, std::move(type));
  region->append_argument(argument);
  return argument;
}

/* result */

result::~result() noexcept
{
  on_input_destroy(this);

  if (output())
    output()->results.erase(this);
}

result::result(
    jlm::rvsdg::region * region,
    jlm::rvsdg::output * origin,
    jlm::rvsdg::structural_output * output,
    const jlm::rvsdg::port & port)
    : input(origin, region, port),
      output_(output)
{
  if (output)
  {
    if (output->node() != region->node())
      throw jlm::util::error("Result cannot be added to output.");

    output->results.push_back(this);
  }
}

result::result(
    jlm::rvsdg::region * region,
    jlm::rvsdg::output * origin,
    jlm::rvsdg::structural_output * output,
    std::shared_ptr<const rvsdg::type> type)
    : input(origin, region, std::move(type)),
      output_(output)
{
  if (output)
  {
    if (output->node() != region->node())
      throw jlm::util::error("Result cannot be added to output.");

    output->results.push_back(this);
  }
}

jlm::rvsdg::result *
result::create(
    jlm::rvsdg::region * region,
    jlm::rvsdg::output * origin,
    jlm::rvsdg::structural_output * output,
    const jlm::rvsdg::port & port)
{
  auto result = new jlm::rvsdg::result(region, origin, output, port);
  region->append_result(result);
  return result;
}

jlm::rvsdg::result *
result::create(
    jlm::rvsdg::region * region,
    jlm::rvsdg::output * origin,
    jlm::rvsdg::structural_output * output,
    std::shared_ptr<const jlm::rvsdg::type> type)
{
  auto result = new jlm::rvsdg::result(region, origin, output, jlm::rvsdg::port(std::move(type)));
  region->append_result(result);
  return result;
}

/* region */

region::~region()
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

region::region(jlm::rvsdg::region * parent, jlm::rvsdg::graph * graph)
    : index_(0),
      graph_(graph),
      node_(nullptr)
{
  on_region_create(this);
}

region::region(jlm::rvsdg::structural_node * node, size_t index)
    : index_(index),
      graph_(node->graph()),
      node_(node)
{
  on_region_create(this);
}

void
region::append_argument(jlm::rvsdg::argument * argument)
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
region::RemoveArgument(size_t index)
{
  JLM_ASSERT(index < narguments());
  jlm::rvsdg::argument * argument = arguments_[index];

  delete argument;
  for (size_t n = index; n < arguments_.size() - 1; n++)
  {
    arguments_[n] = arguments_[n + 1];
    arguments_[n]->index_ = n;
  }
  arguments_.pop_back();
}

void
region::append_result(jlm::rvsdg::result * result)
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
region::RemoveResult(size_t index)
{
  JLM_ASSERT(index < results_.size());
  jlm::rvsdg::result * result = results_[index];

  delete result;
  for (size_t n = index; n < results_.size() - 1; n++)
  {
    results_[n] = results_[n + 1];
    results_[n]->index_ = n;
  }
  results_.pop_back();
}

void
region::remove_node(jlm::rvsdg::node * node)
{
  delete node;
}

void
region::copy(region * target, substitution_map & smap, bool copy_arguments, bool copy_results) const
{
  smap.insert(this, target);

  /* order nodes top-down */
  std::vector<std::vector<const jlm::rvsdg::node *>> context(nnodes());
  for (const auto & node : nodes)
  {
    JLM_ASSERT(node.depth() < context.size());
    context[node.depth()].push_back(&node);
  }

  /* copy arguments */
  if (copy_arguments)
  {
    for (size_t n = 0; n < narguments(); n++)
    {
      auto input = smap.lookup(argument(n)->input());
      auto narg = argument::create(target, input, argument(n)->port());
      smap.insert(argument(n), narg);
    }
  }

  /* copy nodes */
  for (size_t n = 0; n < context.size(); n++)
  {
    for (const auto node : context[n])
    {
      JLM_ASSERT(target == smap.lookup(node->region()));
      node->copy(target, smap);
    }
  }

  /* copy results */
  if (copy_results)
  {
    for (size_t n = 0; n < nresults(); n++)
    {
      auto origin = smap.lookup(result(n)->origin());
      if (!origin)
        origin = result(n)->origin();

      auto output = dynamic_cast<jlm::rvsdg::structural_output *>(smap.lookup(result(n)->output()));
      result::create(target, origin, output, result(n)->port());
    }
  }
}

void
region::prune(bool recursive)
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
region::normalize(bool recursive)
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
region::IsRootRegion() const noexcept
{
  return this->graph()->root() == this;
}

size_t
region::NumRegions(const jlm::rvsdg::region & region) noexcept
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
region::ToTree(const rvsdg::region & region, const util::AnnotationMap & annotationMap) noexcept
{
  std::stringstream stream;
  ToTree(region, annotationMap, 0, stream);
  return stream.str();
}

std::string
region::ToTree(const rvsdg::region & region) noexcept
{
  std::stringstream stream;
  util::AnnotationMap annotationMap;
  ToTree(region, annotationMap, 0, stream);
  return stream.str();
}

void
region::ToTree(
    const rvsdg::region & region,
    const util::AnnotationMap & annotationMap,
    size_t identationDepth,
    std::stringstream & stream) noexcept
{
  static const char identationChar = '-';
  static const char annotationSeparator = ' ';
  static const char labelValueSeparator = ':';

  auto identationString = std::string(identationDepth, identationChar);
  auto regionString =
      region.IsRootRegion() ? "RootRegion" : util::strfmt("Region[", region.index(), "]");
  auto regionAnnotationString =
      GetAnnotationString(&region, annotationMap, annotationSeparator, labelValueSeparator);

  // Convert current region to a string
  stream << identationString << regionString << regionAnnotationString << '\n';
  identationDepth += 1;

  // Convert the region's structural nodes with their subregions to a string
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
      stream << identationString << nodeString << annotationString << '\n';

      for (size_t n = 0; n < structuralNode->nsubregions(); n++)
      {
        ToTree(*structuralNode->subregion(n), annotationMap, identationDepth + 1, stream);
      }
    }
  }
}

std::string
region::GetAnnotationString(
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
region::ToString(
    const std::vector<util::Annotation> & annotations,
    char annotationSeparator,
    char labelValueSeparator)
{
  std::stringstream stream;
  for (auto it = annotations.begin(); it != annotations.end(); it++)
  {
    auto annotationString = ToString(*it, labelValueSeparator);
    stream << annotationSeparator << annotationString;
  }

  return stream.str();
}

std::string
region::ToString(const util::Annotation & annotation, char labelValueSeparator)
{
  std::string value;
  if (annotation.HasValueType<std::string_view>())
  {
    value = util::strfmt(annotation.Value<std::string_view>());
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
nnodes(const jlm::rvsdg::region * region) noexcept
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
nstructnodes(const jlm::rvsdg::region * region) noexcept
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
nsimpnodes(const jlm::rvsdg::region * region) noexcept
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
ninputs(const jlm::rvsdg::region * region) noexcept
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
