/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_AGGREGATION_ANNOTATION_HPP
#define JLM_IR_AGGREGATION_ANNOTATION_HPP

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace jlm {

class variable;

namespace agg {

class node;

typedef std::unordered_set<std::shared_ptr<const jlm::variable>> demand_set;
typedef std::unordered_map<const agg::node*, demand_set> demand_map;

demand_map
annotate(jlm::agg::node & root);

}}

#endif
