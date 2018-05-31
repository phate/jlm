/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_AGGREGATION_AGGREGATION_HPP
#define JLM_IR_AGGREGATION_AGGREGATION_HPP

#include <memory>

namespace jlm {

class cfg;

namespace agg {

class aggnode;

std::unique_ptr<agg::aggnode>
aggregate(jlm::cfg & cfg);

}}

#endif
