/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_AGGREGATION_AGGREGATION_HPP
#define JLM_IR_AGGREGATION_AGGREGATION_HPP

#include <memory>

namespace jlm {

class aggnode;
class cfg;

std::unique_ptr<aggnode>
aggregate(jlm::cfg & cfg);

}

#endif
