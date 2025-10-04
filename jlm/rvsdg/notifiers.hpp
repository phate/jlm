/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_NOTIFIERS_HPP
#define JLM_RVSDG_NOTIFIERS_HPP

#include <jlm/util/callbacks.hpp>

namespace jlm::rvsdg
{

class Input;
class Node;
class Output;
class Region;

extern util::Notifier<Node *> on_node_create;
extern util::Notifier<Node *> on_node_destroy;

extern jlm::util::Notifier<
    jlm::rvsdg::Input *,
    jlm::rvsdg::Output *, /* old */
    jlm::rvsdg::Output *  /* new */
    >
    on_input_change;

}

#endif
