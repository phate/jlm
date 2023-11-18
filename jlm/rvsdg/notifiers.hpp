/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_NOTIFIERS_HPP
#define JLM_RVSDG_NOTIFIERS_HPP

#include <jlm/util/callbacks.hpp>

namespace jlm::rvsdg
{

class input;
class node;
class output;
class region;

extern jlm::util::notifier<jlm::rvsdg::region *> on_region_create;
extern jlm::util::notifier<jlm::rvsdg::region *> on_region_destroy;

extern jlm::util::notifier<jlm::rvsdg::node *> on_node_create;
extern jlm::util::notifier<jlm::rvsdg::node *> on_node_destroy;
extern jlm::util::notifier<jlm::rvsdg::node *, size_t> on_node_depth_change;

extern jlm::util::notifier<jlm::rvsdg::input *> on_input_create;
extern jlm::util::notifier<
    jlm::rvsdg::input *,
    jlm::rvsdg::output *, /* old */
    jlm::rvsdg::output *  /* new */
    >
    on_input_change;
extern jlm::util::notifier<jlm::rvsdg::input *> on_input_destroy;

extern jlm::util::notifier<jlm::rvsdg::output *> on_output_create;
extern jlm::util::notifier<jlm::rvsdg::output *> on_output_destroy;

}

#endif
