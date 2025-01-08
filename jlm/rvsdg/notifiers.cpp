/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/notifiers.hpp>

namespace jlm::rvsdg
{

jlm::util::notifier<rvsdg::Region *> on_region_create;
jlm::util::notifier<rvsdg::Region *> on_region_destroy;

util::notifier<rvsdg::Node *> on_node_create;
util::notifier<rvsdg::Node *> on_node_destroy;
util::notifier<rvsdg::Node *, size_t> on_node_depth_change;

jlm::util::notifier<jlm::rvsdg::input *> on_input_create;
jlm::util::notifier<
    jlm::rvsdg::input *,
    jlm::rvsdg::output *, /* old */
    jlm::rvsdg::output *  /* new */
    >
    on_input_change;
jlm::util::notifier<jlm::rvsdg::input *> on_input_destroy;

jlm::util::notifier<jlm::rvsdg::output *> on_output_create;
jlm::util::notifier<jlm::rvsdg::output *> on_output_destroy;

}
