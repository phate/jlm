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

jlm::util::notifier<jlm::rvsdg::Input *> on_input_create;
jlm::util::notifier<
    jlm::rvsdg::Input *,
    jlm::rvsdg::Output *, /* old */
    jlm::rvsdg::Output *  /* new */
    >
    on_input_change;
jlm::util::notifier<jlm::rvsdg::Input *> on_input_destroy;

jlm::util::notifier<jlm::rvsdg::Output *> on_output_create;
jlm::util::notifier<jlm::rvsdg::Output *> on_output_destroy;

}
