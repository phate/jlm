/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/notifiers.hpp>

namespace jlm::rvsdg
{

util::Notifier<rvsdg::Node *> on_node_create;
util::Notifier<rvsdg::Node *> on_node_destroy;

jlm::util::Notifier<jlm::rvsdg::Input *> on_input_create;
jlm::util::Notifier<
    jlm::rvsdg::Input *,
    jlm::rvsdg::Output *, /* old */
    jlm::rvsdg::Output *  /* new */
    >
    on_input_change;

}
