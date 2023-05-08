/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/notifiers.hpp>

namespace jive {

jlm::util::notifier<jive::region*> on_region_create;
jlm::util::notifier<jive::region*> on_region_destroy;

jlm::util::notifier<jive::node*> on_node_create;
jlm::util::notifier<jive::node*> on_node_destroy;
jlm::util::notifier<jive::node*, size_t> on_node_depth_change;

jlm::util::notifier<jive::input*> on_input_create;
jlm::util::notifier<jive::input*,
	jive::output*,	/* old */
	jive::output*		/* new */
> on_input_change;
jlm::util::notifier<jive::input*> on_input_destroy;

jlm::util::notifier<jive::output*> on_output_create;
jlm::util::notifier<jive::output*> on_output_destroy;

}
