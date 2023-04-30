/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/notifiers.hpp>

namespace jive {

notifier<jive::region*> on_region_create;
notifier<jive::region*> on_region_destroy;

notifier<jive::node*> on_node_create;
notifier<jive::node*> on_node_destroy;
notifier<jive::node*, size_t> on_node_depth_change;

notifier<jive::input*> on_input_create;
notifier<jive::input*,
	jive::output*,	/* old */
	jive::output*		/* new */
> on_input_change;
notifier<jive::input*> on_input_destroy;

notifier<jive::output*> on_output_create;
notifier<jive::output*> on_output_destroy;

}
