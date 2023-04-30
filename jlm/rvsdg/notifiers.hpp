/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_RVSDG_NOTIFIERS_HPP
#define JLM_RVSDG_NOTIFIERS_HPP

#include <jlm/util/callbacks.hpp>

namespace jive {

class input;
class node;
class output;
class region;

extern notifier<jive::region*> on_region_create;
extern notifier<jive::region*> on_region_destroy;

extern notifier<jive::node*> on_node_create;
extern notifier<jive::node*> on_node_destroy;
extern notifier<jive::node*, size_t> on_node_depth_change;

extern notifier<jive::input*> on_input_create;
extern notifier<jive::input*,
	jive::output*,	/* old */
	jive::output*		/* new */
> on_input_change;
extern notifier<jive::input*> on_input_destroy;

extern notifier<jive::output*> on_output_create;
extern notifier<jive::output*> on_output_destroy;

}

#endif
