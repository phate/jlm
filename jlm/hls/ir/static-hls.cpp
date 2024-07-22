/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/ir/static-hls.hpp>

namespace jlm::static_hls
{

jlm::rvsdg::node *
mux_add_input(jlm::rvsdg::node * old_mux, jlm::rvsdg::output * new_input, bool predicate)
{
  JLM_ASSERT(jlm::rvsdg::is<jlm::static_hls::mux_op>(old_mux->operation()));

  // If the input already exists, return the old mux
  // FIXME and add to the fsm
  for (size_t i = 0; i < old_mux->ninputs(); i++)
  {
    if (old_mux->input(i)->origin() == new_input)
    {
      return old_mux;
    }
  }

  std::vector<jlm::rvsdg::output *> new_mux_inputs;
  for (size_t i = 0; i < old_mux->ninputs(); i++)
  {
    new_mux_inputs.push_back(old_mux->input(i)->origin());
  }

  jlm::rvsdg::simple_node * new_mux;
  if (!predicate)
  {
    new_mux_inputs.push_back(new_input);

    new_mux = jlm::static_hls::mux_op::create(new_mux_inputs);
  }
  else
  {
    new_mux = jlm::static_hls::mux_op::create(*new_input, new_mux_inputs);
  }

  old_mux->output(0)->divert_users(new_mux->output(0));

  remove(old_mux);

  // old_mux = static_cast<jlm::rvsdg::node*>(new_mux);
  return new_mux;
};

} // namespace jlm::static_hls
