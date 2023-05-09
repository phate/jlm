/*
 * Copyright 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/notifiers.hpp>
#include <jlm/rvsdg/structural-node.hpp>
#include <jlm/rvsdg/substitution.hpp>

namespace jlm::rvsdg
{

/* structural input */

structural_input::~structural_input() noexcept
{
	JLM_ASSERT(arguments.empty());

	on_input_destroy(this);
}

structural_input::structural_input(
	jlm::rvsdg::structural_node * node,
	jlm::rvsdg::output * origin,
	const jlm::rvsdg::port & port)
: node_input(origin, node, port)
{
	on_input_create(this);
}

/* structural output */

structural_output::~structural_output() noexcept
{
	JLM_ASSERT(results.empty());

	on_output_destroy(this);
}

structural_output::structural_output(
	jlm::rvsdg::structural_node * node,
	const jlm::rvsdg::port & port)
: node_output(node, port)
{
	on_output_create(this);
}

/* structural node */

structural_node::~structural_node()
{
	on_node_destroy(this);

	subregions_.clear();
}

structural_node::structural_node(
	const jlm::rvsdg::structural_op & op,
	jlm::rvsdg::region * region,
	size_t nsubregions)
: node(op.copy(), region)
{
	if (nsubregions == 0)
		throw jlm::util::error("Number of subregions must be greater than zero.");

	for (size_t n = 0; n < nsubregions; n++)
		subregions_.emplace_back(std::unique_ptr<jlm::rvsdg::region>(new jlm::rvsdg::region(this, n)));

	on_node_create(this);
}

structural_input *
structural_node::append_input(std::unique_ptr<structural_input> input)
{
	if (input->node() != this)
		throw jlm::util::error("Appending input to wrong node.");

	auto index = input->index();
	JLM_ASSERT(index == 0);
	if (index != 0
	|| (index == 0 && ninputs() > 0 && this->input(0) == input.get()))
		return this->input(index);

	auto sinput = std::unique_ptr<node_input>(input.release());
	return static_cast<structural_input*>(node::add_input(std::move(sinput)));
}

structural_output *
structural_node::append_output(std::unique_ptr<structural_output> output)
{
	if (output->node() != this)
		throw jlm::util::error("Appending output to wrong node.");

	auto index = output->index();
	JLM_ASSERT(index == 0);
	if (index != 0
	|| (index == 0 && noutputs() > 0 && this->output(0) == output.get()))
		return this->output(index);

	auto soutput = std::unique_ptr<node_output>(output.release());
	return static_cast<structural_output*>(node::add_output(std::move(soutput)));
}

}
