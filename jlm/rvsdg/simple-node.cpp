/*
 * Copyright 2016 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/notifiers.hpp>
#include <jlm/rvsdg/simple-node.hpp>
#include <jlm/rvsdg/substitution.hpp>

namespace jive {

/* inputs */

simple_input::~simple_input() noexcept
{
	on_input_destroy(this);
}

simple_input::simple_input(
	jive::simple_node * node,
	jive::output * origin,
	const jive::port & port)
: node_input(origin, node, port)
{}

/* outputs */

simple_output::simple_output(
	jive::simple_node * node,
	const jive::port & port)
: node_output(node, port)
{}

simple_output::~simple_output() noexcept
{
	on_output_destroy(this);
}

/* simple nodes */

simple_node::~simple_node()
{
	on_node_destroy(this);
}

simple_node::simple_node(
	jive::region * region,
	const jive::simple_op & op,
	const std::vector<jive::output*> & operands)
	: node(op.copy(), region)
{
	if (operation().narguments() != operands.size())
		throw jive::compiler_error(jive::detail::strfmt("Argument error - expected ",
			operation().narguments(), ", received ", operands.size(), " arguments."));

	for (size_t n = 0; n < operation().narguments(); n++) {
		node::add_input(std::unique_ptr<node_input>(
			new simple_input(this, operands[n], operation().argument(n))));
	}

	for (size_t n = 0; n < operation().nresults(); n++)
		node::add_output(std::unique_ptr<node_output>(
			new simple_output(this, operation().result(n))));

	on_node_create(this);
}

jive::node *
simple_node::copy(jive::region * region, const std::vector<jive::output*> & operands) const
{
	auto node = create(region, *static_cast<const simple_op*>(&operation()), operands);
	graph()->mark_denormalized();
	return node;
}

jive::node *
simple_node::copy(jive::region * region, jive::substitution_map & smap) const
{
	std::vector<jive::output*> operands;
	for (size_t n = 0; n < ninputs(); n++) {
		auto origin = input(n)->origin();
		auto operand = smap.lookup(origin);

		if (operand == nullptr) {
			if (region != this->region())
				throw compiler_error("Node operand not in substitution map.");

			operand = origin;
		}

		operands.push_back(operand);
	}

	auto node = copy(region, operands);

	JIVE_DEBUG_ASSERT(node->noutputs() == noutputs());
	for (size_t n = 0; n < node->noutputs(); n++)
		smap.insert(output(n), node->output(n));

	return node;
}

}
