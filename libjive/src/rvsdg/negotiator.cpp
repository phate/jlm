/*
 * Copyright 2010 2011 2012 2014 2015 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/common.hpp>
#include <jive/rvsdg/graph.hpp>
#include <jive/rvsdg/negotiator.hpp>
#include <jive/rvsdg/notifiers.hpp>
#include <jive/rvsdg/node.hpp>
#include <jive/rvsdg/region.hpp>
#include <jive/rvsdg/simple-node.hpp>
#include <jive/rvsdg/traverser.hpp>
#include <jive/rvsdg/type.hpp>

jive_negotiator_option::~jive_negotiator_option() noexcept
{
}

/* required forward decls */

void
jive_negotiator_connection_destroy(jive_negotiator_connection * self);

/* operation */

namespace jive {

bool
negotiator_split_operation::operator==(const operation & other) const noexcept
{
	auto op = dynamic_cast<const negotiator_split_operation*>(&other);
	return op
	    && op->input_option() == input_option()
	    && op->output_option() == output_option()
	    && op->negotiator() == negotiator()
	    && op->argument(0) == argument(0)
	    && op->result(0) == result(0);
}

std::string
negotiator_split_operation::debug_string() const
{
	return "NEGOTIATOR_SPLIT";
}

jive_unop_reduction_path_t
negotiator_split_operation::can_reduce_operand(
	const jive::output * arg) const noexcept
{
	return jive_unop_reduction_none;
}

jive::output *
negotiator_split_operation::reduce_operand(
	jive_unop_reduction_path_t path,
	jive::output * arg) const
{
	return nullptr;
}

std::unique_ptr<jive::operation>
negotiator_split_operation::copy() const
{
	return std::unique_ptr<jive::operation>(new negotiator_split_operation(*this));
}

}

static jive::simple_output *
jive_negotiator_split(jive::negotiator * negotiator, const jive::type * operand_type,
	jive::simple_output * operand, const jive_negotiator_option * input_option,
	const jive::type * output_type, const jive_negotiator_option * output_option)
{
	jive::negotiator_split_operation op(
		negotiator,
		*operand_type, *input_option,
		*output_type, *output_option);

	// Directly create node without going through normalization -- at this
	// point, normalization *must* not interfere in any way.
	auto node = jive::simple_node::create(operand->node()->region(), op, {operand});
	node->graph()->mark_denormalized();

	jive_negotiator_annotate_simple_input(negotiator, dynamic_cast<jive::simple_input*>(
		node->input(0)), input_option);
	jive_negotiator_annotate_simple_output(negotiator, dynamic_cast<jive::simple_output*>(
		node->output(0)), output_option);

	return dynamic_cast<jive::simple_output*>(node->output(0));
}

/* constraints */

static inline void
jive_negotiator_constraint_revalidate(
	jive_negotiator_constraint * self,
	jive_negotiator_port * port)
{
	self->revalidate(port);
}

jive_negotiator_port *
jive_negotiator_port_create(
	jive_negotiator_constraint * constraint,
	jive_negotiator_connection * connection,
	const jive_negotiator_option * option)
{
	auto negotiator = constraint->negotiator;
	jive_negotiator_port * self = new jive_negotiator_port;
	
	self->constraint = constraint;
	constraint->ports.push_back(self);

	self->connection = connection;
	connection->ports.push_back(self);

	self->specialized = false;
	negotiator->unspecialized_ports.push_back(self);

	self->option = option->copy();
	
	self->attach = jive_negotiator_port_attach_none;
	
	jive_negotiator_connection_invalidate(connection);
	
	return self;
}

void
jive_negotiator_port_divert(
	jive_negotiator_port * self,
	jive_negotiator_connection * new_connection)
{
	jive_negotiator_connection * old_connection = self->connection;
	old_connection->ports.erase(self);
	new_connection->ports.push_back(self);
	self->connection = new_connection;
	
	/* don't need to keep around connections with no port attached any longer */
	if (old_connection->ports.empty())
		jive_negotiator_connection_destroy(old_connection);
	
	jive_negotiator_connection_invalidate(new_connection);
}

void
jive_negotiator_port_split(jive_negotiator_port * self)
{
	jive_negotiator_connection * old_connection = self->connection;
	auto negotiator = old_connection->negotiator;
	jive_negotiator_connection * new_connection = jive_negotiator_connection_create(negotiator);
	
	old_connection->ports.erase(self);
	new_connection->ports.push_back(self);
	self->connection = new_connection;
	
	if (old_connection->ports.empty())
		jive_negotiator_connection_destroy(old_connection);
}

void
jive_negotiator_port_destroy(jive_negotiator_port * self)
{
	self->constraint->ports.erase(self);
	self->connection->ports.erase(self);

	auto negotiator = self->constraint->negotiator;
	
	switch (self->attach) {
		case jive_negotiator_port_attach_none:
		case jive_negotiator_port_attach_input:
			negotiator->input_map.erase(self);
			break;
		case jive_negotiator_port_attach_output:
			negotiator->output_map.erase(self);
			break;
	}
	
	delete self->option;
	delete self;
}

void
jive_negotiator_port_specialize(jive_negotiator_port * self)
{
	auto negotiator = self->constraint->negotiator;
	
	JIVE_DEBUG_ASSERT(!self->specialized);
	negotiator->unspecialized_ports.erase(self);
	negotiator->specialized_ports.push_back(self);
	self->specialized = true;
	
	if (!self->option->specialize())
		return;
	jive_negotiator_connection_invalidate(self->connection);
	jive_negotiator_constraint_revalidate(self->constraint, self);
}

jive_negotiator_connection *
jive_negotiator_connection_create(jive::negotiator * negotiator)
{
	jive_negotiator_connection * self = new jive_negotiator_connection;
	self->negotiator = negotiator;
	self->validated = true;
	negotiator->validated_connections.push_back(self);
	return self;
}

void
jive_negotiator_connection_destroy(jive_negotiator_connection * self)
{
	JIVE_DEBUG_ASSERT(self->ports.empty());
	if (self->validated) {
		self->negotiator->validated_connections.erase(self);
	} else {
		self->negotiator->invalidated_connections.erase(self);
	}
	delete self;
}

void
jive_negotiator_connection_revalidate(jive_negotiator_connection * self)
{
	auto negotiator = self->negotiator;
	
	/* compute common intersection of options */
	jive_negotiator_option * option = 0;
	for (auto & port : self->ports) {
		if (option) {
			option->intersect(*port.option);
		} else {
			option = negotiator->create_option();
			option->assign(*port.option);
		}
	}

	/* apply new constraint to all ports, determine those that are
	incompatible with changed option */
	std::vector<jive_negotiator_port*> unsatisfied;
	for (auto it = self->ports.begin(); it != self->ports.end();) {
		if (*it->option == *option) {
			it++; continue;
		}

		if (it->option->intersect(*option)) {
			jive_negotiator_constraint_revalidate(it->constraint, it.ptr());
			it++; continue;
		}

		unsatisfied.push_back(it.ptr());
		it = self->ports.erase(it);
	}
	
	/* if any ports with non-matchable options remain, split them off
	and group them in a new connection */
	if (unsatisfied.empty())
		return;
	
	auto new_connection = jive_negotiator_connection_create(negotiator);
	for (const auto & port : unsatisfied) {
		new_connection->ports.push_back(port);
		port->connection = new_connection;
		jive_negotiator_connection_invalidate(new_connection);
	}
}

void
jive_negotiator_connection_invalidate(jive_negotiator_connection * self)
{
	if (!self->validated)
		return;
	self->validated = false;
	self->negotiator->validated_connections.erase(self);
	self->negotiator->invalidated_connections.push_back(self);
}

static void
jive_negotiator_connection_merge(
	jive_negotiator_connection * self,
	jive_negotiator_connection * other)
{
	if (self == other)
		return;

	while (auto port = other->ports.first()) {
		other->ports.erase(port);
		self->ports.push_back(port);
	}

	jive_negotiator_connection_destroy(other);
	jive_negotiator_connection_invalidate(self);
}

/* constraint methods */

void
jive_negotiator_constraint_init_(
	jive_negotiator_constraint * self,
	jive::negotiator * negotiator)
{
	self->negotiator = negotiator;
	negotiator->constraints.push_back(self);
}

void
jive_negotiator_constraint_fini_(jive_negotiator_constraint * self)
{
	self->negotiator->constraints.erase(self);
}

void
jive_negotiator_constraint_destroy(jive_negotiator_constraint * self)
{
	JIVE_DEBUG_ASSERT(self->ports.empty());
	delete self;
}

void
jive_negotiator_constraint::revalidate(jive_negotiator_port * port)
{
	for (const auto & tmp : ports) {
		if (&tmp == port)
			continue;
		if (tmp.option->assign(*port->option))
			jive_negotiator_connection_invalidate(tmp.connection);
	}
}

jive_negotiator_constraint *
jive_negotiator_identity_constraint_create(jive::negotiator * self)
{
	jive_negotiator_constraint * constraint = new jive_negotiator_constraint;
	jive_negotiator_constraint_init_(constraint, self);
	return constraint;
}

/* glue code */

static void
jive_negotiator_on_node_create_(
	void * closure,
	jive::node * node)
{
	auto self = (jive::negotiator *) closure;
	if (dynamic_cast<const jive::negotiator_split_operation *>(&node->operation())) {
		self->split_nodes.insert(node);
	}
}

static void
jive_negotiator_on_node_destroy_(
	void * closure,
	jive::node * node)
{
	auto self = (jive::negotiator *) closure;
	self->split_nodes.erase(node);
}

#if 0
static jive_negotiator_constraint *
jive_negotiator_map_gate(jive::negotiator * self, jive::gate * gate)
{
	auto i = self->gate_map.find(gate);
	return i != self->gate_map.end() ? i.ptr() : nullptr;
}

static jive_negotiator_constraint *
jive_negotiator_annotate_gate(jive::negotiator * self, jive::gate * gate)
{
	auto constraint = jive_negotiator_map_gate(self, gate);
	if (!constraint) {
		constraint = jive_negotiator_identity_constraint_create(self);
		constraint->hash_key_.gate = gate;
		self->gate_map.insert(constraint);
	}
	return constraint;
}
#endif
static jive_negotiator_connection *
jive_negotiator_create_input_connection(jive::negotiator * self, jive::input * input)
{
	auto output_port = jive_negotiator_map_output(self,
		dynamic_cast<jive::simple_output*>(input->origin()));
	jive_negotiator_connection * connection;
	if (!output_port)
		connection = jive_negotiator_connection_create(self);
	else
		connection = output_port->connection;
	return connection;
}

static jive_negotiator_connection *
jive_negotiator_create_output_connection(jive::negotiator * self, jive::output * output)
{
	jive_negotiator_connection * connection = 0;
	for (const auto & user : *output) {
		auto port = jive_negotiator_map_input(self, dynamic_cast<jive::simple_input*>(user));
		if (connection && port)
			jive_negotiator_connection_merge(connection, port->connection);
		else if (port)
			connection = port->connection;
	}
	if (!connection)
		connection = jive_negotiator_connection_create(self);
	return connection;
}

void
jive_negotiator_negotiate(jive::negotiator * self)
{
	while(auto connection = self->invalidated_connections.first()) {
		connection->validated = true;
		self->invalidated_connections.erase(connection);
		self->validated_connections.push_back(connection);
		jive_negotiator_connection_revalidate(connection);
	}
}

/* negotiator high-level interface */

namespace jive {

negotiator::negotiator(jive::graph * g)
: graph(g)
{
	tmp_option = create_option();

	node_create_callback = jive::on_node_create.connect(
		std::bind(jive_negotiator_on_node_create_, this, std::placeholders::_1));
	node_destroy_callback = jive::on_node_destroy.connect(
		std::bind(jive_negotiator_on_node_destroy_, this, std::placeholders::_1));
}

negotiator::~negotiator()
{
	delete tmp_option;

	while (auto constraint = constraints.first()) {
		constraints.erase(constraint);

		while (constraint->ports.first())
			jive_negotiator_port_destroy(constraint->ports.first());
		
		jive_negotiator_constraint_destroy(constraint);
	}
	
	while(!validated_connections.empty())
		jive_negotiator_connection_destroy(validated_connections.first());

	while(!invalidated_connections.empty())
		jive_negotiator_connection_destroy(invalidated_connections.first());
}

jive_negotiator_option *
negotiator::create_option() const
{
	/*
		FIXME: This function should actually be abstract, but since we are currently calling it
		in the negotiator constructor I cannot make it abstract.
	*/
	return nullptr;
}

void
negotiator::annotate_node(jive::node * node)
{
	JIVE_ASSERT(0);
#if 0
	for(size_t n = 0; n < node->ninputs(); n++) {
		auto input = dynamic_cast<jive::simple_input*>(node->input(n));
		if (!input->port().gate())
			continue;

		if (!store_default_option(tmp_option, input->port().gate()))
			continue;

		auto constraint = jive_negotiator_annotate_gate(this, input->port().gate());
		auto connection = jive_negotiator_create_input_connection(this, input);
		auto port = jive_negotiator_port_create(constraint, connection, tmp_option);
		port->hash_key_.input = input;
		input_map.insert(port);
	}

	for(size_t n = 0; n < node->noutputs(); n++) {
		auto output = dynamic_cast<jive::simple_output*>(node->output(n));
		if (!output->port().gate())
			continue;

		if (!store_default_option(tmp_option, output->port().gate()))
			continue;

		auto constraint = jive_negotiator_annotate_gate(this, output->port().gate());
		auto connection = jive_negotiator_create_output_connection(this, output);
		auto port = jive_negotiator_port_create(constraint, connection, tmp_option);
		port->hash_key_.output = output;
		output_map.insert(port);
	}

	annotate_node_proper(node);
#endif
}

bool
negotiator::store_default_option(
	jive_negotiator_option * dst,
	const jive::gate * gate) const
{
	return false;
}

void
negotiator::process_region(jive::region * region)
{
	for (auto & node : region->nodes)
		annotate_node(&node);

	jive_negotiator_negotiate(this);
}

}	//jive namespace

void
jive_negotiator_fully_specialize(jive::negotiator * self)
{
	while (auto port = self->unspecialized_ports.first()) {
		jive_negotiator_port_specialize(port);
		jive_negotiator_negotiate(self);
	}
}

const jive_negotiator_port *
jive_negotiator_map_input(const jive::negotiator * self, jive::simple_input * input)
{
	auto i = self->input_map.find(input);
	if (i != self->input_map.end())
		return i.ptr();
	else
		return nullptr;
}

jive_negotiator_port *
jive_negotiator_map_input(jive::negotiator * self, jive::simple_input * input)
{
	auto i = self->input_map.find(input);
	if (i != self->input_map.end())
		return i.ptr();
	else
		return nullptr;
}

const jive_negotiator_port *
jive_negotiator_map_output(const jive::negotiator * self, jive::simple_output * output)
{
	auto i = self->output_map.find(output);
	if (i != self->output_map.end())
		return i.ptr();
	else
		return nullptr;
}

jive_negotiator_port *
jive_negotiator_map_output(jive::negotiator * self, jive::simple_output * output)
{
	auto i = self->output_map.find(output);
	if (i != self->output_map.end())
		return i.ptr();
	else
		return nullptr;
}
	
jive_negotiator_constraint *
jive_negotiator_annotate_identity(jive::negotiator * self,
	size_t ninputs, jive::simple_input * const inputs[],
	size_t noutputs, jive::simple_output * const outputs[],
	const jive_negotiator_option * option)
{
	jive_negotiator_constraint * constraint = jive_negotiator_identity_constraint_create(self);
	size_t n;
	for(n = 0; n < ninputs; n++) {
		auto input = inputs[n];
		jive_negotiator_connection * connection = jive_negotiator_create_input_connection(self, input);
		jive_negotiator_port * port = jive_negotiator_port_create(constraint, connection, option);
		port->hash_key_.input = input;
		self->input_map.insert(port);
		port->attach = jive_negotiator_port_attach_input;
	}
	
	for(n = 0; n < noutputs; n++) {
		jive::simple_output * output = outputs[n];
		auto connection = jive_negotiator_create_output_connection(self, output);
		jive_negotiator_port * port = jive_negotiator_port_create(constraint, connection, option);
		port->hash_key_.output = output;
		self->output_map.insert(port);
		port->attach = jive_negotiator_port_attach_output;
	}
	return constraint;
}

jive_negotiator_constraint *
jive_negotiator_annotate_identity_node(
	jive::negotiator * self,
	jive::node * node,
	const jive_negotiator_option * option)
{
	JIVE_ASSERT(0);
#if 0
	/* FIXME: this assumes that all "gates" are at the end of the list
	-- while plausible, this is not strictly correct */
	std::vector<jive::simple_input*> inputs;
	for (size_t n = 0; n < node->ninputs(); n++) {
		auto input = dynamic_cast<jive::simple_input*>(node->input(n));
		if (!input->port().gate()) {
			inputs.push_back(input);
		}
	}

	std::vector<jive::simple_output*> outputs;
	for (size_t n = 0; n < node->noutputs(); n++) {
		auto output = dynamic_cast<jive::simple_output*>(node->output(0));
		if (!output->port().gate()) {
			outputs.push_back(output);
		}
	}

	jive_negotiator_constraint * constraint;
	constraint = jive_negotiator_annotate_identity(
		self, inputs.size(), &inputs[0], outputs.size(), &outputs[0], option);
	constraint->hash_key_.node = node;
	self->node_map.insert(constraint);
	
	return constraint;
#endif
}

jive_negotiator_port *
jive_negotiator_annotate_simple_input(jive::negotiator * self, jive::simple_input * input,
	const jive_negotiator_option * option)
{
	jive_negotiator_constraint * constraint = jive_negotiator_identity_constraint_create(self);
	jive_negotiator_connection * connection = jive_negotiator_create_input_connection(self, input);
	jive_negotiator_port * port = jive_negotiator_port_create(constraint, connection, option);
	port->hash_key_.input = input;
	self->input_map.insert(port);
	port->attach = jive_negotiator_port_attach_input;
	
	return port;
}

jive_negotiator_port *
jive_negotiator_annotate_simple_output(jive::negotiator * self, jive::simple_output * output,
	const jive_negotiator_option * option)
{
	jive_negotiator_constraint * constraint = jive_negotiator_identity_constraint_create(self);
	jive_negotiator_connection * connection = jive_negotiator_create_output_connection(self, output);
	jive_negotiator_port * port = jive_negotiator_port_create(constraint, connection, option);
	port->hash_key_.output = output;
	self->output_map.insert(port);
	port->attach = jive_negotiator_port_attach_output;
	
	return port;
}

static void
jive_negotiator_maybe_split_edge(
	jive::negotiator * self,
	jive::simple_output * origin,
	jive::simple_input * input)
{
	jive_negotiator_port * origin_port = jive_negotiator_map_output(self, origin);
	if (!origin_port)
		return;
	
	jive_negotiator_port * input_port = jive_negotiator_map_input(self, input);
	if (!input_port)
		return;
	
	if (*origin_port->option == *input_port->option)
		return;
	
	auto type = &input->type();
	auto split_output = jive_negotiator_split(self,
		type, dynamic_cast<jive::simple_output*>(input->origin()), origin_port->option,
		type, input_port->option);
	
	jive_negotiator_port * split_output_port = jive_negotiator_map_output(self, split_output);
	
	input->divert_to(split_output);
	jive_negotiator_port_divert(input_port, split_output_port->connection);
}

void
jive_negotiator_process(jive::negotiator * self)
{
	/* FIXME: this function is broken */
/*
	jive::region * region = self->graph->root();
	while(region->subregions.first)
		region = region->subregions.first;
	
	while(region) {
		self->class_->process_region(self, region);
		if (region->region_subregions_list.next)
			region = region->region_subregions_list.next;
		else
			region = region->parent();
	}
*/
	jive_negotiator_fully_specialize(self);
	
	jive_negotiator_insert_split_nodes(self);
}

void
jive_negotiator_insert_split_nodes(jive::negotiator * self)
{
	for (jive::node * node : jive::topdown_traverser(self->graph->root())) {
		for (size_t n = 0; n < node->ninputs(); n++) {
			auto input = dynamic_cast<jive::simple_input*>(node->input(n));
			jive_negotiator_maybe_split_edge(self,
				dynamic_cast<jive::simple_output*>(input->origin()), input);
		}
	}
}

void
jive_negotiator_remove_split_nodes(jive::negotiator * self)
{
	auto i = self->split_nodes.begin();
	while (i != self->split_nodes.end()) {
		jive::node * node = *i;
		++i;
		jive_negotiator_port * input_port = jive_negotiator_map_input(self,
			dynamic_cast<jive::simple_input*>(node->input(0)));
		jive_negotiator_port * output_port = jive_negotiator_map_output(self,
			dynamic_cast<jive::simple_output*>(node->output(0)));
		jive_negotiator_port_destroy(input_port);
		jive_negotiator_port_destroy(output_port);
		node->output(0)->divert_users(node->input(0)->origin());
		node->region()->remove_node(node);
	}
}
