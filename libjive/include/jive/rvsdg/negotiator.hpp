/*
 * Copyright 2010 2011 2012 2014 Helge Bahmann <hcb@chaoticmind.net>
 * Copyright 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JIVE_RVSDG_NEGOTIATOR_HPP
#define JIVE_RVSDG_NEGOTIATOR_HPP

#include <unordered_set>

#include <jive/rvsdg/simple-node.hpp>
#include <jive/rvsdg/unary.hpp>
#include <jive/util/callbacks.hpp>
#include <jive/util/intrusive-hash.hpp>

namespace jive {
	class gate;
	class negotiator;
	class output;
}

struct jive_negotiator_connection;
struct jive_negotiator_constraint;

/* options */

class jive_negotiator_option {
public:
	virtual
	~jive_negotiator_option() noexcept;

	inline constexpr jive_negotiator_option() {}

	/* test two options for equality */
	virtual bool
	operator==(const jive_negotiator_option & other) const noexcept = 0;

	inline bool
	operator!=(const jive_negotiator_option & other) noexcept
	{
		return !(*this == other);
	}

	/* specialize option, return true if changed */
	virtual bool
	specialize() noexcept = 0;

	/* try to compute intersection; return true if changed, return false if
	 * it would be empty (and is therefore unchanged) */
	virtual bool
	intersect(const jive_negotiator_option & other) noexcept = 0;

	/* assign new value to option, return true if changed */
	virtual bool
	assign(const jive_negotiator_option & other) noexcept = 0;

	/* make copy of current option */
	virtual jive_negotiator_option *
	copy() const = 0;
};

/* split node */

namespace jive {

class negotiator_split_operation final : public jive::unary_op {
public:
	inline
	negotiator_split_operation(
		jive::negotiator * negotiator,
		const jive::type & input_type,
		const jive_negotiator_option & input_option,
		const jive::type & output_type,
		const jive_negotiator_option & output_option)
	: unary_op(input_type, output_type)
	, negotiator_(negotiator)
	, input_option_(input_option.copy())
	, output_option_(output_option.copy())
	{}

	inline
	negotiator_split_operation(const negotiator_split_operation & other)
	: unary_op(other)
	, negotiator_(other.negotiator())
	, input_option_(other.input_option().copy())
	, output_option_(other.output_option().copy())
	{}

	virtual bool
	operator==(const operation& other) const noexcept override;

	virtual std::string
	debug_string() const override;

	virtual jive_unop_reduction_path_t
	can_reduce_operand(
		const jive::output * arg) const noexcept override;

	virtual jive::output *
	reduce_operand(
		jive_unop_reduction_path_t path,
		jive::output * arg) const override;

	inline jive::negotiator *
	negotiator() const noexcept
	{
		return negotiator_;
	}

	inline const jive::type &
	input_type() const noexcept
	{
		return argument(0).type();
	}

	inline const jive_negotiator_option &
	input_option() const noexcept { return *input_option_; }

	inline const jive::type &
	output_type() const noexcept
	{
		return result(0).type();
	}

	inline const jive_negotiator_option &
	output_option() const noexcept { return *output_option_; }

	virtual std::unique_ptr<jive::operation>
	copy() const override;

private:
	jive::negotiator * negotiator_;
	std::unique_ptr<jive_negotiator_option> input_option_;
	std::unique_ptr<jive_negotiator_option> output_option_;
};

}

/* ports */

typedef enum {
	jive_negotiator_port_attach_none = 0,
	jive_negotiator_port_attach_input = 1,
	jive_negotiator_port_attach_output = 2
} jive_negotiator_port_attach;

struct jive_negotiator_port {
	jive_negotiator_constraint * constraint;
	jive_negotiator_connection * connection;

	jive_negotiator_option * option;

	jive_negotiator_port_attach attach;

	bool specialized;

private:
	jive::detail::intrusive_list_anchor<
		jive_negotiator_port
	> constraint_port_anchor;

	jive::detail::intrusive_list_anchor<
		jive_negotiator_port
	> connection_port_anchor;

	jive::detail::intrusive_list_anchor<
		jive_negotiator_port
	> specialized_port_anchor;

	jive::detail::intrusive_hash_anchor<jive_negotiator_port> hash_chain_;

public:
	union {
		const jive::simple_input * input;
		const jive::output * output;
	} hash_key_;

	typedef jive::detail::intrusive_list_accessor<
		jive_negotiator_port,
		&jive_negotiator_port::constraint_port_anchor
	> constraint_port_accessor;

	typedef jive::detail::intrusive_list_accessor<
		jive_negotiator_port,
		&jive_negotiator_port::connection_port_anchor
	> connection_port_accessor;

	typedef jive::detail::intrusive_list_accessor<
		jive_negotiator_port,
		&jive_negotiator_port::specialized_port_anchor
	> specialized_port_accessor;

	class input_hash_chain_accessor {
	public:
		inline const jive::simple_input *
		get_key(const jive_negotiator_port * obj) const noexcept
		{
			return obj->hash_key_.input;
		}
		inline jive_negotiator_port *
		get_prev(const jive_negotiator_port * obj) const noexcept
		{
			return obj->hash_chain_.prev;
		}
		inline void
		set_prev(jive_negotiator_port * obj, jive_negotiator_port * prev) const noexcept
		{
			obj->hash_chain_.prev = prev;
		}
		inline jive_negotiator_port *
		get_next(const jive_negotiator_port * obj) const noexcept
		{
			return obj->hash_chain_.next;
		}
		inline void
		set_next(jive_negotiator_port * obj, jive_negotiator_port * next) const noexcept
		{
			obj->hash_chain_.next = next;
		}
	};
	class output_hash_chain_accessor {
	public:
		inline const jive::output *
		get_key(const jive_negotiator_port * obj) const noexcept
		{
			return obj->hash_key_.output;
		}
		inline jive_negotiator_port *
		get_prev(const jive_negotiator_port * obj) const noexcept
		{
			return obj->hash_chain_.prev;
		}
		inline void
		set_prev(jive_negotiator_port * obj, jive_negotiator_port * prev) const noexcept
		{
			obj->hash_chain_.prev = prev;
		}
		inline jive_negotiator_port *
		get_next(const jive_negotiator_port * obj) const noexcept
		{
			return obj->hash_chain_.next;
		}
		inline void
		set_next(jive_negotiator_port * obj, jive_negotiator_port * next) const noexcept
		{
			obj->hash_chain_.next = next;
		}
	};
};

typedef jive::detail::intrusive_hash<
	const jive::simple_input *,
	jive_negotiator_port,
	jive_negotiator_port::input_hash_chain_accessor
> jive_negotiator_input_hash;

typedef jive::detail::intrusive_hash<
	const jive::output *,
	jive_negotiator_port,
	jive_negotiator_port::output_hash_chain_accessor
> jive_negotiator_output_hash;


/* connections */

typedef jive::detail::intrusive_list<
	jive_negotiator_port,
	jive_negotiator_port::connection_port_accessor
> connection_port_list;

struct jive_negotiator_connection {
	jive::negotiator * negotiator;

	connection_port_list ports;

	jive::detail::intrusive_list_anchor<
		jive_negotiator_connection
	> connection_list_anchor;

	typedef jive::detail::intrusive_list_accessor<
		jive_negotiator_connection,
		&jive_negotiator_connection::connection_list_anchor
	> connection_list_accessor;

	bool validated;
};

jive_negotiator_connection *
jive_negotiator_connection_create(jive::negotiator * negotiator);

void
jive_negotiator_connection_invalidate(jive_negotiator_connection * self);

/* constraints */

typedef jive::detail::intrusive_list<
	jive_negotiator_port,
	jive_negotiator_port::constraint_port_accessor
> constraint_port_list;

struct jive_negotiator_constraint {
	jive::negotiator * negotiator;
	constraint_port_list ports;

	struct {
		jive_negotiator_constraint * prev;
		jive_negotiator_constraint * next;
	} negotiator_constraint_list;
	
private:
	jive::detail::intrusive_hash_anchor<jive_negotiator_constraint> hash_chain_;
	jive::detail::intrusive_list_anchor<jive_negotiator_constraint> constraint_list_anchor_;

public:
	typedef jive::detail::intrusive_list_accessor<
		jive_negotiator_constraint,
		&jive_negotiator_constraint::constraint_list_anchor_
	> constraint_list_accessor;

	union {
		const jive::node * node;
		const jive::gate * gate;
	} hash_key_;

	class gate_hash_chain_accessor {
	public:
		inline const jive::gate *
		get_key(const jive_negotiator_constraint * obj) const noexcept
		{
			return obj->hash_key_.gate;
		}
		inline jive_negotiator_constraint *
		get_prev(const jive_negotiator_constraint * obj) const noexcept
		{
			return obj->hash_chain_.prev;
		}
		inline void
		set_prev(jive_negotiator_constraint * obj, jive_negotiator_constraint * prev) const noexcept
		{
			obj->hash_chain_.prev = prev;
		}
		inline jive_negotiator_constraint *
		get_next(const jive_negotiator_constraint * obj) const noexcept
		{
			return obj->hash_chain_.next;
		}
		inline void
		set_next(jive_negotiator_constraint * obj, jive_negotiator_constraint * next) const noexcept
		{
			obj->hash_chain_.next = next;
		}
	};

	class node_hash_chain_accessor {
	public:
		inline const jive::node *
		get_key(const jive_negotiator_constraint * obj) const noexcept
		{
			return obj->hash_key_.node;
		}
		inline jive_negotiator_constraint *
		get_prev(const jive_negotiator_constraint * obj) const noexcept
		{
			return obj->hash_chain_.prev;
		}
		inline void
		set_prev(jive_negotiator_constraint * obj, jive_negotiator_constraint * prev) const noexcept
		{
			obj->hash_chain_.prev = prev;
		}
		inline jive_negotiator_constraint *
		get_next(const jive_negotiator_constraint * obj) const noexcept
		{
			return obj->hash_chain_.next;
		}
		inline void
		set_next(jive_negotiator_constraint * obj, jive_negotiator_constraint * next) const noexcept
		{
			obj->hash_chain_.next = next;
		}
	};

	void
	revalidate(jive_negotiator_port * port);
};

typedef jive::detail::intrusive_hash<
	const jive::gate *,
	jive_negotiator_constraint,
	jive_negotiator_constraint::gate_hash_chain_accessor
> jive_negotiator_gate_hash;

typedef jive::detail::intrusive_hash<
	const jive::node *,
	jive_negotiator_constraint,
	jive_negotiator_constraint::node_hash_chain_accessor
> jive_negotiator_node_hash;

jive_negotiator_constraint *
jive_negotiator_identity_constraint_create(jive::negotiator * self);

typedef jive::detail::intrusive_list<
	jive_negotiator_port,
	jive_negotiator_port::specialized_port_accessor
> specialized_port_list;

typedef jive::detail::intrusive_list<
	jive_negotiator_port,
	jive_negotiator_port::specialized_port_accessor
> unspecialized_port_list;

typedef jive::detail::intrusive_list<
	jive_negotiator_connection,
	jive_negotiator_connection::connection_list_accessor
> validated_connection_list;

typedef jive::detail::intrusive_list<
	jive_negotiator_connection,
	jive_negotiator_connection::connection_list_accessor
> invalidated_connection_list;

typedef jive::detail::intrusive_list<
	jive_negotiator_constraint,
	jive_negotiator_constraint::constraint_list_accessor
> constraint_list;

namespace jive {

class negotiator {
public:
	virtual
	~negotiator();

	negotiator(jive::graph * graph);

	negotiator(const negotiator &) = delete;

	negotiator(negotiator &&) = delete;

	negotiator &
	operator=(const negotiator &) = delete;

	negotiator &
	operator=(negotiator &&) = delete;

	/* create empty option (probably invalid) */
	virtual jive_negotiator_option *
	create_option() const;

	/* store suitable default options for this type/resource class pair */
	virtual bool
	store_default_option(jive_negotiator_option * dst, const jive::gate * gate) const;

	/* annotate non-gate ports of node */
	virtual void
	annotate_node_proper(jive::node * node) = 0;

	/* annotate ports of node */
	virtual void
	annotate_node(jive::node * node);

	virtual void
	process_region(jive::region * region);

	jive::graph * graph;
	jive_negotiator_input_hash input_map;
	jive_negotiator_output_hash output_map;
	jive_negotiator_gate_hash gate_map;
	jive_negotiator_node_hash node_map;

	constraint_list constraints;

	validated_connection_list validated_connections;
	invalidated_connection_list invalidated_connections;

	std::unordered_set<jive::node *> split_nodes;
	
	jive::callback node_create_callback;
	jive::callback node_destroy_callback;

	specialized_port_list specialized_ports;
	unspecialized_port_list unspecialized_ports;

	jive_negotiator_option * tmp_option;
};

}

jive_negotiator_port *
jive_negotiator_port_create(
	jive_negotiator_constraint * constraint,
	jive_negotiator_connection * connection,
	const jive_negotiator_option * option);

void
jive_negotiator_port_divert(
	jive_negotiator_port * self,
	jive_negotiator_connection * new_connection);

void
jive_negotiator_port_split(jive_negotiator_port * self);

/* inheritable initializer for constraint */
void
jive_negotiator_constraint_init_(
	jive_negotiator_constraint * self,
	jive::negotiator * negotiator);

/* inheritable finalizer for constraint */
void
jive_negotiator_constraint_fini_(jive_negotiator_constraint * self);

/* inheritable default node annotator */
void
jive_negotiator_annotate_node_(jive::negotiator * self, jive::node * node);

/* inheritable default proper node annotator */
void
jive_negotiator_annotate_node_proper_(jive::negotiator * self, jive::node * node);

/* inheritable default gate annotator */
bool
jive_negotiator_option_gate_default_(const jive::negotiator * self, jive_negotiator_option * dst,
	const jive::gate * gate);

void
jive_negotiator_process_region_(jive::negotiator * self, jive::region * region);

void
jive_negotiator_process(jive::negotiator * self);

void
jive_negotiator_insert_split_nodes(jive::negotiator * self);

void
jive_negotiator_remove_split_nodes(jive::negotiator * self);

jive_negotiator_constraint *
jive_negotiator_annotate_identity(jive::negotiator * self,
	size_t ninputs, jive::simple_input * const inputs[],
	size_t noutputs, jive::simple_output * const outputs[],
	const jive_negotiator_option * option);

jive_negotiator_constraint *
jive_negotiator_annotate_identity_node(
	jive::negotiator * self,
	jive::node * node,
	const jive_negotiator_option * option);

void
jive_negotiator_fully_specialize(jive::negotiator * self);

const jive_negotiator_port *
jive_negotiator_map_output(const jive::negotiator * self, jive::simple_output * output);

jive_negotiator_port *
jive_negotiator_map_output(jive::negotiator * self, jive::simple_output * output);

const jive_negotiator_port *
jive_negotiator_map_input(const jive::negotiator * self, jive::simple_input * input);

jive_negotiator_port *
jive_negotiator_map_input(jive::negotiator * self, jive::simple_input * input);

/* protected functions that allow to manipulate negotiator state */
void
jive_negotiator_port_destroy(jive_negotiator_port * self);

jive_negotiator_port *
jive_negotiator_annotate_simple_input(jive::negotiator * self, jive::simple_input * input,
	const jive_negotiator_option * option);

jive_negotiator_port *
jive_negotiator_annotate_simple_output(jive::negotiator * self, jive::simple_output * output,
	const jive_negotiator_option * option);

#endif
