/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/basic_block.hpp>
#include <jlm/ir/cfg.hpp>
#include <jlm/ir/module.hpp>
#include <jlm/ir/operators/operators.hpp>
#include <jlm/ir/tac.hpp>
#include <jlm/ir/view.hpp>

#include <deque>

namespace jlm {

/* string converters */

/* FIXME: replace with traverser */
static inline std::vector<jlm::cfg_node*>
breadth_first_traversal(const jlm::cfg & cfg)
{
	std::deque<jlm::cfg_node*> next({cfg.entry_node()});
	std::vector<jlm::cfg_node*> nodes({cfg.entry_node()});
	std::unordered_set<jlm::cfg_node*> visited({cfg.entry_node()});
	while (!next.empty()) {
		auto node = next.front();
		next.pop_front();

		for (auto it = node->begin_outedges(); it != node->end_outedges(); it++) {
			if (visited.find(it->sink()) == visited.end()) {
				visited.insert(it->sink());
				next.push_back(it->sink());
				nodes.push_back(it->sink());
			}
		}
	}

	return nodes;
}

static std::string
emit_tac(const jlm::tac &);

static std::string
emit_tacs(const tacsvector_t & tacs)
{
	std::string str;
	for (const auto & tac : tacs)
		str += emit_tac(*tac) + ", ";

	return "[" + str + "]";
}

static inline std::string
emit_global(const jlm::gblvalue * v)
{
	std::string str = v->debug_string();
	if (!v->initialization().empty())
		str += " = " + emit_tacs(v->initialization());

	return str;
}

static inline std::string
emit_entry(const jlm::cfg_node * node)
{
	JLM_DEBUG_ASSERT(is_entry(node->attribute()));
	auto & entry = *static_cast<const jlm::entry*>(&node->attribute());

	std::string str;
	for (size_t n = 0; n < entry.narguments(); n++)
		str += entry.argument(n)->debug_string() + " ";

	return str + "\n";
}

static inline std::string
emit_exit(const jlm::cfg_node * node)
{
	JLM_DEBUG_ASSERT(is_exit(node->attribute()));
	auto & exit = *static_cast<const jlm::exit*>(&node->attribute());

	std::string str;
	for (size_t n = 0; n < exit.nresults(); n++)
		str += exit.result(n)->debug_string() + " ";

	return str;
}

static inline std::string
emit_tac(const jlm::tac & tac)
{
	/* convert results */
	std::string results;
	for (size_t n = 0; n < tac.noutputs(); n++) {
		results += tac.output(n)->debug_string();
		if (n != tac.noutputs()-1)
			results += ", ";
	}

	/* convert operands */
	std::string operands;
	for (size_t n = 0; n < tac.ninputs(); n++) {
		operands += tac.input(n)->debug_string();
		if (n != tac.ninputs()-1)
			operands += ", ";
	}

	std::string op = tac.operation().debug_string();
	return results + (results.empty() ? "" : " = ") + op + " " + operands;
}

static inline std::string
emit_label(const jlm::cfg_node * node)
{
	return strfmt(node);
}

static inline std::string
emit_targets(const jlm::cfg_node * node)
{
	size_t n = 0;
	std::string str("[");
	for (auto it = node->begin_outedges(); it != node->end_outedges(); it++, n++) {
		str += emit_label(it->sink());
		if (n != node->noutedges()-1)
			str += ", ";
	}
	str += "]";

	return str;
}

static inline std::string
emit_basic_block(const jlm::cfg_node * node)
{
	JLM_DEBUG_ASSERT(is_basic_block(node->attribute()));
	auto & bb = *static_cast<const jlm::basic_block*>(&node->attribute());

	std::string str;
	for (const auto & tac : bb) {
		str += "\t" + emit_tac(*tac);
		if (tac != bb.last())
			str += "\n";
	}

	if (bb.last()) {
		if (is_branch_op(bb.last()->operation()))
			str += " " + emit_targets(node);
		else
			str += "\n\t" + emit_targets(node);
	}	else {
		str += "\t" + emit_targets(node);
	}

	return str + "\n";
}

std::string
to_str(const jlm::cfg & cfg)
{
	static
	std::unordered_map<std::type_index, std::string(*)(const cfg_node*)> map({
	  {std::type_index(typeid(entry)), emit_entry}
	, {std::type_index(typeid(exit)), emit_exit}
	, {std::type_index(typeid(basic_block)), emit_basic_block}
	});

	std::string str;
	auto nodes = breadth_first_traversal(cfg);
	for (const auto & node : nodes) {
		str += emit_label(node) + ":";
		str += (is_basic_block(node->attribute()) ? "\n" : " ");

		JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(node->attribute()))) != map.end());
		str += map[std::type_index(typeid(node->attribute()))](node) + "\n";
	}

	return str;
}

static inline std::string
emit_clg_node(const jlm::callgraph_node & node)
{
	const auto & fcttype = node.fcttype();

	/* convert result types */
	std::string results("<");
	for(size_t n = 0; n < fcttype.nresults(); n++) {
		results += fcttype.result_type(n).debug_string();
		if (n != fcttype.nresults()-1)
			results += ", ";
	}
	results += ">";

	/* convert operand types */
	std::string operands("<");
	for (size_t n = 0; n < fcttype.narguments(); n++) {
		operands += fcttype.argument_type(n).debug_string();
		if (n != fcttype.narguments()-1)
			operands += ", ";
	}
	operands += ">";

	std::string cfg = node.cfg() ? to_str(*node.cfg()) : "";
	std::string exported = !node.exported() ? "static" : "";

	return exported + results + " " + node.name() + " " + operands + "\n{\n" + cfg + "}\n";
}

std::string
to_str(const jlm::callgraph & clg)
{
	std::string str;
	for (const auto & node : clg)
		str += emit_clg_node(node) + "\n";

	return str;
}

std::string
to_str(const jlm::module & module)
{
	std::string str;
	for (const auto & gv : module)
		str += emit_global(gv) + "\n\n";

	str += to_str(module.callgraph());

	return str;
}

/* dot converters */

static inline std::string
emit_entry(const jlm::attribute & attribute)
{
	JLM_DEBUG_ASSERT(is_entry(attribute));
	auto & entry = *static_cast<const jlm::entry*>(&attribute);

	std::string str;
	for (size_t n = 0; n < entry.narguments(); n++) {
		auto argument = entry.argument(n);
		str += "<" + argument->type().debug_string() + "> " + argument->name() + "\\n";
	}

	return str;
}

static inline std::string
emit_exit(const jlm::attribute & attribute)
{
	JLM_DEBUG_ASSERT(is_exit(attribute));
	auto & exit = *static_cast<const jlm::exit*>(&attribute);

	std::string str;
	for (size_t n = 0; n < exit.nresults(); n++) {
		auto result = exit.result(n);
		str += "<" + result->type().debug_string() + "> " + result->name() + "\\n";
	}

	return str;
}

static inline std::string
to_dot(const jlm::tac & tac)
{
	std::string str;
	if (tac.noutputs() != 0) {
		for (size_t n = 0; n < tac.noutputs()-1; n++)
			str +=  tac.output(n)->debug_string() + ", ";
		str += tac.output(tac.noutputs()-1)->debug_string() + " = ";
	}

	str += tac.operation().debug_string();

	if (tac.ninputs() != 0) {
		str += " ";
		for (size_t n = 0; n < tac.ninputs()-1; n++)
			str += tac.input(n)->debug_string() + ", ";
		str += tac.input(tac.ninputs()-1)->debug_string();
	}

	return str;
}

static inline std::string
emit_basic_block(const jlm::attribute & attribute)
{
	JLM_DEBUG_ASSERT(is_basic_block(attribute));
	auto & bb = *static_cast<const jlm::basic_block*>(&attribute);

	std::string str;
	for (const auto & tac : bb)
		str += emit_tac(*tac) + "\\n";

	return str;
}

static inline std::string
emit_header(const jlm::cfg_node & node)
{
	if (is_entry(node.attribute()))
		return "ENTRY";

	if (is_exit(node.attribute()))
		return "EXIT";

	return strfmt(&node);
}

static inline std::string
emit_node(const jlm::cfg_node & node)
{
	static
	std::unordered_map<std::type_index, std::string(*)(const jlm::attribute &)> map({
	  {std::type_index(typeid(jlm::entry)), emit_entry}
	, {std::type_index(typeid(jlm::exit)), emit_exit}
	, {std::type_index(typeid(jlm::basic_block)), emit_basic_block}
	});

	JLM_DEBUG_ASSERT(map.find(std::type_index(typeid(node.attribute()))) != map.end());
	std::string body = map[std::type_index(typeid(node.attribute()))](node.attribute());

	return emit_header(node) + "\\n" + body;
}

std::string
to_dot(const jlm::cfg & cfg)
{
	std::string dot("digraph cfg {\n");
	for (const auto & node : cfg) {
		dot += "{ ";
		if (&node == cfg.entry_node()) dot += "rank = source; ";
		if (&node == cfg.exit_node()) dot += "rank = sink; ";

		dot += strfmt((intptr_t)&node);
		dot += strfmt("[shape = box, label = \"", emit_node(node), "\"]; }\n");
		for (auto it = node.begin_outedges(); it != node.end_outedges(); it++) {
			dot += strfmt((intptr_t)it->source(), " -> ", (intptr_t)it->sink());
			dot += strfmt("[label = \"", it->index(), "\"];\n");
		}
	}
	dot += "}\n";

	return dot;
}

std::string
to_dot(const jlm::callgraph & clg)
{
	std::string dot("digraph clg {\n");
	for (const auto & node : clg) {
		dot += strfmt((intptr_t)&node);
		dot += strfmt("[label = \"", node.name(), "\"];\n");

		for (const auto & call : node)
			dot += strfmt((intptr_t)&node, " -> ", (intptr_t)call, ";\n");
	}
	dot += "}\n";

	return dot;
}

}
