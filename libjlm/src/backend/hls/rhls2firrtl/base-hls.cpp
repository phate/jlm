/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/backend/hls/rhls2firrtl/base-hls.hpp>
#include <jlm/ir/hls/hls.hpp>

#include <algorithm>

bool
jlm::hls::isForbiddenChar(char c) {
	if (('A' <= c && c <= 'Z') ||
		('a' <= c && c <= 'z') ||
		('0' <= c && c <= '9') ||
		'_' == c) {
		return false;
	}
	return true;
}

std::string
jlm::hls::BaseHLS::get_node_name(const jive::node *node) {
	auto found = node_map.find(node);
	if (found != node_map.end()) {
		return found->second;
	}
	std::string append = "";
	auto inPorts = node->ninputs();
	auto outPorts = node->noutputs();
	if (inPorts) {
		append.append("_IN");
		append.append(std::to_string(inPorts));
		append.append("_W");
		append.append(std::to_string(JlmSize(&node->input(inPorts-1)->type())));
	}
	if (outPorts) {
		append.append("_OUT");
		append.append(std::to_string(outPorts));
		append.append("_W");
		append.append(std::to_string(JlmSize(&node->output(outPorts-1)->type())));
	}
	auto name = jive::detail::strfmt("op_", node->operation().debug_string(), append, "_", node_map.size());
	// remove chars that are not valid in firrtl module names
	std::replace_if(name.begin(), name.end(), isForbiddenChar, '_');
	node_map[node] = name;
	return name;
}

std::string
jlm::hls::BaseHLS::get_port_name(jive::input *port) {
	std::string result;
	if (dynamic_cast<const jive::node_input *>(port)) {
		result += "i";
	} else if (dynamic_cast<const jive::result *>(port)) {
		result += "r";
	} else {
		throw std::logic_error(port->debug_string() + " not implemented!");
	}
	result += jive::detail::strfmt(port->index());
	return result;
}

std::string
jlm::hls::BaseHLS::get_port_name(jive::output *port) {
	if (port == nullptr) {
		throw std::logic_error("nullptr!");
	}
	std::string result;
	if (dynamic_cast<const jive::argument *>(port)) {
		result += "a";
	} else if (dynamic_cast<const jive::node_output *>(port)) {
		result += "o";
	} else if (dynamic_cast<const jive::structural_output *>(port)) {
		result += "so";
	} else {
		throw std::logic_error(port->debug_string() + " not implemented!");
	}
	result += jive::detail::strfmt(port->index());
	return result;
}

int
jlm::hls::BaseHLS::JlmSize(const jive::type *type) {
	if (auto bt = dynamic_cast<const jive::bittype *>(type)) {
		return bt->nbits();
	} else if (auto at = dynamic_cast<const jlm::arraytype *>(type)) {
		return JlmSize(&at->element_type()) * at->nelements();
	} else if (dynamic_cast<const jlm::PointerType *>(type)) {
		return 64;
	} else if (auto ct = dynamic_cast<const jive::ctltype *>(type)) {
		return ceil(log2(ct->nalternatives()));
	} else if (dynamic_cast<const jive::statetype *>(type)) {
		return 1;
	} else {
		throw std::logic_error("Size of '" + type->debug_string() + "' is not implemented!");
	}
}

void
jlm::hls::BaseHLS::create_node_names(jive::region *r) {
	for (auto &node : r->nodes) {
		if (dynamic_cast<jive::simple_node *>(&node)) {
			get_node_name(&node);
		} else if (auto oln = dynamic_cast<jlm::hls::loop_node *>(&node)) {
			create_node_names(oln->subregion());
		} else {
			throw jlm::error("Unimplemented op (unexpected structural node) : " + node.operation().debug_string());
		}
	}
}

const jlm::lambda::node *
jlm::hls::BaseHLS::get_hls_lambda(jlm::RvsdgModule &rm) {
	auto region = rm.Rvsdg().root();
	auto ln = dynamic_cast<const jlm::lambda::node *>(region->nodes.begin().ptr());
	if (region->nnodes() == 1 && ln) {
		return ln;
	} else {
		throw jlm::error("Root should have only one lambda node now");
	}
}

std::string
jlm::hls::BaseHLS::get_base_file_name(const jlm::RvsdgModule &rm) {
	auto source_file_name = rm.SourceFileName().name();
	auto base_file_name = source_file_name.substr(0, source_file_name.find_last_of('.'));
	return base_file_name;
}
