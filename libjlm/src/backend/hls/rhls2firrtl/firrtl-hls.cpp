/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include "jlm/backend/hls/rhls2firrtl/firrtl-hls.hpp"

bool
jlm::hls::is_identity_mapping(const jive::match_op &op) {
	for (const auto &pair : op) {
		if (pair.first != pair.second)
			return false;
	}

	return true;
}

std::string
jlm::hls::FirrtlHLS::get_text(jlm::RvsdgModule &rm) {
	std::ostringstream firrtl;
	auto module = lambda_node_to_firrtl(get_hls_lambda(rm));
	firrtl << indent(0) << "circuit " << module.name << ":\n";
	for (auto module: modules) {
		firrtl << module.second.firrtl;
	}
	return firrtl.str();
}

std::string
jlm::hls::FirrtlHLS::to_firrtl_type(const jive::type *type) {
	return jive::detail::strfmt("UInt<", jlm_sizeof(type), ">");
}

std::string
jlm::hls::FirrtlHLS::mem_io() {
	std::ostringstream module;
	module << indent(2)
		   << "output mem_req: {flip ready: UInt<1>, valid: UInt<1>, addr: UInt<64>, data: UInt<64>, write: UInt<1>, width: UInt<3>}\n";
	module << indent(2) << "input mem_res: {valid: UInt<1>, data: UInt<64>}\n";
	module << indent(2) << "mem_req.valid <= " << UInt(1, 0) << "\n";
	module << indent(2) << "mem_req.addr is invalid\n";
	module << indent(2) << "mem_req.write is invalid\n";
	module << indent(2) << "mem_req.data is invalid\n";
	module << indent(2) << "mem_req.width is invalid\n";
	return module.str();
}

std::string
jlm::hls::FirrtlHLS::mux_mem(const std::vector<std::string> &mem_nodes) const {
	std::ostringstream mem;
	std::string previous_granted = UInt(1, 0);
	for (auto node_name:mem_nodes) {
		mem << indent(2) << node_name << ".mem_res.valid <= mem_res.valid\n";
		mem << indent(2) << node_name << ".mem_res.data <= mem_res.data\n";
		mem << indent(2) << node_name << ".mem_req.ready <= " << UInt(1, 0) << "\n";
		mem << indent(2) << "when and(not(" << previous_granted << ")," << node_name
			<< ".mem_req.valid):\n";
		mem << indent(3) << node_name << ".mem_req.ready <= " << UInt(1, 1) << "\n";
		mem << indent(3) << "mem_req.addr <= " << node_name << ".mem_req.addr\n";
		mem << indent(3) << "mem_req.write <= " << node_name << ".mem_req.write\n";
		mem << indent(3) << "mem_req.valid <= " << UInt(1, 1) << "\n";
		mem << indent(3) << "mem_req.data <= " << node_name << ".mem_req.data\n";
		mem << indent(3) << "mem_req.width <= " << node_name << ".mem_req.width\n";
		mem << indent(2) << "node previous_granted_" << node_name << " = or(" << previous_granted << ", "
			<< node_name << ".mem_req.ready)\n";
		previous_granted = "previous_granted_" + node_name;
	}
	return mem.str();
}

std::string
jlm::hls::FirrtlHLS::module_header(const jive::node *node, bool has_mem_io) {
	std::ostringstream module;

	module << indent(1) << "module " << get_module_name(node) << ":\n";
	// io
	module << indent(2) << "; io\n";
	module << indent(2) << "input clk: Clock\n";
	module << indent(2) << "input reset: UInt<1>\n";
	for (size_t i = 0; i < node->ninputs(); ++i) {
		module << indent(2) << "input i" << (i) << ": {flip ready: UInt<1>, valid: UInt<1>, data: " <<
			   to_firrtl_type(&node->input(i)->type()) << "}\n";
	}
	for (size_t i = 0; i < node->noutputs(); ++i) {
		module << indent(2) << "output o" << i <<
			   ": {flip ready: UInt<1>, valid: UInt<1>, data: " <<
			   to_firrtl_type(&node->output(i)->type()) << "}\n";
	}
	if (has_mem_io) {
		module << mem_io();
	}

	return module.str();
}

jlm::hls::FirrtlModule &
jlm::hls::FirrtlHLS::mem_node_to_firrtl(const jive::simple_node *n) {
	std::string module_name = get_module_name(n);
	std::ostringstream module;
	module << module_header(n, true);

	bool store = dynamic_cast<const jlm::StoreOperation *>(&(n->operation()));
	// registers
	module << indent(2) << "; registers\n";
	for (size_t i = 0; i < n->noutputs(); ++i) {
		module << indent(2) << "reg o" << i << "_valid_reg: UInt<1>, clk with: (reset => (reset, "
			   << UInt(1, 0)
			   << "))\n";
		module << indent(2) << "reg o" << i << "_data_reg: " << to_firrtl_type(&n->output(i)->type())
			   << ", clk\n";
	}
	module << indent(2) << "reg sent_reg: UInt<1>, clk with: (reset => (reset, " << UInt(1, 0) << "))\n";
	std::string can_request = "not(sent_reg)"; // only request again once previous request is finished
	for (size_t i = 0; i < n->ninputs(); ++i) {
		can_request = "and(" + can_request + ", " + valid(n->input(i)) + ")";
	}
	for (size_t i = 0; i < n->noutputs(); ++i) {
		can_request = "and(" + can_request + ", not(o" + jive::detail::strfmt(i) + "_valid_reg))";
	}
	// block until all inputs and no outputs are valid
	module << indent(2) << "node can_request = " << can_request << "\n";

	module << indent(2) << "; mem request\n";
	module << indent(2) << "mem_req.valid <= can_request\n";
	module << indent(2) << "mem_req.addr <= " << data(n->input(0)) << "\n";
	int bit_width;
	if (store) {
		module << indent(2) << "mem_req.write <= " << UInt(1, 1) << "\n";
		module << indent(2) << "mem_req.data <= " << data(n->input(1)) << "\n";
		bit_width = dynamic_cast<const jive::bittype *>(&n->input(1)->type())->nbits();
	} else {
		module << indent(2) << "mem_req.write <= " << UInt(1, 0) << "\n";
		module << indent(2) << "mem_req.data is invalid\n";
		if (auto bt = dynamic_cast<const jive::bittype *>(&n->output(0)->type())) {
			bit_width = bt->nbits();
		} else if (dynamic_cast<const jlm::PointerType *>(&n->output(0)->type())) {
			bit_width = 64;
		} else {
			throw jlm::error("unknown width for mem request");
		}
	}
	int log2_bytes = log2(bit_width / 8);
	module << indent(2) << "mem_req.width <= " << UInt(4, log2_bytes) << "\n";
	module << indent(2) << "when mem_req.ready:\n";
	module << indent(3) << "sent_reg <= " << UInt(1, 1) << "\n";
	// set memstate
	if (store) {
		module << indent(3) << "o0_valid_reg <= " << UInt(1, 1) << "\n";
		module << indent(3) << "o0_data_reg <= " << data(n->input(2)) << "\n";
	} else {
		module << indent(3) << "o1_valid_reg <= " << UInt(1, 1) << "\n";
		module << indent(3) << "o1_data_reg <= " << data(n->input(1)) << "\n";
	}
	module << indent(2) << "; mem response\n";
	module << indent(2) << "when and(sent_reg, mem_res.valid):\n";
	module << indent(3) << "sent_reg <= " << UInt(1, 0) << "\n";
	if (!store) {
		module << indent(3) << "o0_valid_reg <= " << UInt(1, 1) << "\n";
		module << indent(3) << "o0_data_reg <= mem_res.data\n";
	}
	// handshaking
	module << indent(2) << "; handshaking\n";
	// inputs are ready when mem interface accepts request (is ready)
	for (size_t i = 0; i < n->ninputs(); ++i) {
		module << indent(2) << ready(n->input(i)) << " <= mem_req.ready\n";
	}
	for (size_t i = 0; i < n->noutputs(); ++i) {
		auto out = n->output(i);
		module << indent(2) << valid(out) << " <= o" << i << "_valid_reg,\n";
		module << indent(2) << data(out) << " <= o" << i << "_data_reg\n";
		module << indent(2) << "when " << fire(out) << ":\n";
		module << indent(3) << "o" << i << "_valid_reg <= " << UInt(1, 0) << "\n";
	}

	modules.emplace_back(&n->operation(), FirrtlModule{module_name, module.str(), true});
	return modules.back().second;
}

jlm::hls::FirrtlModule &
jlm::hls::FirrtlHLS::pred_buffer_to_firrtl(const jive::simple_node *n) {
	std::string module_name = get_module_name(n);
	std::ostringstream module;
	module << module_header(n);
	module << indent(2) << "; registers\n";
	// start initialized with a valid pred 0
	module << indent(2) << "reg buf_valid_reg: UInt<1>, clk with: (reset => (reset, " << UInt(1, 1)
		   << "))\n";
	module << indent(2) << "reg buf_data_reg: " << to_firrtl_type(&n->output(0)->type())
		   << ", clk  with: (reset => (reset, " << UInt(1, 0) << "))\n";
	auto o0 = n->output(0);
	auto i0 = n->input(0);
	module << indent(2) << valid(o0) << " <= or(buf_valid_reg, " << valid(i0) << ")\n";
	module << indent(2) << data(o0) << " <= mux(buf_valid_reg, buf_data_reg, " << data(i0) << ")\n";
	module << indent(2) << ready(i0) << " <= not(buf_valid_reg)\n";
	module << indent(2) << "when " << fire(i0) << ":\n";
	module << indent(3) << "buf_valid_reg <= " << UInt(1, 1) << "\n";
	module << indent(3) << "buf_data_reg <= " << data(i0) << "\n";
	module << indent(2) << "when " << fire(o0) << ":\n";
	module << indent(3) << "buf_valid_reg <= " << UInt(1, 0) << "\n";
	modules.emplace_back(&n->operation(), FirrtlModule{module_name, module.str(), false});
	return modules.back().second;
}

jlm::hls::FirrtlModule &
jlm::hls::FirrtlHLS::buffer_to_firrtl(const jive::simple_node *n) {
	std::string module_name = get_module_name(n);
	std::ostringstream module;
	module << module_header(n);
	module << indent(2) << "; registers\n";
	auto o = dynamic_cast<const hls::buffer_op *>(&(n->operation()));
	auto capacity = o->capacity;
	for (size_t i = 0; i < capacity; ++i) {
		module << indent(2) << "reg buf" << i
			   << "_valid_reg: UInt<1>, clk with: (reset => (reset, " << UInt(1, 0) << "))\n";
		module << indent(2) << "reg buf" << i << "_data_reg: " << to_firrtl_type(&n->output(0)->type())
			   << ", clk\n";
	}
	for (size_t i = 0; i <= capacity; ++i) {
		module << indent(2) << "wire in_consumed" << i << ": UInt<1>\n";
		module << indent(2) << "wire shift_out" << i << ": UInt<1>\n";
	}
	auto o0 = n->output(0);
	// connect out to buf0
	module << indent(2) << valid(o0) << " <= buf0_valid_reg\n";
	module << indent(2) << data(o0) << " <= buf0_data_reg\n";
	auto i0 = n->input(0);
	// buf is ready if last one is empty
	module << indent(2) << ready(i0) << " <= not(buf" << capacity - 1 << "_valid_reg)\n";

	// shift register with insertion into earliest free slot
	module << indent(2) << "shift_out0 <= " << fire(o0) << "\n";
	module << indent(2) << "in_consumed0 <= " << UInt(1, 0) << "\n";
	if(o->pass_through){
		module << indent(2) << valid(o0) << " <= or(" << valid(i0) << ", buf0_valid_reg)\n";
		module << indent(2) << "in_consumed0 <= and(not(buf0_valid_reg), "<<ready(o0)<<")\n";
		module << indent(2) << "when not(buf0_valid_reg):\n";
		module << indent(3) << data(o0) << " <= " << data(i0) << "\n";
	}
	// invalid pseudo slot so we can use the same logic for the last slot
	module << indent(2) << "node buf" << capacity << "_valid_reg = " << UInt(1, 0) << "\n";
	module << indent(2) << "node buf" << capacity << "_data_reg = " << UInt(1, 0) << "\n";
	for (size_t i = 0; i < capacity; ++i) {
		module << indent(2) << "node will_be_empty" << i << " = or(shift_out" << i << ", not(buf" << i
			   << "_valid_reg))\n";
		module << indent(2) << "in_consumed" << i + 1 << " <= in_consumed" << i << "\n";
		module << indent(2) << "node in_available" << i << " = and(" << fire(i0) << ", not(in_consumed" << i
			   << "))\n";
		module << indent(2) << "shift_out" << i + 1 << " <= " << UInt(1, 0) << "\n";
		module << indent(2) << "when shift_out" << i << ":\n";
		module << indent(3) << "buf" << i << "_valid_reg <= " << UInt(1, 0) << "\n";
		module << indent(2) << "when will_be_empty" << i << ":\n";
		module << indent(3) << "when buf" << i + 1 << "_valid_reg:\n";
		module << indent(4) << "buf" << i << "_valid_reg <= " << UInt(1, 1) << "\n";
		module << indent(4) << "buf" << i << "_data_reg <= buf" << i + 1 << "_data_reg\n";
		module << indent(4) << "shift_out" << i + 1 << " <= " << UInt(1, 1) << "\n";
		module << indent(3) << "else when in_available" << i << ":\n";
		module << indent(4) << "in_consumed" << i + 1 << " <= " << UInt(1, 1) << "\n";
		module << indent(4) << "buf" << i << "_valid_reg <= " << UInt(1, 1) << "\n";
		module << indent(4) << "buf" << i << "_data_reg <= " << data(i0) << "\n";

	}
	modules.emplace_back(&n->operation(), FirrtlModule{module_name, module.str(), false});
	return modules.back().second;
}

jlm::hls::FirrtlModule &
jlm::hls::FirrtlHLS::ndmux_to_firrtl(const jive::simple_node *n) {
	std::string module_name = get_module_name(n);
	std::ostringstream module;
	module << module_header(n);

	auto ipred = n->input(0);
	auto o0 = n->output(0);

	module << indent(2) << valid(o0) << " <= " << UInt(1, 0) << "\n";
	module << indent(2) << data(o0) << " is invalid\n";
	module << indent(2) << ready(ipred) << " <= " << UInt(1, 0) << "\n";

	for (size_t i = 1; i < n->ninputs(); ++i) {
		auto in = n->input(i);
		module << indent(2) << ready(in) << " <= " << UInt(1, 0) << "\n";
		module << indent(2) << "when and(" << valid(ipred) << ", eq(" << data(ipred) << ", "
			   << UInt(64, i - 1)
			   << ")):\n";
		module << indent(3) << valid(o0) << " <= " << valid(in) << "\n";
		module << indent(3) << data(o0) << " <= " << data(in) << "\n";
		module << indent(3) << ready(in) << " <= " << ready(o0) << "\n";
		module << indent(3) << ready(ipred) << " <= and(" << ready(o0) << "," << valid(in) << ")\n";
	}
	modules.emplace_back(&n->operation(), FirrtlModule{module_name, module.str(), false});
	return modules.back().second;
}

jlm::hls::FirrtlModule &
jlm::hls::FirrtlHLS::dmux_to_firrtl(const jive::simple_node *n) {
	std::string module_name = get_module_name(n);
	std::ostringstream module;
	module << module_header(n);

	auto ipred = n->input(0);
	auto o0 = n->output(0);

	std::string any_discard_reg = UInt(1, 0);
	for (size_t i = 1; i < n->ninputs(); ++i) {
		// discard and discard_reg are separate in order to allow tokens to be discarded in the same cycle.
		module << indent(2) << "reg i" << i << "_discard_reg: UInt<1>, clk with: (reset => (reset, " << UInt(1, 0)
			   << "))\n";
		module << indent(2) << "wire i" << i << "_discard: UInt<1>\n";
		module << indent(2) << "i" << i << "_discard <= i" << i << "_discard_reg\n";
		module << indent(2) << "i" << i << "_discard_reg <= i" << i << "_discard\n";
		any_discard_reg = jive::detail::strfmt("or(", any_discard_reg, ", i", i, "_discard_reg)");
	}
	module << indent(2) << "node any_discard_reg = " << any_discard_reg << "\n";
	module << indent(2) << "reg processed_reg: UInt<1>, clk with: (reset => (reset, " << UInt(1, 0) << "))\n";
	module << indent(2) << valid(o0) << " <= " << UInt(1, 0) << "\n";
	module << indent(2) << data(o0) << " is invalid\n";
	module << indent(2) << ready(ipred) << " <= " << UInt(1, 0) << "\n";


	for (size_t i = 1; i < n->ninputs(); ++i) {
		auto in = n->input(i);
		module << indent(2) << ready(in) << " <= i" << i << "_discard\n";
		// clear discard reg on fire
		module << indent(2) << "when " << fire(in) << ":\n";
		module << indent(3) << "i" << i << "_discard_reg <= " << UInt(1, 0) << "\n";
		// pred match and no outstanding discards
		module << indent(2) << "when and(and(" << valid(ipred) << ", eq(" << data(ipred)
			   << ", "
			   << UInt(64, i - 1)
			   << ")), not(any_discard_reg)):\n";
		module << indent(3) << valid(o0) << " <= " << valid(in) << "\n";
		module << indent(3) << data(o0) << " <= " << data(in) << "\n";
		module << indent(3) << ready(in) << " <= " << ready(o0) << "\n";
		module << indent(3) << ready(ipred) << " <= and(" << ready(o0) << "," << valid(in) << ")\n";
		module << indent(3) << "when not(processed_reg):\n";
		for (size_t j = 1; j < n->ninputs(); ++j) {
			if (i != j) {
				module << indent(4) << "i" << j << "_discard <= " << UInt(1, 1) << "\n";
			}
		}
		module << indent(4) << "processed_reg <= " << UInt(1, 1) << "\n";
	}
	module << indent(2) << "when " << fire(o0) << ":\n";
	module << indent(3) << "processed_reg <= " << UInt(1, 0) << "\n";
	modules.emplace_back(&n->operation(), FirrtlModule{module_name, module.str(), false});
	return modules.back().second;
}

jlm::hls::FirrtlModule &
jlm::hls::FirrtlHLS::merge_to_firrtl(const jive::simple_node *n) {
	std::string module_name = get_module_name(n);
	std::ostringstream module;
	module << module_header(n);
	auto o0 = n->output(0);

	module << indent(2) << valid(o0) << " <= " << UInt(1, 0) << "\n";
	module << indent(2) << data(o0) << " is invalid\n";


	for (size_t i = 0; i < n->ninputs(); ++i) {
		auto in = n->input(i);
		module << indent(2) << ready(in) << " <= " << ready(o0) << "\n";
		module << indent(2) << "when " << valid(in) << ":\n";
		module << indent(3) << valid(o0) << " <= " << valid(in) << "\n";
		module << indent(3) << data(o0) << " <= " << data(in) << "\n";
	}
	modules.emplace_back(&n->operation(), FirrtlModule{module_name, module.str(), false});
	return modules.back().second;
}

jlm::hls::FirrtlModule &
jlm::hls::FirrtlHLS::fork_to_firrtl(const jive::simple_node *n) {
	std::string module_name = get_module_name(n);
	std::ostringstream module;
	module << module_header(n);

	module << indent(2) << "; registers\n";
	for (size_t i = 0; i < n->noutputs(); ++i) {
		module << indent(2) << "reg out" << i
			   << "_fired: UInt<1>, clk with: (reset => (reset, " << UInt(1, 0) << "))\n";
	}

	auto i0 = n->input(0);
	std::string all_fired = UInt(1, 1); // True by default
	for (size_t i = 0; i < n->noutputs(); ++i) {
		auto out = n->output(i);
		module << indent(2) << valid(out) << " <= and(" << valid(i0) << ", not(out" << i << "_fired))\n";
		module << indent(2) << data(out) << " <= " << data(i0) << "\n";
		all_fired =
				"and(" + all_fired + ", or(" + ready(out) + ", out" + jive::detail::strfmt(i) + "_fired))";
	}
	module << indent(2) << "node all_fired = " << all_fired << "\n";
	module << indent(2) << ready(i0) << " <= all_fired\n";
	module << indent(2) << "when not(all_fired):\n";
	for (size_t i = 0; i < n->noutputs(); ++i) {
		module << indent(3) << "when " << fire(n->output(i)) << ":\n";
		module << indent(4) << "out" << i << "_fired <= " << UInt(1, 1) << "\n";
	}
	module << indent(2) << "else:\n";
	for (size_t i = 0; i < n->noutputs(); ++i) {
		module << indent(3) << "out" << i << "_fired <= " << UInt(1, 0) << "\n";
	}
	modules.emplace_back(&n->operation(), FirrtlModule{module_name, module.str(), false});
	return modules.back().second;
}

jlm::hls::FirrtlModule &
jlm::hls::FirrtlHLS::sink_to_firrtl(const jive::simple_node *n) {
	std::string module_name = get_module_name(n);
	std::ostringstream module;
	module << module_header(n);
	auto i0 = n->input(0);
	module << indent(2) << ready(i0) << " <= " << UInt(1, 1) << "\n";

	modules.emplace_back(&n->operation(), FirrtlModule{module_name, module.str(), false});
	return modules.back().second;
}


jlm::hls::FirrtlModule &
jlm::hls::FirrtlHLS::print_to_firrtl(const jive::simple_node *n) {
	auto pn = dynamic_cast<const jlm::hls::print_op *>(&n->operation());
	std::string module_name = get_module_name(n);
	std::ostringstream module;
	module << module_header(n);
	auto i0 = n->input(0);
	auto o0 = n->output(0);
	module << indent(2) << " ; " << n->operation().debug_string() << "\n";
	module << indent(2) << data(o0) << " <= " << data(i0);
	// handshaking
	module << indent(2) + "; handshaking" + "\n";

	module << indent(2) << valid(o0) << " <= " << valid(i0) << "\n";
	module << indent(2) << ready(i0) << " <= " << ready(o0) << "\n";
	module << indent(2) << "printf(clk, and(" << fire(i0) << ", not(reset)), \"print node " << pn->id() << ": %x\\n\", pad("
		   << data(i0)
		   << ", 64))\n";


	modules.emplace_back(&n->operation(), FirrtlModule{module_name, module.str(), false});
	return modules.back().second;
}

jlm::hls::FirrtlModule &
jlm::hls::FirrtlHLS::branch_to_firrtl(const jive::simple_node *n) {
	std::string module_name = get_module_name(n);
	std::ostringstream module;
	module << module_header(n);

	auto ipred = n->input(0);
	auto ival = n->input(1);

	module << indent(2) << ready(ival) << " <= " << UInt(1, 0) << "\n";
	module << indent(2) << ready(ipred) << " <= " << UInt(1, 0) << "\n";
	for (size_t i = 0; i < n->noutputs(); ++i) {
		auto out = n->output(i);
		module << indent(2) << valid(out) << " <= " << UInt(1, 0) << "\n";
		module << indent(2) << data(out) << " is invalid\n";
		module << indent(2) << "when and(" << valid(ipred) << ", eq(" << data(ipred) << ", " + UInt(64, i)
			   << ")):\n";
		module << indent(3) << ready(ival) << " <= " << ready(out) << "\n";
		module << indent(3) << ready(ipred) << " <= and(" << ready(out) << "," << valid(ival) << ")\n";
		module << indent(3) << valid(out) << " <= " << valid(ival) << "\n";
		module << indent(3) << data(out) << " <= " << data(ival) << "\n";
	}
	modules.emplace_back(&n->operation(), FirrtlModule{module_name, module.str(), false});
	return modules.back().second;
}

jlm::hls::FirrtlModule &
jlm::hls::FirrtlHLS::trigger_to_firrtl(const jive::simple_node *n) {
	std::string module_name = get_module_name(n);
	std::ostringstream module;
	module << module_header(n);

	auto itrig = n->input(0);
	auto ival = n->input(1);
	auto out = n->output(0);
	// inputs have to both fire at the same time
	module << indent(2) << ready(itrig) << " <= and(" << ready(out) << "," << valid(ival) << ")\n";
	module << indent(2) << ready(ival) << " <= and(" << ready(out) << "," << valid(itrig) << ")\n";
	module << indent(2) << valid(out) << " <= and(" << valid(ival) << "," << valid(itrig) << ")\n";
	module << indent(2) << data(out) << " <= " << data(ival) << "\n";
	modules.emplace_back(&n->operation(), FirrtlModule{module_name, module.str(), false});
	return modules.back().second;
}


int
jlm::hls::jlm_sizeof(const jive::type *t) {
	if (auto bt = dynamic_cast<const jive::bittype *>(t)) {
		return bt->nbits();
	} else if (auto at = dynamic_cast<const jlm::arraytype *>(t)) {
		return jlm_sizeof(&at->element_type()) * at->nelements();
	} else if ( dynamic_cast<const jlm::PointerType *>(t)) {
		return 64;
	} else if (auto ct = dynamic_cast<const jive::ctltype *>(t)) {
		return ceil(log2(ct->nalternatives()));
	} else if (dynamic_cast<const jive::statetype *>(t)) {
		return 1;
	} else {
		throw std::logic_error(t->debug_string() + " size of not implemented!");
	}
}

std::string
jlm::hls::FirrtlHLS::gep_op_to_firrtl(const jive::simple_node *n) {
	auto o = dynamic_cast<const jlm::GetElementPtrOperation *>(&(n->operation()));
	std::string result = "cvt("+data(n->input(0))+")"; // start of with base pointer
	//TODO: support structs
	const jive::type *pt = &o->GetPointeeType();
	for (size_t i = 1; i < n->ninputs(); ++i) {
		int bits = jlm_sizeof(pt);
		if (dynamic_cast<const jive::bittype *>(pt)) { ;
		} else if (auto at = dynamic_cast<const jlm::arraytype *>(pt)) {
			pt = &at->element_type();
		} else {
			throw std::logic_error(pt->debug_string() + " pointer not implemented!");
		}
		int bytes = bits / 8;
		// gep inputs are signed
		auto offset = "mul(asSInt(" + data(n->input(i)) + "), cvt(" + UInt(64, bytes) + "))";
		result = "add(" + result + ", " + offset + ")";
	}
	return "asUInt("+result+")";

}

std::string
jlm::hls::FirrtlHLS::match_op_to_firrtl(const jive::simple_node *n) {
	std::string result;
	auto o = dynamic_cast<const jive::match_op *>(&(n->operation()));
	JLM_ASSERT(o);
	if (is_identity_mapping(*o)) {
		return data(n->input(0));
	} else {
		result = UInt(64, o->default_alternative());
		//TODO: optimize?
		for (auto it = o->begin(); it != o->end(); it++) {
			result = "mux(eq(" + UInt(64, it->first) + ", " + data(n->input(0)) + "), " + UInt(64, it->second) + ", " +
					 result + ")";
		}
	}
	return result;
}

std::string
jlm::hls::FirrtlHLS::simple_op_to_firrtl(const jive::simple_node *n) {
	if (dynamic_cast<const jive::match_op *>(&(n->operation()))) {
		return match_op_to_firrtl(n);
	} else if (dynamic_cast<const jive::bitsgt_op *>(&(n->operation()))) {
		return "gt(asSInt(" + data(n->input(0)) + "), asSInt(" + data(n->input(1)) + "))";
	} else if (dynamic_cast<const jive::bitsge_op *>(&(n->operation()))) {
		return "geq(asSInt(" + data(n->input(0)) + "), asSInt(" + data(n->input(1)) + "))";
	} else if (dynamic_cast<const jive::bitsle_op *>(&(n->operation()))) {
		return "leq(asSInt(" + data(n->input(0)) + "), asSInt(" + data(n->input(1)) + "))";
	} else if (auto o = dynamic_cast<const jlm::sext_op *>(&(n->operation()))) {
		return "asUInt(pad(asSInt(" + data(n->input(0)) + "), " + jive::detail::strfmt(o->ndstbits()) +
			   "))";
	} else if (dynamic_cast<const jlm::trunc_op *>(&(n->operation()))) {
		return data(n->input(0));
	} else if (dynamic_cast<const jlm::bitcast_op *>(&(n->operation()))) {
		return data(n->input(0));
	} else if (dynamic_cast<const jlm::zext_op *>(&(n->operation()))) {
		return data(n->input(0));
	} else if (dynamic_cast<const jive::bitugt_op *>(&(n->operation()))) {
		return "gt(" + data(n->input(0)) + ", " + data(n->input(1)) + ")";
	} else if (dynamic_cast<const jive::bitult_op *>(&(n->operation()))) {
		return "lt(" + data(n->input(0)) + ", " + data(n->input(1)) + ")";
	} else if (dynamic_cast<const jive::bitslt_op *>(&(n->operation()))) {
		return "lt(asSInt(" + data(n->input(0)) + "), asSInt(" + data(n->input(1)) + "))";
	} else if (dynamic_cast<const jive::biteq_op *>(&(n->operation()))) {
		return "eq(" + data(n->input(0)) + ", " + data(n->input(1)) + ")";
	} else if (dynamic_cast<const jive::bitadd_op *>(&(n->operation()))) {
		return "add(" + data(n->input(0)) + ", " + data(n->input(1)) + ")";
	} else if (dynamic_cast<const jive::bitmul_op *>(&(n->operation()))) {
		return "mul(" + data(n->input(0)) + ", " + data(n->input(1)) + ")";
	} else if (dynamic_cast<const jive::bitsub_op *>(&(n->operation()))) {
		return "sub(" + data(n->input(0)) + ", " + data(n->input(1)) + ")";
	} else if (dynamic_cast<const jive::bitand_op *>(&(n->operation()))) {
		return "and(" + data(n->input(0)) + ", " + data(n->input(1)) + ")";
	} else if (dynamic_cast<const jive::bitxor_op *>(&(n->operation()))) {
		return "xor(" + data(n->input(0)) + ", " + data(n->input(1)) + ")";
	} else if (dynamic_cast<const jive::bitor_op *>(&(n->operation()))) {
		return "or(" + data(n->input(0)) + ", " + data(n->input(1)) + ")";
	} else if (dynamic_cast<const jive::bitshr_op *>(&(n->operation()))) {
		//TODO: automatic conversion to static shift?
		return "dshr(" + data(n->input(0)) + ", " + data(n->input(1)) + ")";
	} else if (dynamic_cast<const jive::bitashr_op *>(&(n->operation()))) {
		//TODO: automatic conversion to static shift?
		return "asUInt(dshr(asSInt(" + data(n->input(0)) + "), " + data(n->input(1)) + "))";
	} else if (dynamic_cast<const jive::bitsdiv_op *>(&(n->operation()))) {
		return "asUInt(div(asSInt(" + data(n->input(0)) + "), asSInt(" + data(n->input(1)) + ")))";
	}  else if (dynamic_cast<const jive::bitsmod_op *>(&(n->operation()))) {
		return "asUInt(rem(asSInt(" + data(n->input(0)) + "), asSInt(" + data(n->input(1)) + ")))";
	} else if (dynamic_cast<const jive::bitshl_op *>(&(n->operation()))) {
		//TODO: automatic conversion to static shift?
		// TODO: adjust shift limit (bits)
		return "dshl(" + data(n->input(0)) + ", bits(" + data(n->input(1)) + ", 7, 0))";
	} else if (dynamic_cast<const jive::bitne_op *>(&(n->operation()))) {
		return "neq(" + data(n->input(0)) + ", " + data(n->input(1)) + ")";
	} else if (auto o = dynamic_cast<const jive::bitconstant_op *>(&(n->operation()))) {
		auto value = o->value();
		return jive::detail::strfmt("UInt<", value.nbits(), ">(", value.to_uint(), ")");
	} else if (dynamic_cast<const jlm::UndefValueOperation *>(&(n->operation()))) {
		return UInt(1, 0);  // TODO: Fix?
	} else if (auto o = dynamic_cast<const jive::ctlconstant_op *>(&(n->operation()))) {
		return UInt(ceil(log2(o->value().nalternatives())), o->value().alternative());
	} else if (dynamic_cast<const jlm::GetElementPtrOperation *>(&(n->operation()))) {
		return gep_op_to_firrtl(n);
	} else {
		throw std::logic_error(n->operation().debug_string() + " not implemented!");
	}
}

jlm::hls::FirrtlModule &
jlm::hls::FirrtlHLS::single_out_simple_node_to_firrtl(const jive::simple_node *n) {
	std::string module_name = get_module_name(n);
	std::ostringstream module;

	module << module_header(n);

	if (n->noutputs() != 1) {
		throw std::logic_error(n->operation().debug_string() + " has more than 1 output");
	}
	module << indent(2) << "; logic\n";
	module << indent(2) << " ; " << n->operation().debug_string() << "\n";
	auto op = simple_op_to_firrtl(n);
	module << indent(2) << data(n->output(0)) << " <= " << op;
	// handshaking
	module << indent(2) + "; handshaking" + "\n";

	auto out = n->output(0);

	std::string inputs_valids = UInt(1, 1); // True by default
	for (size_t i = 0; i < n->ninputs(); ++i) {
		inputs_valids = "and(" + inputs_valids + ", " + valid(n->input(i)) + ")";
	}
	module << indent(2) << "node inputs_valid = " << inputs_valids << "\n";
	for (size_t i = 0; i < n->noutputs(); ++i) {
		module << indent(2) << valid(out) << " <= inputs_valid\n";
	}
	for (size_t i = 0; i < n->ninputs(); ++i) {
		module << indent(2) + ready(n->input(i)) + "<= and(" << ready(n->output(0)) << ", inputs_valid)\n";
	}

	modules.emplace_back(&n->operation(), FirrtlModule{module_name, module.str(), false});
	return modules.back().second;
}

jlm::hls::FirrtlModule
jlm::hls::FirrtlHLS::node_to_firrtl(const jive::node *node, const int depth) {
	// check if module for operation was already generated
	for (auto pair: modules) {
		if(pair.first != nullptr && *pair.first == node->operation()) {
			return pair.second;
		}
	}
	if (auto n = dynamic_cast<const jive::simple_node *>(node)) {
		if (dynamic_cast<const jlm::LoadOperation *>(&(n->operation()))) {
			return mem_node_to_firrtl(n);
		} else if (dynamic_cast<const jlm::StoreOperation *>(&(n->operation()))) {
			return mem_node_to_firrtl(n);
		} else if (dynamic_cast<const hls::predicate_buffer_op *>(&(n->operation()))) {
			return pred_buffer_to_firrtl(n);
		} else if (dynamic_cast<const hls::buffer_op *>(&(n->operation()))) {
			return buffer_to_firrtl(n);
		} else if (dynamic_cast<const hls::branch_op *>(&(n->operation()))) {
			return branch_to_firrtl(n);
		} else if (dynamic_cast<const hls::trigger_op *>(&(n->operation()))) {
			return trigger_to_firrtl(n);
		} else if (dynamic_cast<const hls::sink_op *>(&(n->operation()))) {
			return sink_to_firrtl(n);
		} else if (dynamic_cast<const hls::print_op *>(&(n->operation()))) {
			return print_to_firrtl(n);
		} else if (dynamic_cast<const hls::fork_op *>(&(n->operation()))) {
			return fork_to_firrtl(n);
		} else if (dynamic_cast<const hls::merge_op *>(&(n->operation()))) {
			return merge_to_firrtl(n);
		} else if (auto o = dynamic_cast<const hls::mux_op *>(&(n->operation()))) {
			if (o->discarding) {
				return dmux_to_firrtl(n);
			} else {
				return ndmux_to_firrtl(n);
			}
		}
		return single_out_simple_node_to_firrtl(n);
	} else {
		throw std::logic_error(node->operation().debug_string() + " not implemented!");
	}
}

std::string
jlm::hls::FirrtlHLS::create_loop_instances(jlm::hls::loop_node *ln) {
	std::ostringstream firrtl;
	auto sr = ln->subregion();
	for (const auto node : jive::topdown_traverser(sr)) {
		if (dynamic_cast<jive::simple_node *>(node)) {
			auto node_module = node_to_firrtl(node, 2);
			std::string inst_name = get_node_name(node);
			if (node_module.has_mem) {
				mem_nodes.push_back(inst_name);
			}
			firrtl << indent(2) << "inst " << inst_name << " of " << node_module.name << "\n";
			for (size_t i = 0; i < node->noutputs(); ++i) {
				output_map[node->output(i)] = inst_name + "." + get_port_name(node->output(i));
			}
		} else if (auto oln = dynamic_cast<hls::loop_node *>(node)) {
			firrtl << create_loop_instances(oln);
		} else {
			throw jlm::error("Unimplemented op (unexpected structural node) : " + node->operation().debug_string());
		}
	}
	for (size_t i = 0; i < sr->narguments(); ++i) {
		auto arg = sr->argument(i);
        auto ba = dynamic_cast<jlm::hls::backedge_argument*>(arg);
        if (!ba) {
            assert(arg->input() != nullptr);
			// map to input of loop
			output_map[arg] = output_map[arg->input()->origin()];
		} else {
			auto result = ba->result();
			assert(result->type() == arg->type());
			// map to end of loop (origin of associated result)
			output_map[arg] = output_map[result->origin()];
		}
	}
	for (size_t i = 0; i < ln->noutputs(); ++i) {
		auto out = ln->output(i);
		assert(out->results.size() == 1);
		output_map[out] = output_map[out->results.begin()->origin()];
	}
	return firrtl.str();
}

std::string
jlm::hls::FirrtlHLS::connect(jive::region *sr) {
	std::ostringstream firrtl;
	for (const auto &node : jive::topdown_traverser(sr)) {
		if (dynamic_cast<jive::simple_node *>(node)) {
			auto inst_name = get_node_name(node);
			firrtl << indent(2) << inst_name << ".clk <= clk\n";
			firrtl << indent(2) << inst_name << ".reset <= reset\n";
			for (size_t i = 0; i < node->ninputs(); ++i) {
				auto in_name = inst_name + "." + get_port_name(node->input(i));
				assert(output_map.count(node->input(i)->origin()));
				auto origin = output_map[node->input(i)->origin()];
				firrtl << indent(2) << origin << ".ready <= " << in_name << ".ready\n";
				firrtl << indent(2) << in_name << ".data <= " << origin << ".data\n";
				firrtl << indent(2) << in_name << ".valid <= " << origin << ".valid\n";
			}
		} else if (auto oln = dynamic_cast<hls::loop_node *>(node)) {
			firrtl << connect(oln->subregion());
		} else {
			throw jlm::error("Unimplemented op (unexpected structural node) : " + node->operation().debug_string());
		}
	}
	return firrtl.str();
}

jlm::hls::FirrtlModule
jlm::hls::FirrtlHLS::subregion_to_firrtl(jive::region *sr) {
	auto module_name = "subregion_mod" + jive::detail::strfmt(modules.size());
	std::ostringstream module;
	module << indent(1) << "module " << module_name << ":\n";
	// io
	module << indent(2) << "; io\n";
	module << indent(2) << "input clk: Clock\n";
	module << indent(2) << "input reset: UInt<1>\n";
	for (size_t i = 0; i < sr->narguments(); ++i) {
		module << indent(2) << "input " << get_port_name(sr->argument(i)) <<
			   ": {flip ready: UInt<1>, valid: UInt<1>, data: " <<
			   to_firrtl_type(&sr->argument(i)->type()) << "}\n";
	}
	for (size_t i = 0; i < sr->nresults(); ++i) {
		module << indent(2) << "output " << get_port_name(sr->result(i)) <<
			   ": {flip ready: UInt<1>, valid: UInt<1>, data: " <<
			   to_firrtl_type(&sr->result(i)->type()) << "}\n";
	}
	module << mem_io();
	module << indent(2) << "; instances\n";
	for (size_t i = 0; i < sr->narguments(); ++i) {
		output_map[sr->argument(i)] = get_port_name(sr->argument(i));
	}
	// create node modules and ios first
	for (const auto node : jive::topdown_traverser(sr)) {
		if (dynamic_cast<jive::simple_node *>(node)) {
			auto node_module = node_to_firrtl(node, 2);
			std::string inst_name = get_node_name(node);
			if (node_module.has_mem) {
				mem_nodes.push_back(inst_name);
			}
			module << indent(2) << "inst " << inst_name << " of " << node_module.name << "\n";
			for (size_t i = 0; i < node->noutputs(); ++i) {
				output_map[node->output(i)] = inst_name + "." + get_port_name(node->output(i));
			}
		} else if (auto oln = dynamic_cast<hls::loop_node *>(node)) {
			module << create_loop_instances(oln);
		} else {
			throw jlm::error("Unimplemented op (unexpected structural node) : " + node->operation().debug_string());
		}
	}
	module << connect(sr);
	for (size_t i = 0; i < sr->nresults(); ++i) {
		auto origin = output_map[sr->result(i)->origin()];
		auto result = get_port_name(sr->result(i));
		module << indent(2) << origin << ".ready <= " << result << ".ready\n";
		module << indent(2) << result << ".data <= " << origin << ".data\n";
		module << indent(2) << result << ".valid <= " << origin << ".valid\n";
	}

	module << mux_mem(mem_nodes);

	modules.emplace_back(nullptr, FirrtlModule{module_name, module.str(), true});
	return modules.back().second;
}

jlm::hls::FirrtlModule
jlm::hls::FirrtlHLS::lambda_node_to_firrtl(const jlm::lambda::node *ln) {
	auto module_name = ln->name() + "_lambda_mod";
	auto sr = ln->subregion();
	create_node_names(sr);
	std::ostringstream module;
	module << indent(1) << "module " << module_name << ":\n";
	// io
	module << indent(2) << "; io" << "\n";
	module << indent(2) << "input clk: Clock" << "\n";
	module << indent(2) << "input reset: UInt<1>" << "\n";
	module << indent(2) << "input i: {flip ready: UInt<1>, valid: UInt<1>";
	for (size_t i = 0; i < sr->narguments(); ++i) {
		module << ", data" << i << ": " <<
			   to_firrtl_type(&sr->argument(i)->type());
	}
	module << "}\n";
	module << indent(2) << "output o: {flip ready: UInt<1>, valid: UInt<1>";
	for (size_t i = 0; i < sr->nresults(); ++i) {
		module << ", data" << i << ": " <<
			   to_firrtl_type(&sr->result(i)->type());
	}
	module << "}\n";
	module << mem_io();
	// registers
	module << indent(2) << "; registers" << "\n";
	for (size_t i = 0; i < sr->narguments(); ++i) {
		module << indent(2) << "reg i" << i <<
			   "_valid_reg: UInt<1>, clk with: (reset => (reset, " << UInt(1, 0) << "))\n";
		module << indent(2) << "reg i" << i << "_data_reg: " <<
			   to_firrtl_type(&sr->argument(i)->type()) + ", clk" + "\n";
	}
	for (size_t i = 0; i < sr->nresults(); ++i) {
		module << indent(2) << "reg o" << i <<
			   "_valid_reg: UInt<1>, clk with: (reset => (reset, " << UInt(1, 0) << "))\n";
		module << indent(2) << "reg o" << i << "_data_reg: " <<
			   to_firrtl_type(&sr->result(i)->type()) + ", clk" + "\n";
	}
	module << indent(2) << "; instances" << "\n";
	auto sr_firrtl = subregion_to_firrtl(sr);
	module << indent(2) << "inst sr of " << sr_firrtl.name << "\n";
	module << indent(2) << "sr.clk <= clk\n";
	module << indent(2) << "sr.reset <= reset\n";
	for (size_t i = 0; i < sr->narguments(); ++i) {
		module << indent(2) << "sr." << valid(sr->argument(i)) << " <= i" << i << "_valid_reg\n";
		module << indent(2) << "sr." << data(sr->argument(i)) << " <= i" << i << "_data_reg\n";
		module << indent(2) << "when and(sr." << valid(sr->argument(i)) << ", sr." << ready(sr->argument(i))
			   << "):\n";
		module << indent(3) << "i" << i << "_valid_reg <= " << UInt(1, 0) << "\n";
	}
	// logic
	module << indent(2) << "; logic" << "\n";
	for (size_t i = 0; i < sr->nresults(); ++i) {
		module << indent(2) << "sr." << ready(sr->result(i)) << " <= not(o" << i <<
			   "_valid_reg)\n";
	}
	for (size_t i = 0; i < sr->nresults(); ++i) {
		module << indent(2) << "when and(sr." << valid(sr->result(i)) << ", sr." << ready(sr->result(i))
			   << "):\n";
		module << indent(3) << "o" << i << "_valid_reg <= " << UInt(1, 1) << "\n";
		module << indent(3) << "o" << i << "_data_reg <= sr." << get_port_name(sr->result(i)) <<
			   ".data\n";
	}
	std::string outputs_valids = UInt(1, 1); // True by default
	for (size_t i = 0; i < sr->nresults(); ++i) {
		outputs_valids = "and(" + outputs_valids + ", o" + jive::detail::strfmt(i) + "_valid_reg)";
	}
	module << indent(2) << "node outputs_valid = " << outputs_valids << "\n";
	module << indent(2) << "o.valid <= outputs_valid\n";
	for (size_t i = 0; i < sr->nresults(); ++i) {
		module << indent(2) << "o.data" << i << " <= o" << i << "_data_reg\n";
	}
	module << indent(2) << "when and(o.valid, o.ready):\n";
	for (size_t i = 0; i < sr->nresults(); ++i) {
		module << indent(3) << "o" << i << "_valid_reg <= " << UInt(1, 0) << "\n";
	}
	std::string inputs_ready = UInt(1, 1); // True by default
	for (size_t i = 0; i < sr->narguments(); ++i) {
		inputs_ready = jive::detail::strfmt("and(", inputs_ready, ", not(i", i, "_valid_reg))");
	}
	module << indent(2) << "node inputs_ready = " << inputs_ready << "\n";
	module << indent(2) << "i.ready <= inputs_ready\n";


	module << indent(2) << "when and(i.ready, i.valid):\n";
	for (size_t i = 0; i < sr->narguments(); ++i) {
		module << indent(3) << "i" << i << "_valid_reg <= " << UInt(1, 1) << "\n";
		module << indent(3) << "i" << i << "_data_reg <= i.data" << i << "\n";
	}

	module << mux_mem({"sr",});
	modules.emplace_back(&ln->operation(), FirrtlModule{module_name, module.str(), false});
	return modules.back().second;
}

std::string
jlm::hls::FirrtlHLS::get_module_name(const jive::node *node) {
	auto new_name = jive::detail::strfmt("op_", node->operation().debug_string(), "_", modules.size());
	// remove chars that are not valid in firrtl module names
	std::replace_if(new_name.begin(), new_name.end(), isForbiddenChar, '_');
	return new_name;
}
