/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_TAC_ADDRESS_H
#define JLM_IR_TAC_ADDRESS_H

#include <jlm/IR/tac/tac.hpp>

#include <jive/arch/address.h>
#include <jive/arch/addresstype.h>
#include <jive/arch/load.h>
#include <jive/arch/memorytype.h>
#include <jive/arch/store.h>
#include <jive/types/bitstring/type.h>

namespace jlm {
namespace frontend {

JIVE_EXPORTED_INLINE const jlm::frontend::variable *
addrload_tac(
	jlm::frontend::basic_block * basic_block,
	const jlm::frontend::variable * address,
	const jlm::frontend::variable * state,
	const jive::value::type & data_type,
	const jlm::frontend::variable * result)
{
	jive::addr::type addrtype;
	std::vector<std::unique_ptr<jive::state::type>> state_type;
	state_type.emplace_back(std::unique_ptr<jive::state::type>(new jive::mem::type()));
	jive::load_op op(addrtype, state_type, data_type);
	const jlm::frontend::tac * tac = basic_block->append(op, {address, state}, {result});
	return tac->output(0);
}

JIVE_EXPORTED_INLINE const jlm::frontend::variable *
addrstore_tac(
	jlm::frontend::basic_block * basic_block,
	const jlm::frontend::variable * address,
	const jlm::frontend::variable * value,
	const jlm::frontend::variable * state,
	const jlm::frontend::variable * result)
{
	jive::addr::type addrtype;
	std::vector<std::unique_ptr<jive::state::type>> state_type;
	state_type.emplace_back(std::unique_ptr<jive::state::type>(new jive::mem::type()));
	jive::store_op op(addrtype, state_type, dynamic_cast<const jive::value::type&>(value->type()));
	const jlm::frontend::tac * tac = basic_block->append(op, {address, value, state}, {result});
	return tac->output(0);
}

JIVE_EXPORTED_INLINE const jlm::frontend::variable *
addrarraysubscript_tac(
	jlm::frontend::basic_block * basic_block,
	const jlm::frontend::variable * base,
	const jlm::frontend::variable * offset,
	const jlm::frontend::variable * result)
{
	const jive::value::type & base_type = dynamic_cast<const jive::value::type&>(base->type());
	const jive::bits::type & offset_type = dynamic_cast<const jive::bits::type&>(offset->type());
	jive::address::arraysubscript_op op(base_type, offset_type);
	const jlm::frontend::tac * tac = basic_block->append(op, {base, offset}, {result});
	return tac->output(0);
}

}
}

#endif
