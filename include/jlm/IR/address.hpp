/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_ADDRESS_H
#define JLM_IR_ADDRESS_H

#include <jlm/IR/tac.hpp>

#include <jive/arch/address.h>
#include <jive/arch/addresstype.h>
#include <jive/arch/load.h>
#include <jive/arch/memorytype.h>
#include <jive/arch/store.h>
#include <jive/types/bitstring/type.h>

namespace jlm {

JIVE_EXPORTED_INLINE const jlm::variable *
addrload_tac(
	jlm::basic_block * basic_block,
	const jlm::variable * address,
	const jlm::variable * state,
	const jive::value::type & data_type,
	const jlm::variable * result)
{
	jive::addr::type addrtype;
	jive::addrload_op op({jive::mem::type()}, data_type);
	const jlm::tac * tac = basic_block->append(op, {address, state}, {result});
	return tac->output(0);
}

JIVE_EXPORTED_INLINE const jlm::variable *
addrstore_tac(
	jlm::basic_block * basic_block,
	const jlm::variable * address,
	const jlm::variable * value,
	const jlm::variable * state,
	const jlm::variable * result)
{
	jive::addr::type addrtype;
	jive::addrstore_op op({jive::mem::type()}, dynamic_cast<const jive::value::type&>(value->type()));
	const jlm::tac * tac = basic_block->append(op, {address, value, state}, {result});
	return tac->output(0);
}

JIVE_EXPORTED_INLINE const jlm::variable *
addrarraysubscript_tac(
	jlm::basic_block * basic_block,
	const jlm::variable * base,
	const jlm::variable * offset,
	const jlm::variable * result)
{
	const jive::value::type & base_type = dynamic_cast<const jive::value::type&>(base->type());
	const jive::bits::type & offset_type = dynamic_cast<const jive::bits::type&>(offset->type());
	jive::address::arraysubscript_op op(base_type, offset_type);
	const jlm::tac * tac = basic_block->append(op, {base, offset}, {result});
	return tac->output(0);
}

}

#endif
