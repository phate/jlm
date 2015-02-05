/*
 * Copyright 2013 2014 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_FRONTEND_BASIC_BLOCK_H
#define JLM_FRONTEND_BASIC_BLOCK_H

#include <jlm/frontend/cfg.hpp>
#include <jlm/frontend/cfg_node.hpp>

namespace jive {
	class operation;
}

namespace jlm {
namespace frontend {

class output;
class tac;

class basic_block final : public cfg_node {
public:
	virtual ~basic_block();

	virtual std::string debug_string() const override;

	const tac *
	append(const jive::operation & operation, const std::vector<const output*> & operands);

	const tac *
	append(const jive::operation & operation, const std::vector<const output*> & operands,
		const std::vector<const jlm::frontend::variable*> & variables);

	/*
		FIXME: add accessor functions
	*/
	inline std::list<const tac*> &
	tacs() noexcept
	{
		return tacs_;
	}

	inline const std::list<const tac*> &
	tacs() const noexcept
	{
		return tacs_;
	}

private:
	basic_block(jlm::frontend::cfg & cfg) noexcept;

	std::list<const tac*> tacs_;

	friend basic_block * cfg::create_basic_block();
};

}
}

#endif
