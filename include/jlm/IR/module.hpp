/*
 * Copyright 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_MODULE_HPP
#define JLM_IR_MODULE_HPP

#include <jlm/IR/clg.hpp>

namespace jlm {

class module final {
public:
	inline
	~module()
	{}

	inline
	module() noexcept
	{}

	inline jlm::clg &
	clg() noexcept
	{
		return clg_;
	}

private:
	jlm::clg clg_;
};

}

#endif
