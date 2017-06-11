/*
 * Copyright 2014 2015 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM2JLM_TYPE_HPP
#define JLM_LLVM2JLM_TYPE_HPP

#include <memory>

namespace jive {
namespace base {
	class type;
}
}

namespace llvm {
	class Type;
}

namespace jlm {

class context;

std::unique_ptr<jive::base::type>
convert_type(const llvm::Type * type, context & ctx);

}

#endif
