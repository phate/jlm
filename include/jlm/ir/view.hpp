/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_VIEW_HPP
#define JLM_IR_VIEW_HPP

#include <string>

namespace jlm {

class cfg;
class clg;
class module;

std::string
to_str(const jlm::cfg & cfg);

std::string
to_dot(const jlm::cfg & cfg);

std::string
to_str(const jlm::clg & clg);

std::string
to_str(const jlm::module & module);

static inline void
view_ascii(const jlm::cfg & cfg, FILE * out)
{
	fputs(to_str(cfg).c_str(), out);
	fflush(out);
}

static inline void
view_dot(const jlm::cfg & cfg, FILE * out)
{
	fputs(to_dot(cfg).c_str(), out);
	fflush(out);
}

static inline void
view(const jlm::clg & clg, FILE * out)
{
	fputs(to_str(clg).c_str(), out);
	fflush(out);
}

static inline void
view(const jlm::module & module, FILE * out)
{
	fputs(to_str(module).c_str(), out);
	fflush(out);
}

}

#endif
