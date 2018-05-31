/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_IR_VIEW_HPP
#define JLM_IR_VIEW_HPP

/* FIXME: I would rather like to forward declare demand_map and demand_set */
#include <jlm/jlm/ir/aggregation/annotation.hpp>

#include <string>

namespace jlm {
namespace agg {
	class aggnode;
}

class cfg;
class ipgraph;
class module;

std::string
to_str(const jlm::cfg & cfg);

std::string
to_dot(const jlm::cfg & cfg);

std::string
to_str(const jlm::ipgraph & ipg);

std::string
to_dot(const jlm::ipgraph & ipg);

std::string
to_str(const jlm::module & module);

std::string
to_str(const agg::aggnode & n, const agg::demand_map & dm);

static inline std::string
to_str(const agg::aggnode & n)
{
	return to_str(n, {});
}

void
view(const agg::aggnode & n, const agg::demand_map & dm, FILE * out);

static inline void
view(const agg::aggnode & n, FILE * out)
{
	view(n, {}, out);
}

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
view_ascii(const jlm::ipgraph & ipg, FILE * out)
{
	fputs(to_str(ipg).c_str(), out);
	fflush(out);
}

static inline void
view_dot(const jlm::ipgraph & ipg, FILE * out)
{
	fputs(to_dot(ipg).c_str(), out);
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
