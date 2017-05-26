/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/IR/aggregation/node.hpp>

#include <string>

namespace jlm {
namespace agg {

std::string
to_str(const node & n)
{
  std::function<std::string(const node & n, size_t)> f = [&] (
    const node & n,
    size_t depth
  ) {
    std::string subtree(depth, '-');
    subtree += n.structure().debug_string() + "\n";

    for (const auto & child : n)
      subtree += f(child, depth+1);

    return subtree;
  };

  return f(n, 0);
}

void
view(const node & n, FILE * out)
{
	fputs(to_str(n).c_str(), out);
	fflush(out);
}

}}
