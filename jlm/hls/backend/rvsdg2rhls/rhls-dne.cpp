/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/rhls-dne.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls {

bool
remove_unused_loop_outputs(hls::loop_node *ln) {
  bool any_changed = false;
  auto sr = ln->subregion();
  // go through in reverse because we remove some
  for (int i = ln->noutputs() - 1; i >= 0; --i) {
    auto out = ln->output(i);
    if (out->nusers() == 0) {
      assert(out->results.size() == 1);
      auto result = out->results.begin();
      sr->remove_result(result->index());
      ln->RemoveOutput(out->index());
      any_changed = true;
    }
  }
  return any_changed;
}

bool
remove_unused_loop_inputs(hls::loop_node *ln) {
  bool any_changed = false;
  auto sr = ln->subregion();
  // go through in reverse because we remove some
  for (int i = ln->ninputs() - 1; i >= 0; --i) {
    auto in = ln->input(i);
    assert(in->arguments.size() == 1);
    auto arg = in->arguments.begin();
    if (arg->nusers() == 0) {
      sr->remove_argument(arg->index());
      ln->remove_input(in->index());
      any_changed = true;
    }
  }
  // clean up unused arguments - only ones without an input should be left
  // go through in reverse because we remove some
  for (int i = sr->narguments() - 1; i >= 0; --i) {
    auto arg = sr->argument(i);
    if (auto ba = dynamic_cast<backedge_argument *>(arg)) {
      auto result = ba->result();
      assert(result->type() == arg->type());
      if (arg->nusers() == 0 || (arg->nusers() == 1 && result->origin() == arg)) {
        sr->remove_result(result->index());
        sr->remove_argument(arg->index());
      }
    } else {
      assert(arg->nusers() != 0);
    }
  }
  return any_changed;
}

bool
dne(jlm::rvsdg::region *sr) {
  bool any_changed = false;
  bool changed;
  do {
    changed = false;
    for (auto &node: jlm::rvsdg::bottomup_traverser(sr)) {
      if (!node->has_users()) {
        remove(node);
        changed = true;
      } else if (auto ln = dynamic_cast<hls::loop_node *>(node)) {
        changed |= remove_unused_loop_outputs(ln);
        changed |= remove_unused_loop_inputs(ln);
        changed |= dne(ln->subregion());
      }
    }
    any_changed |= changed;
  } while (changed);
  assert(sr->bottom_nodes.empty());
  return any_changed;
}

void
dne(llvm::RvsdgModule &rm) {
  auto &graph = rm.Rvsdg();
  auto root = graph.root();
  if (root->nodes.size() != 1) {
    throw util::error("Root should have only one node now");
  }
  auto ln = dynamic_cast<const llvm::lambda::node *>(root->nodes.begin().ptr());
  if (!ln) {
    throw util::error("Node needs to be a lambda");
  }
  dne(ln->subregion());
}

}