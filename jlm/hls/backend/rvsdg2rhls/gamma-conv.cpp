/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/gamma-conv.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls {

void
gamma_conv(llvm::RvsdgModule &rm, bool allow_speculation) {
  auto &graph = rm.Rvsdg();
  auto root = graph.root();
  gamma_conv(root, allow_speculation);
}

void
gamma_conv(jlm::rvsdg::region *region, bool allow_speculation) {
  for (auto &node: jlm::rvsdg::topdown_traverser(region)) {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node)) {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
        gamma_conv(structnode->subregion(n), allow_speculation);
      if (auto gamma = dynamic_cast<jlm::rvsdg::gamma_node *>(node)) {
        if (allow_speculation && gamma_can_be_spec(gamma)) {
          gamma_conv_spec(gamma);
        } else {
          gamma_conv_nonspec(gamma);
        }
      }
    }
  }
}

void
gamma_conv_spec(jlm::rvsdg::gamma_node *gamma) {
  jlm::rvsdg::substitution_map smap;
  // connect arguments to origins of inputs. Forks will automatically be created later
  auto pro = gamma->predicate()->origin();
  for (size_t i = 0; i < gamma->nentryvars(); i++) {
    auto envo = gamma->entryvar(i)->origin();
    for (size_t s = 0; s < gamma->nsubregions(); s++) {
      smap.insert(gamma->subregion(s)->argument(i), envo);
    }
  }
  // copy each of the subregions
  for (size_t s = 0; s < gamma->nsubregions(); s++) {
    gamma->subregion(s)->copy(gamma->region(), smap, false, false);
  }
  for (size_t i = 0; i < gamma->nexitvars(); i++) {
    std::vector<jlm::rvsdg::output *> alternatives;
    for (size_t s = 0; s < gamma->nsubregions(); s++) {
      alternatives.push_back(smap.lookup(gamma->subregion(s)->result(i)->origin()));
    }
    // create discarding mux for each exitvar
    auto merge = hls::mux_op::create(*pro, alternatives, true);
    // divert users of exitvars to merge instead
    gamma->exitvar(i)->divert_users(merge[0]);
  }
  remove(gamma);
}

void
gamma_conv_nonspec(jlm::rvsdg::gamma_node *gamma) {
  jlm::rvsdg::substitution_map smap;
  // create a branch for each entryvar and map the corresponding argument of each subregion to an output of the branch
  auto pro = gamma->predicate()->origin();
  for (size_t i = 0; i < gamma->nentryvars(); i++) {
    auto envo = gamma->entryvar(i)->origin();
    auto bros = hls::branch_op::create(*pro, *envo);
    for (size_t s = 0; s < gamma->nsubregions(); s++) {
      smap.insert(gamma->subregion(s)->argument(i), bros[s]);
    }
  }
  // copy each of the subregions
  for (size_t s = 0; s < gamma->nsubregions(); s++) {
//                std::cout << "copying gamma subregion:\n";
//                jlm::rvsdg::view(gamma->subregion(s), stdout);
    gamma->subregion(s)->copy(gamma->region(), smap, false, false);
  }
  for (size_t i = 0; i < gamma->nexitvars(); i++) {
    std::vector<jlm::rvsdg::output *> alternatives;
    for (size_t s = 0; s < gamma->nsubregions(); s++) {
      alternatives.push_back(smap.lookup(gamma->subregion(s)->result(i)->origin()));
    }
    // create mux nodes for each exitvar
    // use mux instead of merge in case of paths with different delay - otherwise one could overtake the other
    // see https://ieeexplore.ieee.org/abstract/document/9515491
    auto mux = hls::mux_op::create(*pro, alternatives, false);
    // divert users of exitvars to mux instead
    gamma->exitvar(i)->divert_users(mux[0]);
  }
  remove(gamma);
}

bool
gamma_can_be_spec(jlm::rvsdg::gamma_node *gamma) {
  for (size_t i = 0; i < gamma->noutputs(); ++i) {
    auto out = gamma->output(i);
    if (jlm::rvsdg::is<jlm::rvsdg::statetype>(out->type())) {
      // don't allow state outputs since they imply operations with side effects
      return false;
    }
  }
  for (size_t i = 0; i < gamma->nsubregions(); ++i) {
    auto sr = gamma->subregion(i);
    for (auto &node: jlm::rvsdg::topdown_traverser(sr)) {
      if (jlm::rvsdg::is<jlm::rvsdg::theta_op>(node) || jlm::rvsdg::is<hls::loop_op>(node)) {
        // don't allow thetas or loops since they could potentially block forever
        return false;
      } else if (auto g = dynamic_cast<jlm::rvsdg::gamma_node *>(node)) {
        if (!gamma_can_be_spec(g)) {
          // only allow gammas that can also be speculated on
          return false;
        }
      } else if (dynamic_cast<jlm::rvsdg::structural_node *>(node)) {
        throw util::error("Unexpected structural node: " + node->operation().debug_string());
      }
    }
  }
  return true;
}

}