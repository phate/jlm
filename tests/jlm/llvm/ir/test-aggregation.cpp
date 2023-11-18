/*
 * Copyright 2017 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include "test-registry.hpp"

#include <jlm/llvm/ir/aggregation.hpp>
#include <jlm/llvm/ir/ipgraph-module.hpp>
#include <jlm/llvm/ir/print.hpp>

static bool
is_entry(const jlm::llvm::aggnode * node)
{
  return jlm::llvm::is<jlm::llvm::entryaggnode>(node) && node->nchildren() == 0;
}

static bool
is_exit(const jlm::llvm::aggnode * node)
{
  return jlm::llvm::is<jlm::llvm::exitaggnode>(node) && node->nchildren() == 0;
}

static bool
is_block(const jlm::llvm::aggnode * node)
{
  return jlm::llvm::is<jlm::llvm::blockaggnode>(node) && node->nchildren() == 0;
}

static bool
is_linear(const jlm::llvm::aggnode * node, size_t nchildren)
{
  if (!jlm::llvm::is<jlm::llvm::linearaggnode>(node))
    return false;

  if (node->nchildren() != nchildren)
    return false;

  for (auto & child : *node)
  {
    if (child.parent() != node)
      return false;
  }

  return true;
}

static bool
is_loop(const jlm::llvm::aggnode * node)
{
  return jlm::llvm::is<jlm::llvm::loopaggnode>(node) && node->nchildren() == 1
      && node->child(0)->parent() == node;
}

static bool
is_branch(const jlm::llvm::aggnode * node, size_t nchildren)
{
  if (!jlm::llvm::is<jlm::llvm::branchaggnode>(node))
    return false;

  if (node->nchildren() != nchildren)
    return false;

  for (auto & child : *node)
  {
    if (child.parent() != node)
      return false;
  }

  return true;
}

static void
test_linear_reduction()
{
  using namespace jlm::llvm;

  auto setup_cfg = [](jlm::llvm::ipgraph_module & module)
  {
    auto cfg = cfg::create(module);

    auto bb = jlm::llvm::basic_block::create(*cfg);
    cfg->exit()->divert_inedges(bb);
    bb->add_outedge(cfg->exit());

    return cfg;
  };

  auto verify_aggtree = [](const jlm::llvm::aggnode & root)
  {
    assert(is_linear(&root, 3));
    {
      assert(is_entry(root.child(0)));
      assert(is_block(root.child(1)));
      assert(is_exit(root.child(2)));
    }
  };

  jlm::llvm::ipgraph_module module(jlm::util::filepath(""), "", "");
  auto cfg = setup_cfg(module);

  auto root = jlm::llvm::aggregate(*cfg);
  jlm::llvm::aggnode::normalize(*root);
  jlm::llvm::print(*root, stdout);

  verify_aggtree(*root);
}

static void
test_loop_reduction()
{
  using namespace jlm::llvm;

  auto setup_cfg = [](jlm::llvm::ipgraph_module & module)
  {
    auto cfg = cfg::create(module);

    auto bb1 = jlm::llvm::basic_block::create(*cfg);
    auto bb2 = jlm::llvm::basic_block::create(*cfg);

    cfg->exit()->divert_inedges(bb1);
    bb1->add_outedge(bb2);
    bb2->add_outedge(cfg->exit());
    bb2->add_outedge(bb1);

    return cfg;
  };

  auto verify_aggtree = [](const jlm::llvm::aggnode & root)
  {
    assert(is_linear(&root, 3));
    {
      assert(is_entry(root.child(0)));

      auto loop = root.child(1);
      assert(is_loop(loop));
      {
        auto linear = loop->child(0);
        assert(is_linear(linear, 2));
        {
          assert(is_block(linear->child(0)));
          assert(is_block(linear->child(1)));
        }
      }

      assert(is_exit(root.child(2)));
    }
  };

  jlm::llvm::ipgraph_module module(jlm::util::filepath(""), "", "");
  auto cfg = setup_cfg(module);

  auto root = jlm::llvm::aggregate(*cfg);
  jlm::llvm::aggnode::normalize(*root);
  jlm::llvm::print(*root, stdout);

  verify_aggtree(*root);
}

static void
test_branch_reduction()
{
  using namespace jlm::llvm;

  auto setup_cfg = [](jlm::llvm::ipgraph_module & module)
  {
    auto cfg = cfg::create(module);

    auto split = jlm::llvm::basic_block::create(*cfg);
    auto bb1 = jlm::llvm::basic_block::create(*cfg);
    auto bb2 = jlm::llvm::basic_block::create(*cfg);
    auto bb3 = jlm::llvm::basic_block::create(*cfg);
    auto bb4 = jlm::llvm::basic_block::create(*cfg);
    auto join = jlm::llvm::basic_block::create(*cfg);

    cfg->exit()->divert_inedges(split);
    split->add_outedge(bb1);
    split->add_outedge(bb3);
    bb1->add_outedge(bb2);
    bb2->add_outedge(join);
    bb3->add_outedge(bb4);
    bb4->add_outedge(join);
    join->add_outedge(cfg->exit());

    return cfg;
  };

  auto verify_aggtree = [](const jlm::llvm::aggnode & root)
  {
    assert(is_linear(&root, 5));
    {
      assert(is_entry(root.child(0)));
      assert(is_block(root.child(1)));

      auto branch = root.child(2);
      assert(is_branch(branch, 2));
      {
        auto linear = branch->child(0);
        assert(is_linear(linear, 2));
        {
          assert(is_block(linear->child(0)));
          assert(is_block(linear->child(1)));
        }

        linear = branch->child(1);
        assert(is_linear(linear, 2));
        {
          assert(is_block(linear->child(0)));
          assert(is_block(linear->child(1)));
        }
      }

      assert(is_block(root.child(3)));
      assert(is_exit(root.child(4)));
    }
  };

  jlm::llvm::ipgraph_module module(jlm::util::filepath(""), "", "");
  auto cfg = setup_cfg(module);

  auto root = jlm::llvm::aggregate(*cfg);
  jlm::llvm::aggnode::normalize(*root);
  jlm::llvm::print(*root, stdout);

  verify_aggtree(*root);
}

static void
test_branch_loop_reduction()
{
  using namespace jlm::llvm;

  auto setup_cfg = [](jlm::llvm::ipgraph_module & module)
  {
    auto cfg = cfg::create(module);
    auto split = jlm::llvm::basic_block::create(*cfg);
    auto bb1 = jlm::llvm::basic_block::create(*cfg);
    auto bb2 = jlm::llvm::basic_block::create(*cfg);
    auto bb3 = jlm::llvm::basic_block::create(*cfg);
    auto bb4 = jlm::llvm::basic_block::create(*cfg);
    auto join = jlm::llvm::basic_block::create(*cfg);

    cfg->exit()->divert_inedges(split);
    split->add_outedge(bb1);
    split->add_outedge(bb3);
    bb1->add_outedge(bb2);
    bb2->add_outedge(join);
    bb2->add_outedge(bb1);
    bb3->add_outedge(bb4);
    bb4->add_outedge(join);
    bb4->add_outedge(bb3);
    join->add_outedge(cfg->exit());

    return cfg;
  };

  auto verify_aggtree = [](const jlm::llvm::aggnode & root)
  {
    assert(is_linear(&root, 5));
    {
      assert(is_entry(root.child(0)));
      assert(is_block(root.child(1)));

      auto branch = root.child(2);
      assert(is_branch(branch, 2));
      {
        auto loop = branch->child(0);
        assert(is_loop(loop));
        {
          auto linear = loop->child(0);
          assert(is_linear(linear, 2));
          {
            assert(is_block(linear->child(0)));
            assert(is_block(linear->child(1)));
          }
        }

        loop = branch->child(1);
        assert(is_loop(loop));
        {
          auto linear = loop->child(0);
          assert(is_linear(linear, 2));
          {
            assert(is_block(linear->child(0)));
            assert(is_block(linear->child(1)));
          }
        }
      }

      assert(is_block(root.child(3)));
      assert(is_exit(root.child(4)));
    }
  };

  jlm::llvm::ipgraph_module module(jlm::util::filepath(""), "", "");
  auto cfg = setup_cfg(module);

  auto root = jlm::llvm::aggregate(*cfg);
  jlm::llvm::aggnode::normalize(*root);
  jlm::llvm::print(*root, stdout);

  verify_aggtree(*root);
}

static void
test_loop_branch_reduction()
{
  using namespace jlm::llvm;

  auto setup_cfg = [](jlm::llvm::ipgraph_module & module)
  {
    auto cfg = cfg::create(module);

    auto split = jlm::llvm::basic_block::create(*cfg);
    auto bb1 = jlm::llvm::basic_block::create(*cfg);
    auto bb2 = jlm::llvm::basic_block::create(*cfg);
    auto join = jlm::llvm::basic_block::create(*cfg);
    auto bb3 = jlm::llvm::basic_block::create(*cfg);

    cfg->exit()->divert_inedges(split);
    split->add_outedge(bb1);
    split->add_outedge(bb2);
    bb1->add_outedge(join);
    bb2->add_outedge(join);
    join->add_outedge(bb3);
    bb3->add_outedge(cfg->exit());
    bb3->add_outedge(split);

    return cfg;
  };

  auto verify_aggtree = [](const jlm::llvm::aggnode & root)
  {
    assert(is_linear(&root, 3));
    {
      assert(is_entry(root.child(0)));

      auto loop = root.child(1);
      assert(is_loop(loop));
      {
        auto linear = loop->child(0);
        assert(is_linear(linear, 4));
        {
          assert(is_block(linear->child(0)));

          auto branch = linear->child(1);
          assert(is_branch(branch, 2));
          {
            assert(is_block(branch->child(0)));
            assert(is_block(branch->child(1)));
          }

          assert(is_block(linear->child(2)));
          assert(is_block(linear->child(3)));
        }
      }

      assert(is_exit(root.child(2)));
    }
  };

  jlm::llvm::ipgraph_module module(jlm::util::filepath(""), "", "");
  auto cfg = setup_cfg(module);

  auto root = jlm::llvm::aggregate(*cfg);
  jlm::llvm::aggnode::normalize(*root);
  jlm::llvm::print(*root, stdout);

  verify_aggtree(*root);
}

static void
test_ifthen_reduction()
{
  using namespace jlm::llvm;

  auto setup_cfg = [](jlm::llvm::ipgraph_module & module)
  {
    auto cfg = cfg::create(module);

    auto split = jlm::llvm::basic_block::create(*cfg);
    auto bb1 = jlm::llvm::basic_block::create(*cfg);
    auto bb2 = jlm::llvm::basic_block::create(*cfg);
    auto bb3 = jlm::llvm::basic_block::create(*cfg);
    auto join = jlm::llvm::basic_block::create(*cfg);

    cfg->exit()->divert_inedges(split);
    split->add_outedge(bb3);
    split->add_outedge(bb1);
    bb1->add_outedge(bb2);
    bb2->add_outedge(join);
    bb3->add_outedge(join);
    join->add_outedge(cfg->exit());

    return cfg;
  };

  auto verify_aggtree = [](const jlm::llvm::aggnode & root)
  {
    assert(is_linear(&root, 5));
    {
      assert(is_entry(root.child(0)));
      assert(is_block(root.child(1)));

      auto branch = root.child(2);
      assert(is_branch(branch, 2));
      {
        assert(is_block(branch->child(0)));

        auto linear = branch->child(1);
        assert(is_linear(linear, 2));
        {
          assert(is_block(linear->child(0)));
          assert(is_block(linear->child(1)));
        }
      }

      assert(is_block(root.child(3)));
      assert(is_exit(root.child(4)));
    }
  };

  jlm::llvm::ipgraph_module module(jlm::util::filepath(""), "", "");
  auto cfg = setup_cfg(module);

  auto root = jlm::llvm::aggregate(*cfg);
  jlm::llvm::aggnode::normalize(*root);
  jlm::llvm::print(*root, stdout);

  verify_aggtree(*root);
}

static void
test_branch_and_loop()
{
  using namespace jlm::llvm;

  auto setup_cfg = [](jlm::llvm::ipgraph_module & module)
  {
    auto cfg = cfg::create(module);

    auto split = jlm::llvm::basic_block::create(*cfg);
    auto bb1 = jlm::llvm::basic_block::create(*cfg);
    auto bb2 = jlm::llvm::basic_block::create(*cfg);
    auto loop = jlm::llvm::basic_block::create(*cfg);

    cfg->exit()->divert_inedges(split);
    split->add_outedge(bb1);
    split->add_outedge(bb2);
    bb1->add_outedge(loop);
    bb2->add_outedge(loop);
    loop->add_outedge(cfg->exit());
    loop->add_outedge(loop);

    return cfg;
  };

  auto verify_aggtree = [](const jlm::llvm::aggnode & root)
  {
    assert(is_linear(&root, 5));
    {
      assert(is_entry(root.child(0)));
      assert(is_block(root.child(1)));

      auto branch = root.child(2);
      assert(is_branch(branch, 2));
      {
        assert(is_block(branch->child(0)));
        assert(is_block(branch->child(1)));
      }

      auto loop = root.child(3);
      assert(is_loop(loop));
      {
        assert(is_block(loop->child(0)));
      }

      assert(is_exit(root.child(4)));
    }
  };

  jlm::llvm::ipgraph_module module(jlm::util::filepath(""), "", "");
  auto cfg = setup_cfg(module);

  auto root = jlm::llvm::aggregate(*cfg);
  jlm::llvm::aggnode::normalize(*root);
  jlm::llvm::print(*root, stdout);

  verify_aggtree(*root);
}

static int
test()
{
  test_linear_reduction();
  test_loop_reduction();
  test_branch_reduction();
  test_branch_loop_reduction();
  test_loop_branch_reduction();
  test_ifthen_reduction();
  test_branch_and_loop();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/test-aggregation", test)
