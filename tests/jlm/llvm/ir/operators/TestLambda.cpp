/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/ir/operators/lambda.hpp>

static void
TestArgumentIterators()
{
  using namespace jlm;

  valuetype vt;
  RvsdgModule rvsdgModule(filepath(""), "", "");

  {
    FunctionType functionType({&vt}, {&vt});

    auto lambda = lambda::node::create(
      rvsdgModule.Rvsdg().root(),
      functionType,
      "f",
      linkage::external_linkage);
    lambda->finalize({lambda->fctargument(0)});

    std::vector<jive::argument*> functionArguments;
    for (auto & argument : lambda->fctarguments())
      functionArguments.push_back(&argument);

    assert(functionArguments.size() == 1
           && functionArguments[0] == lambda->fctargument(0));
  }

  {
    FunctionType functionType({}, {&vt});

    auto lambda = lambda::node::create(
      rvsdgModule.Rvsdg().root(),
      functionType,
      "f",
      linkage::external_linkage);

    auto nullaryNode = create_testop(lambda->subregion(), {}, {&vt});

    lambda->finalize({nullaryNode});

    assert(lambda->nfctarguments() == 0);
  }

  {
    auto rvsdgImport = rvsdgModule.Rvsdg().add_import({vt, ""});

    FunctionType functionType({&vt, &vt, &vt}, {&vt, &vt});

    auto lambda = lambda::node::create(
      rvsdgModule.Rvsdg().root(),
      functionType,
      "f",
      linkage::external_linkage);

    auto cv = lambda->add_ctxvar(rvsdgImport);

    lambda->finalize({lambda->fctargument(0), cv});

    std::vector<jive::argument*> functionArguments;
    for (auto & argument : lambda->fctarguments())
      functionArguments.push_back(&argument);

    assert(functionArguments.size() == 3);
    assert(functionArguments[0] == lambda->fctargument(0));
    assert(functionArguments[1] == lambda->fctargument(1));
    assert(functionArguments[2] == lambda->fctargument(2));
  }
}

static void
TestInvalidOperandRegion()
{
  using namespace jlm;

  valuetype vt;
  FunctionType functionType({}, {&vt});

  auto rvsdgModule = RvsdgModule::Create(filepath(""), "", "");
  auto rvsdg = &rvsdgModule->Rvsdg();

  auto lambdaNode = lambda::node::create(
    rvsdg->root(),
    functionType,
    "f",
    linkage::external_linkage);
  auto result = create_testop(rvsdg->root(), {}, {&vt})[0];

  bool invalidRegionErrorCaught = false;
  try {
    lambdaNode->finalize({result});
  } catch (jlm::error&) {
    invalidRegionErrorCaught = true;
  }

  assert(invalidRegionErrorCaught);
}

static int
Test()
{
  TestArgumentIterators();
  TestInvalidOperandRegion();

  return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/llvm/ir/operators/TestLambda", Test)
