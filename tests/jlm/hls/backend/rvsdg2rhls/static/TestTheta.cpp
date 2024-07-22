/*
* Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
* See COPYING for terms of redistribution.
*/
#include "test-registry.hpp"

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/rvsdg/theta.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/traverser.hpp>

#include <jlm/rvsdg/view.hpp>

#include <jlm/hls/backend/rvsdg2rhls/static/rvsdg2rhls.hpp>
#include <jlm/hls/ir/static/loop.hpp>

std::unique_ptr<jlm::llvm::RvsdgModule>
CreateTestModule() {
  using namespace jlm::llvm;

  auto module = jlm::llvm::RvsdgModule::Create(jlm::util::filepath(""), "", "");
  auto graph = &module->Rvsdg();

  auto nf = graph->node_normal_form(typeid(jlm::rvsdg::operation));
  nf->set_mutable(false);

  MemoryStateType mt;
  auto pointerType = PointerType::Create();
  auto fcttype = FunctionType::Create(
        { jlm::rvsdg::bittype::Create(32) },
        // { jlm::rvsdg::bittype::Create(32), jlm::rvsdg::bittype::Create(32) },
        { jlm::rvsdg::bittype::Create(32) });
  auto fct = lambda::node::create(graph->root(), fcttype, "f", linkage::external_linkage);

  auto thetanode = jlm::rvsdg::theta_node::create(fct->subregion());

  auto sum_loop_var = thetanode->add_loopvar(fct->fctargument(0));
  // auto loop_var2 = thetanode->add_loopvar(fct->fctargument(1));

  auto one = jlm::rvsdg::create_bitconstant(thetanode->subregion(), 32, 1);
  auto five = jlm::rvsdg::create_bitconstant(thetanode->subregion(), 32, 5);
  auto sum = jlm::rvsdg::bitadd_op::create(32, sum_loop_var->argument(), one);
  auto cmp = jlm::rvsdg::bitult_op::create(32, sum, five);
  auto predicate = jlm::rvsdg::match(1, { { 1, 1 } }, 0, 2, cmp);

  // auto sum2 = jlm::rvsdg::bitadd_op::create(32, sum_loop_var->argument(), loop_var2->argument());

  // change to loop_var result origin to the ouput of sum
  // (by default the loop_var result origin is connected to the loop_var argument (to itself))
  sum_loop_var->result()->divert_to(sum);

  // loop_var2->result()->divert_to(sum2);

  thetanode->set_predicate(predicate);

  fct->finalize({ sum_loop_var });

  // Make the function external func
  graph->add_export(fct->output(), jlm::rvsdg::expport(pointerType, "f"));

  return module;
}

static int
TestTheta() {
    auto rvsdgModule = CreateTestModule();
    
    std::cout << "**********  Original graph  **********" << std::endl;
    jlm::rvsdg::view(*rvsdgModule->Rvsdg().root()->graph(), stdout);


    std::cout << "**********  Running static rvsdg2rhls  **********" << std::endl;
    jlm::static_hls::rvsdg2rhls(*rvsdgModule);

    std::cout << "**********  Converted graph  **********" << std::endl;
    jlm::rvsdg::view(*rvsdgModule->Rvsdg().root()->graph(), stdout);



    auto lambda = &*rvsdgModule->Rvsdg().root()->begin();
    auto lambda_node = jlm::util::AssertedCast<jlm::llvm::lambda::node>(lambda);

    auto loop = &*lambda_node->subregion()->begin();
    auto loop_node = jlm::util::AssertedCast<jlm::static_hls::loop_node>(loop);

    auto orig_module = CreateTestModule();

    auto orig_lambda = &*orig_module->Rvsdg().root()->begin();
    auto orig_lambda_node = static_cast<jlm::llvm::lambda::node*>(orig_lambda);

    auto orig_theta = &*orig_lambda_node->subregion()->begin();
    auto orig_theta_node = static_cast<jlm::rvsdg::theta_node*>(orig_theta);



    for (auto& node : jlm::rvsdg::topdown_traverser(orig_theta_node->subregion()))
    {
      auto imp_node = loop_node->is_op_implemented(node->operation());
      JLM_ASSERT(imp_node);

      for (size_t i=0; i<imp_node->ninputs(); i++)
      {
        if (!dynamic_cast<const jlm::rvsdg::node_output*>(imp_node->input(i)->origin())) continue;

        bool origin_found_in_users = false;
        for (auto user : loop_node->get_users(imp_node->input(i)))
        {
          auto reg_smap = loop_node->get_reg_smap();
          auto output_origin = reg_smap->lookup(imp_node->input(i)->origin());
          if (!output_origin) 
          {
            std::cout << "output_origin not in reg_smap for " << imp_node->operation().debug_string() << "input " << i << std::endl;
          }
          if (output_origin == user) origin_found_in_users = true;
        }
        JLM_ASSERT(origin_found_in_users);
      }
      
    }

    return 0;
}

JLM_UNIT_TEST_REGISTER("jlm/hls/backend/rvsdg2rhls/static/TestTheta", TestTheta)