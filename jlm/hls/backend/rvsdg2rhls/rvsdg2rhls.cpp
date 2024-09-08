/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/backend/rvsdg2rhls/add-buffers.hpp>
#include <jlm/hls/backend/rvsdg2rhls/add-forks.hpp>
#include <jlm/hls/backend/rvsdg2rhls/add-prints.hpp>
#include <jlm/hls/backend/rvsdg2rhls/add-sinks.hpp>
#include <jlm/hls/backend/rvsdg2rhls/add-triggers.hpp>
#include <jlm/hls/backend/rvsdg2rhls/alloca-conv.hpp>
#include <jlm/hls/backend/rvsdg2rhls/check-rhls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/dae-conv.hpp>
#include <jlm/hls/backend/rvsdg2rhls/distribute-constants.hpp>
#include <jlm/hls/backend/rvsdg2rhls/GammaConversion.hpp>
#include <jlm/hls/backend/rvsdg2rhls/instrument-ref.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-conv.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-queue.hpp>
#include <jlm/hls/backend/rvsdg2rhls/mem-sep.hpp>
#include <jlm/hls/backend/rvsdg2rhls/memstate-conv.hpp>
#include <jlm/hls/backend/rvsdg2rhls/merge-gamma.hpp>
#include <jlm/hls/backend/rvsdg2rhls/remove-redundant-buf.hpp>
#include <jlm/hls/backend/rvsdg2rhls/remove-unused-state.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rhls-dne.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/hls/backend/rvsdg2rhls/ThetaConversion.hpp>
#include <jlm/hls/opt/cne.hpp>
#include <jlm/hls/util/view.hpp>
#include <jlm/llvm/backend/jlm2llvm/jlm2llvm.hpp>
#include <jlm/llvm/backend/rvsdg2jlm/rvsdg2jlm.hpp>
#include <jlm/llvm/ir/operators/alloca.hpp>
#include <jlm/llvm/ir/operators/call.hpp>
#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/llvm/opt/alias-analyses/Optimization.hpp>
#include <jlm/llvm/opt/DeadNodeElimination.hpp>
#include <jlm/llvm/opt/inlining.hpp>
#include <jlm/llvm/opt/InvariantValueRedirection.hpp>
#include <jlm/llvm/opt/inversion.hpp>
#include <jlm/llvm/opt/reduction.hpp>
#include <jlm/rvsdg/traverser.hpp>
#include <jlm/rvsdg/view.hpp>
#include <jlm/util/Statistics.hpp>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SourceMgr.h>

#include <regex>

namespace jlm::hls
{

void
split_opt(llvm::RvsdgModule & rm)
{
  // TODO: figure out which optimizations to use here
  jlm::llvm::DeadNodeElimination dne;
  jlm::hls::cne cne;
  jlm::llvm::InvariantValueRedirection ivr;
  jlm::llvm::tginversion tgi;
  jlm::llvm::nodereduction red;
  jlm::util::StatisticsCollector statisticsCollector;
  tgi.run(rm, statisticsCollector);
  dne.run(rm, statisticsCollector);
  cne.run(rm, statisticsCollector);
  ivr.run(rm, statisticsCollector);
  red.run(rm, statisticsCollector);
  dne.run(rm, statisticsCollector);
}

void
pre_opt(jlm::llvm::RvsdgModule & rm)
{
  // TODO: figure out which optimizations to use here
  jlm::llvm::DeadNodeElimination dne;
  jlm::hls::cne cne;
  jlm::llvm::InvariantValueRedirection ivr;
  jlm::llvm::tginversion tgi;
  jlm::util::StatisticsCollector statisticsCollector;
  tgi.run(rm, statisticsCollector);
  dne.run(rm, statisticsCollector);
  cne.run(rm, statisticsCollector);
  ivr.run(rm, statisticsCollector);
  dne.run(rm, statisticsCollector);
  cne.run(rm, statisticsCollector);
  dne.run(rm, statisticsCollector);
}

void
dump_xml(llvm::RvsdgModule & rvsdgModule, const std::string & file_name)
{
  auto xml_file = fopen(file_name.c_str(), "w");
  jlm::rvsdg::view_xml(rvsdgModule.Rvsdg().root(), xml_file);
  fclose(xml_file);
}

bool
function_match(llvm::lambda::node * ln, const std::string & function_name)
{
  const std::regex fn_regex(function_name);
  if (std::regex_match(ln->name(), fn_regex))
  { // TODO: handle C++ name mangling
    return true;
  }
  return false;
}

const jlm::rvsdg::output *
trace_call(jlm::rvsdg::input * input)
{
  auto graph = input->region()->graph();

  auto argument = dynamic_cast<const rvsdg::RegionArgument *>(input->origin());
  const jlm::rvsdg::output * result;
  if (auto to = dynamic_cast<const jlm::rvsdg::theta_output *>(input->origin()))
  {
    result = trace_call(to->input());
  }
  else if (argument == nullptr)
  {
    result = input->origin();
  }
  else if (argument->region() == graph->root())
  {
    result = argument;
  }
  else
  {
    JLM_ASSERT(argument->input() != nullptr);
    result = trace_call(argument->input());
  }
  return result;
}

void
inline_calls(jlm::rvsdg::region * region)
{
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        inline_calls(structnode->subregion(n));
      }
    }
    else if (dynamic_cast<const llvm::CallOperation *>(&(node->operation())))
    {
      auto traced = jlm::hls::trace_call(node->input(0));
      auto so = dynamic_cast<const jlm::rvsdg::structural_output *>(traced);
      if (!so)
      {
        if (auto graphImport = dynamic_cast<const llvm::GraphImport *>(traced))
        {
          if (graphImport->Name().rfind("decouple_", 0) == 0)
          {
            // can't inline pseudo functions used for decoupling
            continue;
          }
          throw jlm::util::error("can not inline external function " + graphImport->Name());
        }
      }
      JLM_ASSERT(rvsdg::is<llvm::lambda::operation>(so->node()));
      auto ln = dynamic_cast<const jlm::rvsdg::structural_output *>(traced)->node();
      llvm::inlineCall(
          dynamic_cast<jlm::rvsdg::simple_node *>(node),
          dynamic_cast<const llvm::lambda::node *>(ln));
      // restart for this region
      inline_calls(region);
      return;
    }
  }
}

size_t alloca_cnt = 0;

void
convert_alloca(jlm::rvsdg::region * region)
{
  for (auto & node : jlm::rvsdg::topdown_traverser(region))
  {
    if (auto structnode = dynamic_cast<jlm::rvsdg::structural_node *>(node))
    {
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        convert_alloca(structnode->subregion(n));
      }
    }
    else if (auto po = dynamic_cast<const llvm::alloca_op *>(&(node->operation())))
    {
      auto rr = region->graph()->root();
      auto delta_name = jlm::util::strfmt("hls_alloca_", alloca_cnt++);
      auto delta_type = llvm::PointerType::Create();
      std::cout << "alloca " << delta_name << ": " << po->value_type().debug_string() << "\n";
      auto db = llvm::delta::node::Create(
          rr,
          std::static_pointer_cast<const rvsdg::valuetype>(po->ValueType()),
          delta_name,
          llvm::linkage::external_linkage,
          "",
          false);
      // create zero constant of allocated type
      jlm::rvsdg::output * cout;
      if (auto bt = dynamic_cast<const jlm::rvsdg::bittype *>(&po->value_type()))
      {
        cout = jlm::rvsdg::create_bitconstant(db->subregion(), bt->nbits(), 0);
      }
      else
      {
        llvm::ConstantAggregateZero cop(po->ValueType());
        cout = jlm::rvsdg::simple_node::create_normalized(db->subregion(), cop, {})[0];
      }
      auto delta = db->finalize(cout);
      jlm::llvm::GraphExport::Create(*delta, delta_name);
      auto delta_local = route_to_region(delta, region);
      node->output(0)->divert_users(delta_local);
      // TODO: check that the input to alloca is a bitconst 1
      // TODO: handle general case of other nodes getting state edge without a merge
      JLM_ASSERT(node->output(1)->nusers() == 1);
      auto mux_in = *node->output(1)->begin();
      auto mux_node = rvsdg::input::GetNode(*mux_in);
      if (dynamic_cast<const llvm::MemoryStateMergeOperation *>(&mux_node->operation()))
      {
        // merge after alloca -> remove merge
        JLM_ASSERT(mux_node->ninputs() == 2);
        auto other_index = mux_in->index() ? 0 : 1;
        mux_node->output(0)->divert_users(mux_node->input(other_index)->origin());
        jlm::rvsdg::remove(mux_node);
      }
      else
      {
        // TODO: fix this properly by adding a state edge to the LambdaEntryMemState and routing it
        // to the region
        JLM_ASSERT(false);
      }
      jlm::rvsdg::remove(node);
    }
  }
}

llvm::delta::node *
rename_delta(llvm::delta::node * odn)
{
  auto name = odn->name();
  std::replace_if(
      name.begin(),
      name.end(),
      [](char c)
      {
        return c == '.';
      },
      '_');
  std::cout << "renaming delta node " << odn->name() << " to " << name << "\n";
  auto db = llvm::delta::node::Create(
      odn->region(),
      std::static_pointer_cast<const rvsdg::valuetype>(odn->Type()),
      name,
      llvm::linkage::external_linkage,
      "",
      odn->constant());
  /* add dependencies */
  jlm::rvsdg::substitution_map rmap;
  for (size_t i = 0; i < odn->ncvarguments(); i++)
  {
    auto input = odn->input(i);
    auto nd = db->add_ctxvar(input->origin());
    rmap.insert(input->argument(), nd);
  }

  /* copy subregion */
  odn->subregion()->copy(db->subregion(), rmap, false, false);

  auto result = rmap.lookup(odn->subregion()->result(0)->origin());
  auto data = db->finalize(result);

  odn->output()->divert_users(data);
  jlm::rvsdg::remove(odn);
  return static_cast<llvm::delta::node *>(jlm::rvsdg::node_output::node(data));
}

llvm::lambda::node *
change_linkage(llvm::lambda::node * ln, llvm::linkage link)
{
  auto lambda =
      llvm::lambda::node::create(ln->region(), ln->Type(), ln->name(), link, ln->attributes());

  /* add context variables */
  jlm::rvsdg::substitution_map subregionmap;
  for (auto & cv : ln->ctxvars())
  {
    auto origin = cv.origin();
    auto newcv = lambda->add_ctxvar(origin);
    subregionmap.insert(cv.argument(), newcv);
  }

  /* collect function arguments */
  for (size_t n = 0; n < ln->nfctarguments(); n++)
  {
    lambda->fctargument(n)->set_attributes(ln->fctargument(n)->attributes());
    subregionmap.insert(ln->fctargument(n), lambda->fctargument(n));
  }

  /* copy subregion */
  ln->subregion()->copy(lambda->subregion(), subregionmap, false, false);

  /* collect function results */
  std::vector<jlm::rvsdg::output *> results;
  for (auto & result : ln->fctresults())
    results.push_back(subregionmap.lookup(result.origin()));

  /* finalize lambda */
  lambda->finalize(results);

  divert_users(ln, outputs(lambda));
  jlm::rvsdg::remove(ln);

  return lambda;
}

std::unique_ptr<jlm::llvm::RvsdgModule>
split_hls_function(llvm::RvsdgModule & rm, const std::string & function_name)
{
  // TODO: use a different datastructure for rhls?
  // create a copy of rm
  auto rhls = llvm::RvsdgModule::Create(rm.SourceFileName(), rm.TargetTriple(), rm.DataLayout());
  std::cout << "processing " << rm.SourceFileName().name() << "\n";
  auto root = rm.Rvsdg().root();
  for (auto node : jlm::rvsdg::topdown_traverser(root))
  {
    if (auto ln = dynamic_cast<llvm::lambda::node *>(node))
    {
      if (!function_match(ln, function_name))
      {
        continue;
      }
      inline_calls(ln->subregion());
      split_opt(rm);
      //            convert_alloca(ln->subregion());
      jlm::rvsdg::substitution_map smap;
      for (size_t i = 0; i < ln->ninputs(); ++i)
      {
        auto orig_node_output = dynamic_cast<jlm::rvsdg::node_output *>(ln->input(i)->origin());
        if (!orig_node_output)
        {
          // handle decouple stuff
          auto oldGraphImport = dynamic_cast<llvm::GraphImport *>(ln->input(i)->origin());
          auto & newGraphImport = llvm::GraphImport::Create(
              rhls->Rvsdg(),
              oldGraphImport->ValueType(),
              oldGraphImport->Name(),
              oldGraphImport->Linkage());
          smap.insert(ln->input(i)->origin(), &newGraphImport);
          continue;
        }
        auto orig_node = orig_node_output->node();
        if (auto oln = dynamic_cast<llvm::lambda::node *>(orig_node))
        {
          throw jlm::util::error("Inlining of function " + oln->name() + " not supported");
        }
        else if (auto odn = dynamic_cast<llvm::delta::node *>(orig_node))
        {
          // modify name to not contain .
          if (odn->name().find('.') != std::string::npos)
          {
            odn = rename_delta(odn);
          }
          std::cout << "delta node " << odn->name() << ": " << odn->type().debug_string() << "\n";
          // add import for delta to rhls
          auto & graphImport = llvm::GraphImport::Create(
              rhls->Rvsdg(),
              odn->Type(),
              odn->name(),
              llvm::linkage::external_linkage);
          smap.insert(ln->input(i)->origin(), &graphImport);
          // add export for delta to rm
          // TODO: check if not already exported and maybe adjust linkage?
          jlm::llvm::GraphExport::Create(*odn->output(), odn->name());
        }
        else
        {
          throw jlm::util::error("Unsupported node type: " + orig_node->operation().debug_string());
        }
      }
      // copy function into rhls
      auto new_ln = ln->copy(rhls->Rvsdg().root(), smap);
      new_ln = change_linkage(new_ln, llvm::linkage::external_linkage);
      auto oldExport = ln->ComputeCallSummary()->GetRvsdgExport();
      jlm::llvm::GraphExport::Create(*new_ln->output(), oldExport ? oldExport->Name() : "");
      // add function as input to rm and remove it
      auto & graphImport = llvm::GraphImport::Create(
          rm.Rvsdg(),
          ln->Type(),
          ln->name(),
          llvm::linkage::external_linkage); // TODO: change linkage?
      ln->output()->divert_users(&graphImport);
      remove(ln);
      std::cout << "function " << new_ln->name() << " extracted for HLS\n";
      return rhls;
    }
  }
  throw jlm::util::error("HLS function " + function_name + " not found");
}

void
rvsdg2ref(llvm::RvsdgModule & rhls, std::string path)
{
  dump_ref(rhls, path);
}

void
rvsdg2rhls(llvm::RvsdgModule & rhls)
{
  pre_opt(rhls);
  merge_gamma(rhls);
  util::StatisticsCollector statisticsCollector;
  llvm::DeadNodeElimination llvmDne;
  llvmDne.run(rhls, statisticsCollector);

  mem_sep_argument(rhls);
  remove_unused_state(rhls);
  // main conversion steps
  distribute_constants(rhls);
  ConvertGammaNodes(rhls);
  ConvertThetaNodes(rhls);
  hls::cne hlsCne;
  hlsCne.run(rhls, statisticsCollector);
  // rhls optimization
  dne(rhls);
  alloca_conv(rhls);
  mem_queue(rhls);
  MemoryConverter(rhls);
  memstate_conv(rhls);
  remove_redundant_buf(rhls);
  // enforce 1:1 input output relationship
  add_sinks(rhls);
  add_forks(rhls);
  add_buffers(rhls, true);
  // ensure that all rhls rules are met
  check_rhls(rhls);
}

void
dump_ref(llvm::RvsdgModule & rhls, std::string & path)
{
  auto reference =
      llvm::RvsdgModule::Create(rhls.SourceFileName(), rhls.TargetTriple(), rhls.DataLayout());
  jlm::rvsdg::substitution_map smap;
  rhls.Rvsdg().root()->copy(reference->Rvsdg().root(), smap, true, true);
  pre_opt(*reference);
  instrument_ref(*reference);
  for (size_t i = 0; i < reference->Rvsdg().root()->narguments(); ++i)
  {
    auto graphImport =
        util::AssertedCast<const llvm::GraphImport>(reference->Rvsdg().root()->argument(i));
    std::cout << "impport " << graphImport->Name() << ": " << graphImport->type().debug_string()
              << "\n";
  }
  ::llvm::LLVMContext ctx;
  jlm::util::StatisticsCollector statisticsCollector;
  auto jm2 = llvm::rvsdg2jlm::rvsdg2jlm(*reference, statisticsCollector);
  auto lm2 = llvm::jlm2llvm::convert(*jm2, ctx);
  std::error_code EC;
  ::llvm::raw_fd_ostream os(path, EC);
  lm2->print(os, nullptr);
}

}
