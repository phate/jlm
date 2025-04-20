/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include "hls-function-util.hpp"
#include "jlm/hls/util/view.hpp"
#include "jlm/llvm/ir/operators/operators.hpp"
#include "jlm/rvsdg/lambda.hpp"
#include <jlm/hls/backend/rvsdg2rhls/add-buffers.hpp>
#include <jlm/hls/backend/rvsdg2rhls/rvsdg2rhls.hpp>
#include <jlm/hls/ir/hls.hpp>
#include <jlm/rvsdg/traverser.hpp>

namespace jlm::hls
{
rvsdg::input *
GetUser(rvsdg::output * out)
{
  // This works because at this point we have 1:1 relationships through forks
  auto user = *out->begin();
  JLM_ASSERT(user);
  return user;
}

rvsdg::input *
FindUserNode(rvsdg::output * out)
{

  auto user = GetUser(out);
  if (auto br = dynamic_cast<backedge_result *>(user))
  {
    return FindUserNode(br->argument());
  }
  else if (auto rr = dynamic_cast<rvsdg::RegionResult *>(user))
  {

    if (rr->output() && rvsdg::TryGetOwnerNode<loop_node>(*rr->output()))
    {
      return FindUserNode(rr->output());
    }
    else
    {
      // lambda result
      return rr;
    }
  }
  else if (auto si = dynamic_cast<rvsdg::StructuralInput *>(user))
  {
    JLM_ASSERT(rvsdg::TryGetOwnerNode<loop_node>(*user));
    return FindUserNode(si->arguments.begin().ptr());
  }
  else
  {
    auto result = dynamic_cast<rvsdg::SimpleInput *>(user);
    JLM_ASSERT(result);
    return result;
  }
}

void
PlaceBuffer(rvsdg::output * out, size_t capacity, bool passThrough)
{
  // places or re-places a buffer on an output
  auto user = FindUserNode(out);
  // don't place buffers after constants
  if (is_constant(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*out)))
  {
    return;
  }
  auto fork = TryGetOwnerOp<fork_op>(*out);
  if (fork && fork->IsConstant())
  {
    return;
  }

  // TODO: handle out being a buf?
  auto buf = TryGetOwnerOp<buffer_op>(*user);
  if (buf && (buf->pass_through != passThrough || buf->capacity != capacity))
  {
    // replace buffer and keep larger size
    auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*user);
    passThrough = passThrough && buf->pass_through;
    capacity = std::max(capacity, buf->capacity);
    auto bufOut = buffer_op::create(*node->input(0)->origin(), capacity, passThrough)[0];
    node->output(0)->divert_users(bufOut);
    JLM_ASSERT(node->IsDead());
    remove(node);
  }
  else
  {
    // create new buffer
    auto directUser = *out->begin();
    auto newOut = buffer_op::create(*out, capacity, passThrough)[0];
    directUser->divert_to(newOut);
  }
}

const size_t BufferSizeForkState = 1;
const size_t BufferSizeForkControl = 8;
const size_t BufferSizeForkOther = 4;

void
OptimizeFork(rvsdg::SimpleNode * node)
{
  auto fork = dynamic_cast<const fork_op *>(&node->GetOperation());
  JLM_ASSERT(fork);
  bool inLoop = rvsdg::is<loop_op>(node->region()->node());
  if (fork->IsConstant() || !inLoop)
  {
    // cForks and forks outside of loops should have no buffers after it
    for (size_t i = 0; i < node->noutputs(); ++i)
    {
      auto user = FindUserNode(node->output(0));
      auto buf = TryGetOwnerOp<buffer_op>(*user);
      if (buf)
      {
        auto bufNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*user);
        bufNode->output(0)->divert_users(node->output(0));
        JLM_ASSERT(bufNode->IsDead());
        remove(bufNode);
      }
    }
  }
  else
  {
    // forks inside of loops should have buffers after it
    size_t bufferSize = BufferSizeForkOther;
    if (rvsdg::is<rvsdg::ControlType>(node->input(0)->type()))
    {
      bufferSize = BufferSizeForkControl;
    }
    else if (rvsdg::is<rvsdg::StateType>(node->input(0)->type()))
    {
      bufferSize = BufferSizeForkState;
    }
    for (size_t i = 0; i < node->noutputs(); ++i)
    {
      PlaceBuffer(node->output(i), bufferSize, true);
    }
  }
}

const size_t BufferSizeBranchState = BufferSizeForkControl;

void
OptimizeBranch(rvsdg::SimpleNode * node)
{
  auto branch = dynamic_cast<const branch_op *>(&node->GetOperation());
  JLM_ASSERT(branch);
  bool inLoop = rvsdg::is<loop_op>(node->region()->node());
  if (inLoop && !branch->loop)
  {
    // TODO: this optimization is for long stores with responses. It might be better to do it
    // somewhere else and more selectively (only when there is a store in one of the gamma
    // subregions, and only on outputs that don't go to store)
    if (rvsdg::is<rvsdg::StateType>(node->input(1)->type()))
    {
      for (size_t i = 0; i < node->noutputs(); ++i)
      {
        PlaceBuffer(node->output(i), BufferSizeBranchState, true);
      }
    }
  }
}

void
OptimizeStateGate(rvsdg::SimpleNode * node)
{
  // TODO: remove duplicate? somewhere else?
  // TODO: place buffers on state outputs?
}

void
OptimizeAddrQ(rvsdg::SimpleNode * node)
{
  auto addrq = dynamic_cast<const addr_queue_op *>(&node->GetOperation());
  JLM_ASSERT(addrq);
  // place buffer on addr output
  PlaceBuffer(node->output(0), addrq->capacity, true);
}

void
OptimizeBuffer(rvsdg::SimpleNode * node)
{
  auto buf = dynamic_cast<const buffer_op *>(&node->GetOperation());
  JLM_ASSERT(buf);
  auto user = FindUserNode(node->output(0));
  auto buf2 = TryGetOwnerOp<buffer_op>(*user);
  if (buf2)
  {
    auto node2 = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*user);
    // merge buffers and keep larger size
    bool passThrough = buf->pass_through && buf2->pass_through;
    auto capacity = std::max(buf->capacity, buf2->capacity);
    auto newOut = buffer_op::create(*node->input(0)->origin(), capacity, passThrough)[0];
    JLM_ASSERT(node2->region() == newOut->region());
    node2->output(0)->divert_users(newOut);
    JLM_ASSERT(node2->IsDead());
    remove(node2);
    JLM_ASSERT(node->IsDead());
    remove(node);
  }
}

void
OptimizeLoop(loop_node * loopNode)
{
  // TODO: should this be changed?
  bool outerLoop = !rvsdg::is<loop_op>(loopNode->region()->node());
  if (outerLoop)
  {
    // push buffers above branches, so they also act as output buffers
    for (size_t i = 0; i < loopNode->noutputs(); ++i)
    {
      auto out = loopNode->output(i);
      auto res = out->results.begin().ptr();
      auto branch = TryGetOwnerOp<branch_op>(*res->origin());
      if (!branch)
      {
        // this is a memory operation or stream
        continue;
      }
      JLM_ASSERT(branch->loop);
      auto branchNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*res->origin());
      auto oldBufInput = GetUser(branchNode->output(1));
      auto oldBuf = TryGetOwnerOp<buffer_op>(*oldBufInput);
      auto isSink = TryGetOwnerOp<sink_op>(*oldBufInput);
      if (isSink)
      {
        // no backedge
        continue;
      }
      JLM_ASSERT(oldBuf);
      auto oldBufNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*oldBufInput);
      // place new buffers
      PlaceBuffer(branchNode->input(1)->origin(), oldBuf->capacity, oldBuf->pass_through);
      // this buffer should just make the fork buf non-passthrough - needed to avoid combinatorial
      // cycle
      PlaceBuffer(branchNode->input(0)->origin(), oldBuf->capacity, oldBuf->pass_through);
      // remove old buffer
      oldBufNode->output(0)->divert_users(oldBufInput->origin());
      JLM_ASSERT(oldBufNode->IsDead());
      remove(oldBufNode);
    }
  }
  else
  {
    // add input buffers
    for (size_t i = 0; i < loopNode->ninputs(); ++i)
    {
      auto in = loopNode->input(i);
      auto arg = in->arguments.begin().ptr();
      auto user = GetUser(arg);
      // only do this for proper loop variables
      if (auto mux = TryGetOwnerOp<mux_op>(*user))
      {
        if (!mux->loop)
        {
          // stream
          continue;
        }
      }
      else if (TryGetOwnerOp<loop_constant_buffer_op>(*user))
      {
      }
      else
      {
        continue;
      }
      PlaceBuffer(in->origin(), 2, false);
    }
  }
}

void
AddBuffers(rvsdg::Region * region)
{
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      auto loop = dynamic_cast<loop_node *>(node);
      JLM_ASSERT(loop);
      OptimizeLoop(loop);
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        AddBuffers(structnode->subregion(n));
      }
    }
    else if (auto simple = dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
    {
      if (jlm::rvsdg::is<buffer_op>(node))
      {
        OptimizeBuffer(simple);
      }
      else if (jlm::rvsdg::is<fork_op>(node))
      {
        //        OptimizeFork(simple);
      }
      else if (jlm::rvsdg::is<branch_op>(node))
      {
        //        OptimizeBranch(simple);
      }
      else if (jlm::rvsdg::is<state_gate_op>(node))
      {
        //        OptimizeStateGate(simple);
      }
      else if (jlm::rvsdg::is<addr_queue_op>(node))
      {
        OptimizeAddrQ(simple);
      }
    }
  }
}

size_t MemoryLatency = 10;

constexpr uint32_t
round_up_pow2(uint32_t x)
{
  if (x == 0)
    return 1;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

void
MaximizeBuffers(rvsdg::Region * region)
{
  //  const size_t capacity = 256;
  std::vector<jlm::rvsdg::SimpleNode *> nodes;
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
      auto loop = dynamic_cast<loop_node *>(node);
      JLM_ASSERT(loop);
      for (size_t n = 0; n < structnode->nsubregions(); n++)
      {
        MaximizeBuffers(structnode->subregion(n));
      }
    }
    else if (auto sn = dynamic_cast<jlm::rvsdg::SimpleNode *>(node))
    {
      if (dynamic_cast<const buffer_op *>(&node->GetOperation()))
      {
        nodes.push_back(sn);
      }
      else if (dynamic_cast<const decoupled_load_op *>(&node->GetOperation()))
      {
        nodes.push_back(sn);
      }
    }
  }
  for (auto node : nodes)
  {
    //    if (auto buf = dynamic_cast<const buffer_op *>(&node->GetOperation()))
    //    {
    //      divert_users(node, buffer_op::create(*node->input(0)->origin(), capacity,
    //      buf->pass_through)); remove(node);
    //    }
    //    else
    if (auto dl = dynamic_cast<const decoupled_load_op *>(&node->GetOperation()))
    {
      auto capacity = round_up_pow2(MemoryLatency);
      if (dl->capacity < capacity)
      {
        divert_users(
            node,
            decoupled_load_op::create(
                *node->input(0)->origin(),
                *node->input(1)->origin(),
                capacity));
        remove(node);
      }
    }
  }
}

std::vector<size_t>
NodeCycles(rvsdg::SimpleNode * node, std::vector<size_t> & input_cycles)
{
  auto max_cycles = *std::max_element(input_cycles.begin(), input_cycles.end());
  if (auto op = dynamic_cast<const llvm::fpbin_op *>(&node->GetOperation()))
  {
    if (op->fpop() == llvm::fpop::add)
    {
      return { max_cycles + 1 };
    }
    if (op->fpop() == llvm::fpop::mul)
    {
      return { max_cycles + 1 };
    }
  }
  else if (auto op = dynamic_cast<const buffer_op *>(&node->GetOperation()))
  {
    if (op->pass_through)
    {
      return { max_cycles + 0 };
    }
    return { max_cycles + 1 };
  }
  else if (dynamic_cast<const addr_queue_op *>(&node->GetOperation()))
  {
    return { input_cycles[0] };
  }
  else if (dynamic_cast<const decoupled_load_op *>(&node->GetOperation()))
  {
    return { max_cycles + MemoryLatency, 0 };
  }
  else if (rvsdg::is<state_gate_op>(node))
  {
    // handle special state gate that sits on dec_load response
    auto sg0_user = GetUser(node->output(0));
    if (TryGetOwnerOp<decoupled_load_op>(*sg0_user) && sg0_user->index() == 1)
    {
      JLM_ASSERT(max_cycles == 0);
      return { 0, MemoryLatency };
    }
  }
  else if (rvsdg::is<store_op>(node))
  {
    JLM_ASSERT(node->noutputs() == 3);
    return { max_cycles + MemoryLatency, 0, 0 };
  }
  return std::vector<size_t>(node->noutputs(), max_cycles);
}

const size_t UnlimitedBufferCapacity = std::numeric_limits<uint32_t>::max();

std::vector<size_t>
NodeCapacity(rvsdg::SimpleNode * node, std::vector<size_t> & input_capacities)
{
  auto min_capacity = *std::min_element(input_capacities.begin(), input_capacities.end());
  if (auto op = dynamic_cast<const llvm::fpbin_op *>(&node->GetOperation()))
  {
    if (op->fpop() == llvm::fpop::add)
    {
      return { min_capacity + 1 };
    }
    if (op->fpop() == llvm::fpop::mul)
    {
      return { min_capacity + 1 };
    }
  }
  else if (auto op = dynamic_cast<const buffer_op *>(&node->GetOperation()))
  {
    return { min_capacity + op->capacity };
  }
  else if (dynamic_cast<const addr_queue_op *>(&node->GetOperation()))
  {
    return { input_capacities[0] };
  }
  else if (auto op = dynamic_cast<const decoupled_load_op *>(&node->GetOperation()))
  {
    return { min_capacity + op->capacity, 0 };
  }
  else if (rvsdg::is<state_gate_op>(node))
  {
    // handle special state gate that sits on dec_load response
    auto sg0_user = GetUser(node->output(0));
    if (TryGetOwnerOp<decoupled_load_op>(*sg0_user) && sg0_user->index() == 1)
    {
      JLM_ASSERT(min_capacity == UnlimitedBufferCapacity);
      return { 0, MemoryLatency };
    }
  }
  else if (dynamic_cast<const store_op *>(&node->GetOperation()))
  {
    return { min_capacity + MemoryLatency, 0, 0 };
  }
  return std::vector<size_t>(node->noutputs(), min_capacity);
}

void
CreateLoopFrontier(
    const loop_node * loop,
    std::unordered_map<rvsdg::output *, size_t> & output_cycles,
    std::unordered_set<rvsdg::input *> & frontier,
    std::unordered_set<backedge_result *> & stream_backedges,
    std::unordered_set<rvsdg::SimpleNode *> & top_muxes)
{
  for (size_t i = 0; i < loop->ninputs(); ++i)
  {
    auto in = loop->input(i);
    auto arg = in->arguments.begin().ptr();

    auto user = GetUser(arg);
    auto userNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*user);
    auto mux = TryGetOwnerOp<mux_op>(*user);
    if ((mux && mux->loop) || TryGetOwnerOp<loop_constant_buffer_op>(*user))
    {
      top_muxes.insert(userNode);
      // we start from these
      auto out = userNode->output(0);
      output_cycles[out] = output_cycles[in->origin()];
      frontier.insert(GetUser(out));
    }
    // these are needed so we can finish with an empty frontier
    output_cycles[arg] = output_cycles[in->origin()];
    frontier.insert(GetUser(arg));
  }
  for (auto & tn : loop->subregion()->TopNodes())
  {
    JLM_ASSERT(is_constant(&tn));
    auto out = tn.output(0);
    output_cycles[out] = 0;
    frontier.insert(GetUser(out));
  }
  for (auto arg : loop->subregion()->Arguments())
  {
    auto backedge = dynamic_cast<backedge_argument *>(arg);
    if (!backedge)
    {
      continue;
    }
    auto user = GetUser(arg);
    auto mux = TryGetOwnerOp<mux_op>(*user);
    if ((mux && mux->loop))
    {
      continue;
    }
    if (TryGetOwnerOp<buffer_op>(*user))
    {
      auto bufNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*user);
      if (TryGetOwnerOp<predicate_buffer_op>(*GetUser(bufNode->output(0))))
      {
        // skip predicate buffer
        continue;
      }
    }
    // this comes from somewhere inside the loop
    output_cycles[arg] = 0;
    frontier.insert(GetUser(arg));
    stream_backedges.insert(backedge->result());
  }
}

void
CalculateLoopCycleDepth(
    loop_node * loop,
    std::unordered_map<rvsdg::output *, size_t> & output_cycles,
    bool analyze_inner_loop = false);

void
PushCycleFrontier(
    const loop_node * loop,
    std::unordered_map<rvsdg::output *, size_t> & output_cycles,
    std::unordered_set<rvsdg::input *> & frontier,
    std::unordered_set<backedge_result *> & stream_backedges,
    std::unordered_set<rvsdg::SimpleNode *> & top_muxes)
{
  bool changed = false;
  //    uint32_t iteration = 0;
  do
  {
    //        std::stringstream stream;
    //        stream << "CalculateLoopCycleDepth_" << loop << "_" << iteration++
    //               << ".dot";
    //        std::unordered_map<rvsdg::output *, std::string> o_color;
    //        std::unordered_map<rvsdg::input *, std::string> i_color;
    //        for (auto i : frontier)
    //        {
    //          i_color.insert({ i, "red" });
    //        }
    //        dump_dot(&loop->region()->graph()->GetRootRegion(), stream.str(), o_color, i_color);
    //        dot_to_svg(stream.str());

    changed = false;
    for (auto in : frontier)
    {
      if (auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*in))
      {
        bool all_contained = true;
        for (size_t i = 0; i < simpleNode->ninputs(); ++i)
        {
          auto f = frontier.find(simpleNode->input(i));
          if (f == frontier.end())
          {
            all_contained = false;
          }
        }
        if (!all_contained)
          continue;
        // all inputs of node are in frontier - move them forward
        std::vector<size_t> input_cycles;
        for (size_t i = 0; i < simpleNode->ninputs(); ++i)
        {
          input_cycles.push_back(output_cycles[simpleNode->input(i)->origin()]);
          frontier.erase(simpleNode->input(i));
        }
        std::vector<size_t> out_cycles = NodeCycles(simpleNode, input_cycles);

        if (top_muxes.find(simpleNode) != top_muxes.end())
        {
          if (dynamic_cast<const mux_op *>(&simpleNode->GetOperation()))
          {
            // TODO: do this in NodeCycles instead?
            // this works for most cases, but is not ideal if the backedge has an II > 1, and the
            // predicate hasn't
            auto pred_latency = output_cycles[simpleNode->input(0)->origin()];
            auto input_latency = output_cycles[simpleNode->input(1)->origin()];
            auto backedge_latency = output_cycles[simpleNode->input(2)->origin()];
            auto out_latency = backedge_latency - pred_latency + input_latency;
            std::cout << "top_mux " << simpleNode << " pred latency: " << pred_latency
                      << " input latency: " << input_latency
                      << " backedge latency: " << backedge_latency
                      << " out latency: " << out_latency << std::endl;
            output_cycles[simpleNode->output(0)] = out_latency;
          }
          else
          {
            JLM_ASSERT(dynamic_cast<const loop_constant_buffer_op *>(&simpleNode->GetOperation()));
            // don't update output cycles
          }
        }
        else
        {
          for (size_t i = 0; i < simpleNode->noutputs(); ++i)
          {
            auto out = simpleNode->output(i);
            output_cycles[out] = out_cycles[i];
            frontier.insert(GetUser(out));
          }
        }
        changed = true;
        break;
      }
      else if (auto be = dynamic_cast<backedge_result *>(in))
      {
        frontier.erase(in);
        auto out = be->argument();
        if (stream_backedges.find(be) == stream_backedges.end())
        {
          // skip stream backedges
          output_cycles[out] = output_cycles[in->origin()];
          frontier.insert(GetUser(out));
        }
        changed = true;
        break;
      }
      else if (auto rr = dynamic_cast<rvsdg::RegionResult *>(in))
      {
        frontier.erase(in);
        auto out = rr->output();
        JLM_ASSERT(out);
        output_cycles[out] = output_cycles[in->origin()];
        // don't continue frontier out of loop
        changed = true;
        break;
      }
      else
      {
        auto inner_loop = rvsdg::TryGetOwnerNode<loop_node>(*in);
        JLM_ASSERT(inner_loop);
        bool all_contained = true;
        for (size_t i = 0; i < inner_loop->ninputs(); ++i)
        {
          auto f = frontier.find(inner_loop->input(i));
          if (f == frontier.end())
          {
            all_contained = false;
          }
        }
        if (!all_contained)
          continue;
        for (size_t i = 0; i < inner_loop->ninputs(); ++i)
        {
          frontier.erase(inner_loop->input(i));
        }
        // TODO: do we just want the latency of a single iteration here?
        CalculateLoopCycleDepth(inner_loop, output_cycles, true);
        for (size_t i = 0; i < inner_loop->noutputs(); ++i)
        {
          std::cout << "output latency " << i << " " << output_cycles[inner_loop->output(i)]
                    << std::endl;
          frontier.insert(GetUser(inner_loop->output(i)));
        }
        changed = true;
        break;
      }
    }
  } while (changed);
  // TODO: is "changed" even necessary or can we just wait for frontier to be empty?
  if (!frontier.empty())
  {
    std::unordered_map<rvsdg::output *, std::string> o_color;
    std::unordered_map<rvsdg::input *, std::string> i_color;
    for (auto i : frontier)
    {
      i_color.insert({ i, "red" });
    }
    dump_dot(&loop->region()->graph()->GetRootRegion(), "crash.dot", o_color, i_color, {});
  }
  JLM_ASSERT(frontier.empty());
}

void
CalculateLoopCycleDepth(
    loop_node * loop,
    std::unordered_map<rvsdg::output *, size_t> & output_cycles,
    bool analyze_inner_loop)
{
  if (!analyze_inner_loop)
  {
    for (size_t i = 0; i < loop->ninputs(); ++i)
    {
      auto in = loop->input(i);
      output_cycles[in->origin()] = 0;
    }
  }
  std::unordered_set<rvsdg::input *> frontier;
  std::unordered_set<backedge_result *> stream_backedges;
  std::unordered_set<rvsdg::SimpleNode *> top_muxes;
  CreateLoopFrontier(loop, output_cycles, frontier, stream_backedges, top_muxes);
  std::unordered_set<rvsdg::input *> frontier2(frontier);
  std::cout << "CalculateLoopCycleDepth(" << loop << ", " << analyze_inner_loop << ")" << std::endl;
  /* the reason for having two iterations here is a loop value being updated at the end of the loop,
   * for example the nextRow in SPMV. In theory more iterations could be necessary, until things
   * only increase by the iterative intensity.
   * */
  // TODO: should there be more iterations of this? We could iterate until there is no more change
  // in the difference. This would also give us the II
  PushCycleFrontier(loop, output_cycles, frontier, stream_backedges, top_muxes);

  std::unordered_map<rvsdg::output *, std::string> o_color;
  std::unordered_map<rvsdg::input *, std::string> i_color;
  std::unordered_map<rvsdg::output *, std::string> tail_label;
  if (!analyze_inner_loop)
  {
    for (auto i : frontier2)
    {
      i_color.insert({ i, "red" });
    }
    for (auto [o, l] : output_cycles)
    {
      tail_label[o] = std::to_string(l);
    }
    dump_dot(
        &loop->region()->graph()->GetRootRegion(),
        util::strfmt("CalculateLoopCycleDepth_", loop, "_first.dot"),
        o_color,
        i_color,
        tail_label);
  }
  std::cout << "second iteration" << std::endl;
  PushCycleFrontier(loop, output_cycles, frontier2, stream_backedges, top_muxes);
  if (!analyze_inner_loop)
  {
    for (auto [o, l] : output_cycles)
    {
      tail_label[o] = std::to_string(l);
    }
    dump_dot(
        &loop->region()->graph()->GetRootRegion(),
        util::strfmt("CalculateLoopCycleDepth_", loop, "_second.dot"),
        o_color,
        i_color,
        tail_label);
  }
  //  }
}

void
setMemoryLatency(size_t memoryLatency)
{
  MemoryLatency = memoryLatency;
}

const size_t MinimumBufferSize = 2;
const size_t MaximumBufferSize = 512;

size_t
PlaceBufferLoop(rvsdg::output * out, size_t min_capacity, bool passThrough)
{
  // places or re-places a buffer on an output
  // don't place buffers after constants
  JLM_ASSERT(!is_constant(rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*out)));
  auto fork = TryGetOwnerOp<fork_op>(*out);
  JLM_ASSERT(!(fork && fork->IsConstant()));

  auto node = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*out);

  if (rvsdg::is<rvsdg::LambdaOperation>(out->region()->node()))
  {
    // don't place buffers outside loops
    return min_capacity;
  }

  auto arg = dynamic_cast<rvsdg::RegionArgument *>(out);
  if (arg && arg->input())
  {
    return PlaceBufferLoop(arg->input()->origin(), min_capacity, passThrough);
  }

  // push buf above loop_const_buf
  if (TryGetOwnerOp<loop_constant_buffer_op>(*out))
  {
    return std::min(
        PlaceBufferLoop(node->input(0)->origin(), min_capacity, passThrough),
        PlaceBufferLoop(node->input(1)->origin(), min_capacity, passThrough));
  }

  if (auto buf = TryGetOwnerOp<buffer_op>(*out))
  {
    // replace buffer and keep larger size
    passThrough = passThrough && buf->pass_through;
    size_t capacity = round_up_pow2(buf->capacity + min_capacity);
    // if the maximum buffer size is exceeded place a smaller buffer, but pretend a large one was
    // placed, to prevent additional buffers further down
    auto actual_capacity = std::max(MinimumBufferSize, std::min(capacity, MaximumBufferSize));
    auto bufOut = buffer_op::create(*node->input(0)->origin(), actual_capacity, passThrough)[0];
    node->output(0)->divert_users(bufOut);
    JLM_ASSERT(node->IsDead());
    remove(node);
    return capacity;
  }
  else
  {
    // create new buffer
    auto directUser = *out->begin();
    size_t capacity = round_up_pow2(min_capacity);
    // if the maximum buffer size is exceeded place a smaller buffer, but pretend a large one was
    // placed, to prevent additional buffers further down
    auto actual_capacity = std::max(MinimumBufferSize, std::min(capacity, MaximumBufferSize));
    auto newOut = buffer_op::create(*out, actual_capacity, passThrough)[0];
    directUser->divert_to(newOut);
    return capacity;
  }
}

void
AdjustLoopBuffers(
    loop_node * loop,
    std::unordered_map<rvsdg::output *, size_t> & output_cycles,
    std::unordered_map<rvsdg::output *, size_t> & buffer_capacity,
    bool analyze_inner_loop = false)
{
  if (!analyze_inner_loop)
  {
    for (size_t i = 0; i < loop->ninputs(); ++i)
    {
      auto in = loop->input(i);
      buffer_capacity[in->origin()] = 0;
    }
  }
  std::unordered_set<rvsdg::input *> frontier;
  std::unordered_set<backedge_result *> stream_backedges;
  std::unordered_set<rvsdg::SimpleNode *> top_muxes;
  CreateLoopFrontier(loop, buffer_capacity, frontier, stream_backedges, top_muxes);
  // set buffer capacity for constant nodes to max
  for (auto & tn : loop->subregion()->TopNodes())
  {
    auto out = tn.output(0);
    // don't use size_t max, since that is used to signal down below
    buffer_capacity[out] = UnlimitedBufferCapacity;
  }
  // same for inputs - we only care what happens within the loop
  for (size_t i = 0; i < loop->ninputs(); ++i)
  {
    auto in = loop->input(i);
    buffer_capacity[in->origin()] = UnlimitedBufferCapacity;
    buffer_capacity[in->arguments.begin().ptr()] = UnlimitedBufferCapacity;
  }
  // TODO: unlimited buffers for stream backedges, but loose that property if a fork is reached?
  // this might also make the current special case handling of addrqs unnecessary

  std::unordered_map<rvsdg::output *, std::string> o_color;
  std::unordered_map<rvsdg::input *, std::string> i_color;
  std::unordered_map<rvsdg::output *, std::string> tail_label;
  if (!analyze_inner_loop)
  {
    for (auto i : frontier)
    {
      i_color.insert({ i, "red" });
    }
    for (auto [o, l] : buffer_capacity)
    {
      tail_label[o] = std::to_string(l);
    }
    dump_dot(
        &loop->region()->graph()->GetRootRegion(),
        util::strfmt("AdjustLoopBuffers_", loop, "_begin.dot"),
        o_color,
        i_color,
        tail_label);
  }

  bool changed = false;
  //  uint32_t iteration = 0;
  do
  {
    //    std::stringstream stream;
    //    stream << "AdjustLoopBuffers_" << << loop << "_" << << iteration++
    //           << ".dot";
    //    std::unordered_map<rvsdg::output *, std::string> o_color;
    //    std::unordered_map<rvsdg::input *, std::string> i_color;
    //    for (auto i : frontier)
    //    {
    //      i_color.insert({ i, "red" });
    //    }
    //    dump_dot(&loop->region()->graph()->GetRootRegion(), stream.str(), o_color, i_color);
    //    dot_to_svg(stream.str());

    changed = false;
    for (auto in : frontier)
    {
      if (auto simpleNode = rvsdg::TryGetOwnerNode<rvsdg::SimpleNode>(*in))
      {
        bool all_contained = true;
        for (size_t i = 0; i < simpleNode->ninputs(); ++i)
        {
          auto f = frontier.find(simpleNode->input(i));
          if (f == frontier.end())
          {
            all_contained = false;
          }
        }
        if (!all_contained)
          continue;
        // all inputs of node are in frontier - move them forward
        size_t max_cycles = 0;
        for (size_t i = 0; i < simpleNode->ninputs(); ++i)
        {
          max_cycles = std::max(max_cycles, output_cycles[simpleNode->input(i)->origin()]);
          frontier.erase(simpleNode->input(i));
        }

        std::vector<size_t> input_capacities;
        // adjust capacities
        for (size_t i = 0; i < simpleNode->ninputs(); ++i)
        {
          auto capacity = buffer_capacity[simpleNode->input(i)->origin()];
          if (!analyze_inner_loop && (!rvsdg::is<addr_queue_op>(simpleNode))
              && capacity < max_cycles)
          {
            size_t capacity_diff = max_cycles - capacity;
            capacity += PlaceBufferLoop(simpleNode->input(i)->origin(), capacity_diff, true);
            buffer_capacity[simpleNode->input(i)->origin()] = capacity;
          }
          input_capacities.push_back(capacity);
        }

        if (top_muxes.find(simpleNode) != top_muxes.end())
        {
          // we reached our starting point again
          auto mux = dynamic_cast<const mux_op *>(&simpleNode->GetOperation());
          if (mux)
          {
            std::cout << "top_mux " << simpleNode
                      << " pred capacity: " << buffer_capacity[simpleNode->input(0)->origin()]
                      << " backedge capacity: " << buffer_capacity[simpleNode->input(2)->origin()]
                      << std::endl;
          }
        }
        else
        {
          std::vector<size_t> out_capacities = NodeCapacity(simpleNode, input_capacities);
          for (size_t i = 0; i < simpleNode->noutputs(); ++i)
          {
            auto out = simpleNode->output(i);
            buffer_capacity[out] = out_capacities[i];
            JLM_ASSERT(analyze_inner_loop || buffer_capacity[out] >= output_cycles[out]);
            frontier.insert(GetUser(out));
          }
        }
        changed = true;
        break;
      }
      else if (auto be = dynamic_cast<backedge_result *>(in))
      {
        frontier.erase(in);
        auto out = be->argument();
        buffer_capacity[out] = buffer_capacity[in->origin()];
        if (stream_backedges.find(be) == stream_backedges.end())
        {
          frontier.insert(GetUser(out));
        }
        changed = true;
        break;
      }
      else if (auto rr = dynamic_cast<rvsdg::RegionResult *>(in))
      {
        frontier.erase(in);
        auto out = rr->output();
        JLM_ASSERT(out);
        buffer_capacity[out] = buffer_capacity[in->origin()];
        // don't continue frontier out of loop
        changed = true;
        break;
      }
      else
      {
        auto inner_loop = rvsdg::TryGetOwnerNode<loop_node>(*in);
        JLM_ASSERT(inner_loop);
        bool all_contained = true;
        for (size_t i = 0; i < inner_loop->ninputs(); ++i)
        {
          auto f = frontier.find(inner_loop->input(i));
          if (f == frontier.end())
          {
            all_contained = false;
          }
        }
        if (!all_contained)
          continue;
        // all inputs of node are in frontier - move them forward
        size_t max_cycles = 0;
        for (size_t i = 0; i < inner_loop->ninputs(); ++i)
        {
          max_cycles = std::max(max_cycles, output_cycles[inner_loop->input(i)->origin()]);
          frontier.erase(inner_loop->input(i));
        }
        //        dump_dot(&loop->region()->graph()->GetRootRegion(), "crash.dot");
        // adjust capacities
        for (size_t i = 0; i < inner_loop->ninputs(); ++i)
        {
          auto capacity = buffer_capacity[inner_loop->input(i)->origin()];
          if (!analyze_inner_loop && capacity < max_cycles)
          {
            auto user = GetUser(inner_loop->input(i)->arguments.begin().ptr());
            auto mux = TryGetOwnerOp<mux_op>(*user);
            if ((mux && mux->loop) || TryGetOwnerOp<loop_constant_buffer_op>(*user))
            {
              size_t capacity_diff = max_cycles - capacity;
              capacity += PlaceBufferLoop(inner_loop->input(i)->origin(), capacity_diff, true);
              buffer_capacity[inner_loop->input(i)->origin()] = capacity;
            }
            else
            {
              // don't put buffers on decouples, streams, and addrq stuff
            }
          }
        }

        AdjustLoopBuffers(inner_loop, output_cycles, buffer_capacity, true);
        for (size_t i = 0; i < inner_loop->noutputs(); ++i)
        {
          frontier.insert(GetUser(inner_loop->output(i)));
        }
        changed = true;
        break;
      }
    }
  } while (changed);
  // TODO: is "changed" even necessary or can we just wait for frontier to be empty?
  // benefit of it is we won't get infinite loop in case something violates this
  JLM_ASSERT(frontier.empty());
  // TODO: take iterative intensity into account. E.g. we can use half the buffer capacity if II is
  // 2
  // TODO: remove buffers on cycles?
  // TODO: don't place buf2 on const buf if it gos up to another const buf - even through a branch?
  // TODO: still run buffer resize pass, but without upsizing?

  if (!analyze_inner_loop)
  {
    for (auto [o, l] : buffer_capacity)
    {
      tail_label[o] = std::to_string(l);
    }
    dump_dot(
        &loop->region()->graph()->GetRootRegion(),
        util::strfmt("AdjustLoopBuffers_", loop, "_end.dot"),
        o_color,
        i_color,
        tail_label);
  }
}

void
CalculateLoopDepths(rvsdg::Region * region)
{
  for (auto node : rvsdg::TopDownTraverser(region))
  {
    if (auto loop = dynamic_cast<loop_node *>(node))
    {
      // process inner loops first
      CalculateLoopDepths(loop->subregion());
      std::unordered_map<rvsdg::output *, size_t> output_cycles;
      CalculateLoopCycleDepth(loop, output_cycles);
      std::unordered_map<rvsdg::output *, size_t> buffer_capacity;
      AdjustLoopBuffers(loop, output_cycles, buffer_capacity);
    }
  }
}

void
add_buffers(llvm::RvsdgModule & rm, bool pass_through)
{
  auto & graph = rm.Rvsdg();
  auto root = &graph.GetRootRegion();
  auto lambda = dynamic_cast<rvsdg::LambdaNode *>(root->Nodes().begin().ptr());
  AddBuffers(lambda->subregion());
  MaximizeBuffers(lambda->subregion());
  CalculateLoopDepths(lambda->subregion());
}

}
