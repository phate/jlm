/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include "hls-function-util.hpp"
#include "jlm/hls/util/view.hpp"
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
    node2->output(0)->divert_users(newOut);
    JLM_ASSERT(node2->IsDead());
    remove(node2);
    JLM_ASSERT(node->IsDead());
    remove(node);
  }
}

void
AddBuffers(rvsdg::Region * region)
{
  for (auto & node : rvsdg::TopDownTraverser(region))
  {
    if (auto structnode = dynamic_cast<rvsdg::StructuralNode *>(node))
    {
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
        OptimizeFork(simple);
      }
      else if (jlm::rvsdg::is<state_gate_op>(node))
      {
        OptimizeStateGate(simple);
      }
      else if (jlm::rvsdg::is<addr_queue_op>(node))
      {
        OptimizeAddrQ(simple);
      }
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
}

}
