/*
 * Copyright 2025 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <test-operation.hpp>
#include <test-registry.hpp>
#include <test-types.hpp>

#include <jlm/rvsdg/graph.hpp>
#include <jlm/rvsdg/tracker.hpp>

static void
TestTrackingOrder()
{
  using namespace jlm::rvsdg;
  using namespace jlm::tests;

  // Arrange & Act & Assert
  auto valueType = jlm::tests::ValueType::Create();

  Graph rvsdg;
  auto & rootRegion = rvsdg.GetRootRegion();
  auto & i0 = GraphImport::Create(rvsdg, valueType, "i");

  constexpr size_t myTrackerState = 0;
  Tracker tracker(rootRegion, 1);

  auto node0 = TestOperation::create(&rootRegion, {}, { valueType });
  auto node1 = TestOperation::create(&rootRegion, {}, { valueType });
  auto node2 = TestOperation::create(&rootRegion, {}, { valueType });
  auto node3 = TestOperation::create(&rootRegion, { &i0 }, { valueType });
  auto node4 = TestOperation::create(&rootRegion, {}, { valueType });

  // Ensure that the node identifiers have the relative order that we expect. Otherwise, the tests
  // below will fail for sure.
  assert(node0->GetNodeId() < node1->GetNodeId());
  assert(node1->GetNodeId() < node2->GetNodeId());
  assert(node2->GetNodeId() < node3->GetNodeId());
  assert(node3->GetNodeId() < node4->GetNodeId());

  // The nodes have not been added to the tracker yet, so we expect them to not be in any state
  assert(static_cast<size_t>(tracker.get_nodestate(node0)) == tracker_nodestate_none);
  assert(static_cast<size_t>(tracker.get_nodestate(node1)) == tracker_nodestate_none);
  assert(static_cast<size_t>(tracker.get_nodestate(node2)) == tracker_nodestate_none);
  assert(static_cast<size_t>(tracker.get_nodestate(node3)) == tracker_nodestate_none);
  assert(static_cast<size_t>(tracker.get_nodestate(node4)) == tracker_nodestate_none);

  // Adding the nodes to the tracker for myTrackerState
  tracker.set_nodestate(node0, myTrackerState);
  tracker.set_nodestate(node1, myTrackerState);
  tracker.set_nodestate(node2, myTrackerState);
  tracker.set_nodestate(node3, myTrackerState);
  tracker.set_nodestate(node4, myTrackerState);

  // Now we expect them all to be in myTrackerState
  assert(tracker.get_nodestate(node0) == myTrackerState);
  assert(tracker.get_nodestate(node1) == myTrackerState);
  assert(tracker.get_nodestate(node2) == myTrackerState);
  assert(tracker.get_nodestate(node3) == myTrackerState);
  assert(tracker.get_nodestate(node4) == myTrackerState);

  // We expect node0 to always be on top as it is the one with the smallest node identifier
  assert(tracker.peek_top(myTrackerState) == node0);

  // node0 was popped above. Let's add it back and see whether we still get it back as it still has
  // the smallest identifier.
  tracker.set_nodestate(node0, myTrackerState);
  assert(tracker.peek_top(myTrackerState) == node0);

  // The node with the next smallest identifier is node1
  assert(tracker.peek_top(myTrackerState) == node1);

  // Add node1 back, then remove it from the graph to see that the tracker registered the change
  tracker.set_nodestate(node1, myTrackerState);
  assert(tracker.get_nodestate(node1) == myTrackerState);
  remove(node1);
  assert(static_cast<size_t>(tracker.get_nodestate(node1)) == tracker_nodestate_none);

  // All nodes have the same depth, which means the bottom and top nodes should be the same.
  // Thus, the next nodes that is returned should be node2
  assert(tracker.peek_bottom(myTrackerState) == node2);
  assert(static_cast<size_t>(tracker.get_nodestate(node2)) == tracker_nodestate_none);

  // Let's just take all nodes out to see that nothing is tracked any longer
  assert(tracker.peek_top(myTrackerState) == node3);
  assert(tracker.peek_top(myTrackerState) == node4);
  assert(tracker.peek_top(myTrackerState) == nullptr);
  assert(tracker.peek_bottom(myTrackerState) == nullptr);

  // Let's track node3 and node4 again
  tracker.set_nodestate(node3, myTrackerState);
  tracker.set_nodestate(node4, myTrackerState);

  // Let's put node3 below node4 in the graph
  // This means that node4 should be now in the "top depth" node set and node3 in the "bottom depth"
  // node set
  node3->input(0)->divert_to(node4->output(0));

  // We expect node4 to be returned as the next top node (even though it has a greater
  // identifier than node3) as node3 is not in the "top depth" node set any longer
  assert(tracker.peek_top(myTrackerState) == node4);
  tracker.set_nodestate(node4, myTrackerState);

  // We expect node3 to be returned as it is the only one in the "bottom depth" node set
  assert(tracker.peek_bottom(myTrackerState) == node3);
  tracker.set_nodestate(node3, myTrackerState);

  // Divert the input of node3 again
  node3->input(0)->divert_to(&i0);

  // Now, all nodes have the same depth again. This means that the "top depth" and "bottom depth"
  // node set should be the same again. Thus, we expect node3 and then node4 to be returned for both
  // the "top depth" and "bottom depth " set.
  assert(tracker.peek_top(myTrackerState) == node3);
  assert(tracker.peek_top(myTrackerState) == node4);
  assert(tracker.peek_top(myTrackerState) == nullptr);

  // Put node3 and node4 back, and do the same checks for the "bottom depth" node set
  tracker.set_nodestate(node3, myTrackerState);
  tracker.set_nodestate(node4, myTrackerState);
  assert(tracker.peek_bottom(myTrackerState) == node3);
  assert(tracker.peek_bottom(myTrackerState) == node4);
  assert(tracker.peek_bottom(myTrackerState) == nullptr);
}

JLM_UNIT_TEST_REGISTER("jlm/rvsdg/TrackerTests-TestTrackingOrder", TestTrackingOrder)
