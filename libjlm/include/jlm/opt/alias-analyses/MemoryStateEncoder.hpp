/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP
#define JLM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP

#include <memory>

namespace jive {

class gamma_node;
class region;
class simple_node;
class structural_node;
class theta_node;

}

namespace jlm {

class CallNode;
class RvsdgModule;
class StatisticsDescriptor;

namespace delta { class node; }
namespace lambda { class node; }
namespace phi { class node; }

class LoadNode;
class StoreNode;

namespace aa {

class PointsToGraph;

/** \brief Memory State Encoder Interface
 *
 * A memory state encoder encodes a points-to graph in the RVSDG. The basic idea is that there exists a one-to-one
 * correspondence between memory nodes in the points-to graph and memory states in the RVSDG, i.e., for each memory
 * node in the points-to graph, there exists a memory state edge in the RVSDG. A memory state encoder routes these
 * state edges through the RVSDG's structural nodes and ensures that simple nodes operating on a
 * memory location represented by a corresponding memory node in the points-to graph are sequentialized with the
 * respective memory state edge. For example, a store node that modifies a global variable needs to have the respective
 * state edge that corresponds to its memory location routed through it, i.e., the store node
 * is sequentialized by this state edge. Such an encoding ensures that the ordering of side-effecting operations
 * touching on the same memory locations is preserved, while rendering operations independent that are not operating on
 * the same memory locations.
 */
class MemoryStateEncoder {
public:
	virtual
	~MemoryStateEncoder();

	virtual void
	Encode(
    RvsdgModule & module,
    const StatisticsDescriptor & sd) = 0;

	virtual void
	EncodeAlloca(const jive::simple_node & node) = 0;

	virtual void
	EncodeMalloc(const jive::simple_node & node) = 0;

	virtual void
	EncodeLoad(const LoadNode & loadNode) = 0;

	virtual void
	EncodeStore(const StoreNode & storeNode) = 0;

	virtual void
	EncodeFree(const jive::simple_node & node) = 0;

	virtual void
	EncodeCall(const CallNode & callNode) = 0;

	virtual void
	EncodeMemcpy(const jive::simple_node & node) = 0;

	virtual void
	Encode(const lambda::node & lambda) = 0;

	virtual void
	Encode(const phi::node & phi) = 0;

	virtual void
	Encode(const delta::node & delta) = 0;

	virtual void
	Encode(jive::gamma_node & gamma) = 0;

	virtual void
	Encode(jive::theta_node & theta) = 0;

protected:
	void
	Encode(jive::region & region);

	void
	Encode(jive::structural_node & node);

	void
	Encode(const jive::simple_node & node);
};

}}

#endif //JLM_OPT_ALIAS_ANALYSES_MEMORYSTATEENCODER_HPP
