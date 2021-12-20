/*
 * Copyright 2021 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/ir/operators.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/opt/alias-analyses/encoders.hpp>
#include <jlm/opt/alias-analyses/pointsto-graph.hpp>

#include <jive/rvsdg/traverser.hpp>

namespace jlm {
namespace aa {

MemoryStateEncoder::~MemoryStateEncoder() = default;

void
MemoryStateEncoder::Encode(const jive::simple_node & node)
{
	static std::unordered_map<
		std::type_index
	, std::function<void(MemoryStateEncoder&, const jive::simple_node&)>
	> nodes({
	  {typeid(alloca_op), [](auto & mse, auto & node){ mse.EncodeAlloca(node); }}
	, {typeid(malloc_op), [](auto & mse, auto & node){ mse.EncodeMalloc(node); }}
	, {typeid(load_op),   [](auto & mse, auto & node){ mse.EncodeLoad(node);   }}
	, {typeid(store_op),  [](auto & mse, auto & node){ mse.EncodeStore(node);  }}
	, {typeid(call_op),   [](auto & mse, auto & node){ mse.EncodeCall(node);   }}
	, {typeid(free_op),   [](auto & mse, auto & node){ mse.EncodeFree(node);   }}
	, {typeid(Memcpy),    [](auto & mse, auto & node){ mse.EncodeMemcpy(node); }}
	});

	auto & op = node.operation();
	if (nodes.find(typeid(op)) == nodes.end())
		return;

	nodes[typeid(op)](*this, node);
}

void
MemoryStateEncoder::Encode(jive::structural_node & node)
{
	auto encodeLambda = [](auto & mse, auto & n){mse.Encode(*static_cast<lambda::node*>(&n));     };
	auto encodeDelta  = [](auto & mse, auto & n){mse.Encode(*static_cast<delta::node*>(&n));      };
	auto encodePhi    = [](auto & mse, auto & n){mse.Encode(*static_cast<jive::phi::node*>(&n));  };
	auto encodeGamma  = [](auto & mse, auto & n){mse.Encode(*static_cast<jive::gamma_node*>(&n)); };
	auto encodeTheta  = [](auto & mse, auto & n){mse.Encode(*static_cast<jive::theta_node*>(&n)); };

	static std::unordered_map<
		std::type_index,
		std::function<void(MemoryStateEncoder&, jive::structural_node&)>
	> nodes({
	  {typeid(lambda::operation),    encodeLambda }
	, {typeid(delta::operation),     encodeDelta  }
	, {typeid(jive::phi::operation), encodePhi    }
	, {typeid(jive::gamma_op),       encodeGamma  }
	, {typeid(jive::theta_op),       encodeTheta  }
	});

	auto & op = node.operation();
	JLM_ASSERT(nodes.find(typeid(op)) != nodes.end());
	nodes[typeid(op)](*this, node);
}

void
MemoryStateEncoder::Encode(jive::region & region)
{
	using namespace jive;

	topdown_traverser traverser(&region);
	for (auto & node : traverser) {
		if (auto simpnode = dynamic_cast<const simple_node*>(node)) {
			Encode(*simpnode);
			continue;
		}

		JLM_ASSERT(is<structural_op>(node));
		auto structnode = static_cast<structural_node*>(node);
		Encode(*structnode);
	}
}

}}
