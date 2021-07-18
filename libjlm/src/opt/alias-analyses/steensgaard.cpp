/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/rvsdg/graph.hpp>
#include <jive/rvsdg/node.hpp>
#include <jive/rvsdg/structural-node.hpp>
#include <jive/rvsdg/traverser.hpp>

#include <jlm/ir/rvsdg-module.hpp>
#include <jlm/ir/types.hpp>
#include <jlm/ir/operators.hpp>
#include <jlm/opt/alias-analyses/pointsto-graph.hpp>
#include <jlm/opt/alias-analyses/steensgaard.hpp>
#include <jlm/util/strfmt.hpp>

/*
	FIXME: to be removed again
*/
#include <iostream>

namespace jlm {
namespace aa {

/**
* FIXME: Some documentation
*/
class location {
public:
	virtual
	~location()
	{}

	constexpr
	location(bool unknown)
	: unknown_(unknown)
	, pointsto_(nullptr)
	{}

	location(const location &) = delete;

	location(location &&) = delete;

	location &
	operator=(const location &) = delete;

	location &
	operator=(location &&) = delete;

	virtual std::string
	debug_string() const noexcept = 0;

	bool
	unknown() const noexcept
	{
		return unknown_;
	}

	location *
	pointsto() const noexcept
	{
		return pointsto_;
	}

	void
	set_pointsto(location * l) noexcept
	{
		JLM_ASSERT(l != nullptr);

		pointsto_ = l;
	}

	void
	set_unknown(bool unknown) noexcept
	{
		unknown_ = unknown;
	}

private:
	bool unknown_;
	location * pointsto_;
};

class regloc final : public location {
public:
	constexpr
	regloc(const jive::output * output, bool unknown = false)
	: location(unknown)
	, output_(output)
	{}

	const jive::output *
	output() const noexcept
	{
		return output_;
	}

	virtual std::string
	debug_string() const noexcept override
	{
		auto node = jive::node_output::node(output_);
		auto index = output_->index();

		if (jive::is<jive::simple_op>(node)) {
			auto nodestr = node->operation().debug_string();
			auto outputstr = output_->type().debug_string();
			return strfmt(nodestr, ":", index, "[" + outputstr + "]");
		}

		if (is<lambda::cvargument>(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":cv:", index);
		}

		if (is<lambda::fctargument>(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":arg:", index);
		}

		if (is<delta::cvargument>(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":cv:", index);
		}

		if (is_gamma_argument(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":arg", index);
		}

		if (is_theta_argument(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":arg", index);
		}

		if (is_theta_output(output_)) {
			auto dbgstr = jive::node_output::node(output_)->operation().debug_string();
			return strfmt(dbgstr, ":out", index);
		}

		if (is_gamma_output(output_)) {
			auto dbgstr = jive::node_output::node(output_)->operation().debug_string();
			return strfmt(dbgstr, ":out", index);
		}

		if (is_import(output_)) {
			auto imp = static_cast<const jive::impport*>(&output_->port());
			return strfmt("imp:", imp->name());
		}

		if (is<jive::phi::rvargument>(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":rvarg", index);
		}

		if (is<jive::phi::cvargument>(output_)) {
			auto dbgstr = output_->region()->node()->operation().debug_string();
			return strfmt(dbgstr, ":cvarg", index);
		}

		return strfmt(jive::node_output::node(output_)->operation().debug_string(), ":", index);
	}

	static std::unique_ptr<location>
	create(const jive::output * output, bool unknown)
	{
		return std::unique_ptr<location>(new regloc(output, unknown));
	}

private:
	const jive::output * output_;
};

/** \brief FIXME: write documentation
*
*/
class memloc final : public location {
public:
	virtual
	~memloc()
	{}

	constexpr
	memloc(const jive::node * node)
	: location(false)
	, node_(node)
	{}

	const jive::node *
	node() const noexcept
	{
		return node_;
	}

	virtual std::string
	debug_string() const noexcept override
	{
		return node_->operation().debug_string();
	}

	static std::unique_ptr<location>
	create(const jive::node * node)
	{
		return std::unique_ptr<location>(new memloc(node));
	}

private:
	const jive::node * node_;
};

/** \brief FIXME: write documentation
*
* FIXME: This class should be derived from a meloc, but we do not
* have a node to hand in.
*/
class imploc final : public location {
public:
	virtual
	~imploc()
	{}

	imploc(
		const jive::argument * argument,
		bool pointsToUnknown)
	: location(pointsToUnknown)
	, argument_(argument)
	{
		JLM_ASSERT(dynamic_cast<const jlm::impport*>(&argument->port()));
	}

	const jive::argument *
	argument() const noexcept
	{
		return argument_;
	}

	virtual std::string
	debug_string() const noexcept override
	{
		return "IMPORT[" + argument_->debug_string() + "]";
	}

	static std::unique_ptr<location>
	create(const jive::argument * argument)
	{
		JLM_ASSERT(is<ptrtype>(argument->type()));
		auto ptr = static_cast<const ptrtype*>(&argument->type());

		bool pointsToUnknown = is<ptrtype>(ptr->pointee_type());
		return std::unique_ptr<location>(new imploc(argument, pointsToUnknown));
	}
private:
	const jive::argument * argument_;
};

/** \brief FIXME: write documentation
*/
class DummyLocation final : public location {
public:
	virtual
	~DummyLocation()
	{}

	DummyLocation()
	: location(false)
	{}

	virtual std::string
	debug_string() const noexcept override
	{
		return "UNNAMED";
	}

	static std::unique_ptr<location>
	create()
	{
		return std::make_unique<DummyLocation>();
	}
};

/* locationmap class */

locationset::~locationset() = default;

locationset::locationset() = default;

void
locationset::clear()
{
	map_.clear();
	djset_.clear();
	locations_.clear();
}

jlm::aa::location *
locationset::insert(const jive::output * output, bool unknown)
{
	JLM_ASSERT(lookup(output) == nullptr);

	locations_.push_back(regloc::create(output, unknown));
	auto location = locations_.back().get();

	map_[output] = location;
	djset_.insert(location);

	return location;
}

jlm::aa::location *
locationset::insert(const jive::node * node)
{
	locations_.push_back(memloc::create(node));
	auto location = locations_.back().get();
	djset_.insert(location);

	return location;
}

location *
locationset::insertDummy()
{
	locations_.push_back(DummyLocation::create());
	auto location = locations_.back().get();
	djset_.insert(location);

	return location;
}

location *
locationset::insert(const jive::argument * argument)
{
	locations_.push_back(imploc::create(argument));
	auto location = locations_.back().get();
	djset_.insert(location);

	return location;
}

jlm::aa::location *
locationset::lookup(const jive::output * output)
{
	auto it = map_.find(output);
	return it == map_.end() ? nullptr : it->second;
}

jlm::aa::location *
locationset::find_or_insert(const jive::output * output)
{
	if (auto location = lookup(output))
		return find(location);

	return insert(output, false);
}

jlm::aa::location *
locationset::find(jlm::aa::location * l) const
{
	return djset_.find(l)->value();
}


jlm::aa::location *
locationset::find(const jive::output * output)
{
	auto loc = lookup(output);
	JLM_ASSERT(loc != nullptr);

	return djset_.find(loc)->value();
}

jlm::aa::location *
locationset::merge(jlm::aa::location * l1, jlm::aa::location * l2)
{
	return djset_.merge(l1, l2)->value();
}
/*
jlm::aa::location *
locationset::merge(const jive::output * o1, const jive::output * o2)
{
	auto l1 = find_or_insert(o1);
	auto l2 = find_or_insert(o2);
	return djset_.merge((l1), (l2))->value();
}
*/
std::string
locationset::to_dot() const
{
	auto dot_node = [](const locdjset::set & set)
	{
		auto root = set.value();

		std::string label;
		for (auto & l : set) {
			auto unknownstr = l->unknown() ? "{U}" : "";
			auto ptstr = strfmt("{pt:", (intptr_t)l->pointsto(), "}");
			auto locstr = strfmt((intptr_t)l, " : ", l->debug_string());

			if (l == root) {
				label += strfmt("*", locstr, unknownstr, ptstr, "*\\n");
			} else {
				label += strfmt(locstr, "\\n");
			}
		}

		return strfmt("{ ", (intptr_t)&set, " [label = \"", label, "\"]; }");
	};

	//FIXME: This should be const location &
	auto dot_edge = [&](const locdjset::set & set, const locdjset::set & ptset)
	{
		return strfmt((intptr_t)&set, " -> ", (intptr_t)&ptset);
	};

	std::string str;
	str.append("digraph ptg {\n");

	for (auto & set : djset_) {
		str += dot_node(set) + "\n";

		auto pt = set.value()->pointsto();
		if (pt != nullptr) {
			auto ptset = djset_.find(pt);
			str += dot_edge(set, *ptset) + "\n";
		}
	}

	str.append("}\n");

	return str;
}

/* steensgaard class */

Steensgaard::~Steensgaard() = default;

void
Steensgaard::join(location & x, location & y)
{
	std::function<location*(location*, location*)>
	join = [&](location * x, location * y)
	{
		if (x == nullptr)
			return y;

		if (y == nullptr)
			return x;

		if (x == y)
			return x;

		auto rootx = lset_.find(x);
		auto rooty = lset_.find(y);
		rootx->set_unknown(rootx->unknown() | rooty->unknown());
		rooty->set_unknown(rootx->unknown() | rooty->unknown());
		auto tmp = lset_.merge(rootx, rooty);

		if (auto root = join(rootx->pointsto(), rooty->pointsto()))
			tmp->set_pointsto(root);

		return tmp;
	};

	join(&x, &y);
}

void
Steensgaard::Analyze(const jive::simple_node & node)
{
	static std::unordered_map<
		std::type_index
	, std::function<void(Steensgaard&, const jive::simple_node&)>> nodes
	({
	  {typeid(alloca_op),            [](auto & stg, auto & node){ stg.AnalyzeAlloca(node);        }}
	, {typeid(malloc_op),            [](auto & stg, auto & node){ stg.AnalyzeMalloc(node);        }}
	, {typeid(load_op),              [](auto & stg, auto & node){ stg.AnalyzeLoad(node);          }}
	, {typeid(store_op),             [](auto & stg, auto & node){ stg.AnalyzeStore(node);         }}
	, {typeid(call_op),              [](auto & stg, auto & node){ stg.AnalyzeCall(node);          }}
	, {typeid(getelementptr_op),     [](auto & stg, auto & node){ stg.AnalyzeGep(node);           }}
	, {typeid(bitcast_op),           [](auto & stg, auto & node){ stg.AnalyzeBitcast(node);       }}
	, {typeid(bits2ptr_op),          [](auto & stg, auto & node){ stg.AnalyzeBits2ptr(node);      }}
	, {typeid(ptr_constant_null_op), [](auto & stg, auto & node){ stg.AnalyzeNull(node);          }}
	, {typeid(undef_constant_op),    [](auto & stg, auto & node){ stg.AnalyzeUndef(node);         }}
	, {typeid(ConstantArray),        [](auto & stg, auto & node){ stg.AnalyzeConstantArray(node); }}
	, {typeid(Memcpy),               [](auto & stg, auto & node){ stg.AnalyzeMemcpy(node);        }}
	});

	auto & op = node.operation();
	if (nodes.find(typeid(op)) != nodes.end()) {
		nodes[typeid(op)](*this, node);
		return;
	}

	/*
		Ensure that we really took care of all pointer-producing instructions
	*/
	for (size_t n = 0; n < node.noutputs(); n++) {
		if (jive::is<ptrtype>(node.output(n)->type()))
			JLM_UNREACHABLE("We should have never reached this statement.");
	}
}

void
Steensgaard::AnalyzeAlloca(const jive::simple_node & node)
{
	JLM_ASSERT(jive::is<alloca_op>(&node));

	auto ptr = lset_.find_or_insert(node.output(0));
	ptr->set_pointsto(lset_.insert(&node));
}

void
Steensgaard::AnalyzeMalloc(const jive::simple_node & node)
{
	JLM_ASSERT(jive::is<malloc_op>(&node));

	auto ptr = lset_.find_or_insert(node.output(0));
	ptr->set_pointsto(lset_.insert(&node));
}

void
Steensgaard::AnalyzeLoad(const jive::simple_node & node)
{
	JLM_ASSERT(jive::is<load_op>(&node));

	if (!dynamic_cast<const ptrtype*>(&node.output(0)->type()))
		return;

	auto address = lset_.find(node.input(0)->origin());
	auto result = lset_.insert(node.output(0), address->unknown());

	if (address->pointsto() == nullptr) {
		address->set_pointsto(result);
		return;
	}

	join(*result, *address->pointsto());
}

void
Steensgaard::AnalyzeStore(const jive::simple_node & node)
{
	JLM_ASSERT(jive::is<store_op>(&node));

	if (!dynamic_cast<const ptrtype*>(&node.input(1)->origin()->type()))
		return;

	auto address = lset_.find_or_insert(node.input(0)->origin());
	auto value = lset_.find_or_insert(node.input(1)->origin());

	if (address->pointsto() == nullptr) {
		address->set_pointsto(value);
		return;
	}

	join(*address->pointsto(), *value);
}

void
Steensgaard::AnalyzeCall(const jive::simple_node & node)
{
	JLM_ASSERT(jive::is<call_op>(&node));

	auto handle_direct_call = [&](const jive::simple_node & call, const lambda::node & lambda)
	{
		/*
			FIXME: What about varargs
		*/

		/* handle call node arguments */
		auto lambdaarg = lambda.fctarguments().begin();
		JLM_ASSERT(lambda.nfctarguments() == call.ninputs()-1);
		for (size_t n = 1; n < call.ninputs(); n++, lambdaarg++) {
			auto callarg = call.input(n)->origin();
			if (!jive::is<ptrtype>(callarg->type()))
				continue;

			auto callargloc = lset_.find_or_insert(callarg);
			auto lambdaargloc = lset_.find_or_insert(lambdaarg.value());
			join(*callargloc, *lambdaargloc);
		}

		/* handle call node results */
		auto subregion = lambda.subregion();
		JLM_ASSERT(subregion->nresults() == node.noutputs());
		for (size_t n = 0; n < call.noutputs(); n++) {
			auto callres = call.output(n);
			if (!jive::is<ptrtype>(callres->type()))
				continue;

			auto callresloc = lset_.find_or_insert(callres);
			auto lambdaresloc = lset_.find_or_insert(subregion->result(n)->origin());
			join(*callresloc, *lambdaresloc);
		}
	};

	auto handle_indirect_call = [&](const jive::simple_node & call)
	{
		/*
			Nothing can be done for the call/lambda arguments, as it is
			an indirect call and the lambda node cannot be retrieved.
		*/

		/* handle call node results */
		for (size_t n = 0; n < call.noutputs(); n++) {
			auto callres = call.output(n);
			if (!jive::is<ptrtype>(callres->type()))
				continue;

			lset_.insert(callres, true);
		}
	};

	if (auto lambda = is_direct_call(node)) {
		handle_direct_call(node, *lambda);
		return;
	}

	handle_indirect_call(node);
}

void
Steensgaard::AnalyzeGep(const jive::simple_node & node)
{
	JLM_ASSERT(jive::is<getelementptr_op>(&node));

	auto base = lset_.find_or_insert(node.input(0)->origin());
	auto value = lset_.find_or_insert(node.output(0));

	join(*base, *value);
}

void
Steensgaard::AnalyzeBitcast(const jive::simple_node & node)
{
	JLM_ASSERT(jive::is<bitcast_op>(&node));

	auto operand = lset_.find_or_insert(node.input(0)->origin());
	auto result = lset_.find_or_insert(node.output(0));

	join(*operand, *result);
}

void
Steensgaard::AnalyzeBits2ptr(const jive::simple_node & node)
{
	JLM_ASSERT(jive::is<bits2ptr_op>(&node));

	lset_.insert(node.output(0), true);
}

void
Steensgaard::AnalyzeNull(const jive::simple_node & node)
{
	JLM_ASSERT(jive::is<ptr_constant_null_op>(&node));

	/*
		FIXME: This should not point to unknown, but to a NULL memory location.
	*/
	lset_.insert(node.output(0), true);
}

void
Steensgaard::AnalyzeUndef(const jive::simple_node & node)
{
	JLM_ASSERT(is<undef_constant_op>(&node));
	auto output = node.output(0);

	if (is<ptrtype>(output->type()) == false)
		return;

	/*
		FIXME: Overthink whether it is correct that undef points to unknown.
	*/
	lset_.insert(node.output(0), true);
}

void
Steensgaard::AnalyzeConstantArray(const jive::simple_node & node)
{
	JLM_ASSERT(is<ConstantArray>(&node));
	auto & op = *static_cast<const ConstantArray*>(&node.operation());

	if (is<ptrtype>(op.type()) == false)
		return;

	auto result = lset_.insert(node.output(0), false);
	for (size_t n = 0; n < node.ninputs(); n++)	{
		auto operand = lset_.find(node.input(n)->origin());
		join(*operand, *result);
	}
}

void
Steensgaard::AnalyzeMemcpy(const jive::simple_node & node)
{
	JLM_ASSERT(is<Memcpy>(&node));

	/*
		FIXME: handle unknown
	*/

	/*
		FIXME: write some documentation about the implementation
	*/

	auto dstAddress = lset_.find(node.input(0)->origin());
	auto srcAddress = lset_.find(node.input(1)->origin());

	if (srcAddress->pointsto() == nullptr) {
		/*
			If we do not know where the source address points to yet(!),
			insert a dummy location so we have something to work with.
		*/
		auto dummyLocation = lset_.insertDummy();
		srcAddress->set_pointsto(dummyLocation);
	}

	if (dstAddress->pointsto() == nullptr) {
		/*
			If we do not know where the destination address points to yet(!),
			insert a dummy location so we have somehting to work with.
		*/
		auto dummyLocation = lset_.insertDummy();
		dstAddress->set_pointsto(dummyLocation);
	}

	auto srcMemory = lset_.find(srcAddress->pointsto());
	auto dstMemory = lset_.find(dstAddress->pointsto());

	if (srcMemory->pointsto() == nullptr) {
		auto dummyLocation = lset_.insertDummy();
		srcMemory->set_pointsto(dummyLocation);
	}

	if (dstMemory->pointsto() == nullptr) {
		auto dummyLocation = lset_.insertDummy();
		dstMemory->set_pointsto(dummyLocation);
	}

	join(*srcMemory->pointsto(), *dstMemory->pointsto());
}

void
Steensgaard::Analyze(const lambda::node & lambda)
{
	if (lambda.direct_calls()) {
		/* handle context variables */
		for (auto & cv : lambda.ctxvars()) {
			if (!jive::is<ptrtype>(cv.type()))
				continue;

			auto origin = lset_.find_or_insert(cv.origin());
			auto argument = lset_.insert(cv.argument(), false);
			join(*origin, *argument);
		}

		/* handle function arguments */
		for (auto & argument : lambda.fctarguments()) {
			if (!jive::is<ptrtype>(argument.type()))
				continue;

			lset_.insert(&argument, false);
		}

		Analyze(*lambda.subregion());

		auto ptr = lset_.find_or_insert(lambda.output());
		ptr->set_pointsto(lset_.insert(&lambda));
	} else {
		/* handle context variables */
		for (auto & cv : lambda.ctxvars()) {
			if (!jive::is<ptrtype>(cv.type()))
				continue;

			auto origin = lset_.find_or_insert(cv.origin());
			auto argument = lset_.insert(cv.argument(), false);
			join(*origin, *argument);
		}

		/* handle function arguments */
		for (auto & argument : lambda.fctarguments()) {
			if (!jive::is<ptrtype>(argument.type()))
				continue;

			lset_.insert(&argument, true);
		}

		Analyze(*lambda.subregion());

		auto ptr = lset_.find_or_insert(lambda.output());
		ptr->set_pointsto(lset_.insert(&lambda));
	}
}

void
Steensgaard::Analyze(const delta::node & delta)
{
	/* handle context variables */
	for (auto & input : delta.ctxvars()) {
		if (!jive::is<ptrtype>(input.type()))
			continue;

		auto origin = lset_.find(input.origin());
		auto argument = lset_.insert(input.arguments.first(), false);
		join(*origin, *argument);
	}

	Analyze(*delta.subregion());

	auto deltaptr = lset_.find_or_insert(delta.output());
	auto deltaloc = lset_.insert(&delta);
	deltaptr->set_pointsto(deltaloc);

	if (auto loc = lset_.lookup(delta.result()->origin()))
		join(*loc, *deltaloc);
}

void
Steensgaard::Analyze(const jive::phi::node & phi)
{
	/* handle context variables */
	for (auto cv = phi.begin_cv(); cv != phi.end_cv(); cv++) {
		if (is<ptrtype>(cv->type()) == false)
			continue;

		auto origin = lset_.find(cv->origin());
		auto argument = lset_.insert(cv->argument(), false);
		join(*origin, *argument);
	}

	/* handle recursion variable arguments */
	for (auto rv = phi.begin_rv(); rv != phi.end_rv(); rv++) {
		if (is<ptrtype>(rv->type()) == false)
			continue;

		lset_.insert(rv->argument(), false);
	}

	Analyze(*phi.subregion());

	/* handle recursion variable outputs */
	for (auto rv = phi.begin_rv(); rv != phi.end_rv(); rv++) {
		if (is<ptrtype>(rv->type()) == false)
			continue;

		auto origin = lset_.find(rv->result()->origin());
		auto argument = lset_.find(rv->argument());
		join(*origin, *argument);

		auto output = lset_.insert(rv.output(), false);
		join(*argument, *output);
	}
}

void
Steensgaard::Analyze(const jive::gamma_node & node)
{
	/* handle entry variables */
	for (auto ev = node.begin_entryvar(); ev != node.end_entryvar(); ev++) {
		if (!jive::is<ptrtype>(ev->type()))
			continue;

		auto originloc = lset_.find(ev->origin());
		for (auto & argument : *ev) {
			auto argumentloc = lset_.insert(&argument, false);
			join(*argumentloc, *originloc);
		}
	}

	/* handle subregions */
	for (size_t n = 0; n < node.nsubregions(); n++)
		Analyze(*node.subregion(n));

	/* handle exit variables */
	for (auto ex = node.begin_exitvar(); ex != node.end_exitvar(); ex++) {
		if (!jive::is<ptrtype>(ex->type()))
			continue;

		auto outputloc = lset_.insert(ex.output(), false);
		for (auto & result : *ex) {
			auto resultloc = lset_.find(result.origin());
			join(*outputloc, *resultloc);
		}
	}
}

void
Steensgaard::Analyze(const jive::theta_node & theta)
{
	for (auto lv : theta) {
		if (!jive::is<ptrtype>(lv->type()))
			continue;

		auto originloc = lset_.find(lv->input()->origin());
		auto argumentloc = lset_.insert(lv->argument(), false);

		join(*argumentloc, *originloc);
	}

	Analyze(*theta.subregion());

	for (auto lv : theta) {
		if (!jive::is<ptrtype>(lv->type()))
			continue;

		auto originloc = lset_.find(lv->result()->origin());
		auto argumentloc = lset_.find(lv->argument());
		auto outputloc = lset_.insert(lv, false);

		join(*originloc, *argumentloc);
		join(*originloc, *outputloc);
	}
}

void
Steensgaard::Analyze(const jive::structural_node & node)
{
	auto analyzeLambda = [](auto& s, auto& n){s.Analyze(*static_cast<const lambda::node*>(&n));    };
	auto analyzeDelta  = [](auto& s, auto& n){s.Analyze(*static_cast<const delta::node*>(&n));     };
	auto analyzeGamma  = [](auto& s, auto& n){s.Analyze(*static_cast<const jive::gamma_node*>(&n));};
	auto analyzeTheta  = [](auto& s, auto& n){s.Analyze(*static_cast<const jive::theta_node*>(&n));};
	auto analyzePhi    = [](auto& s, auto& n){s.Analyze(*static_cast<const jive::phi::node*>(&n)); };

	static std::unordered_map<
		std::type_index
	, std::function<void(Steensgaard&, const jive::structural_node&)>> nodes
	({
	  {typeid(lambda::operation),    analyzeLambda }
	, {typeid(delta::operation),     analyzeDelta  }
	, {typeid(jive::gamma_op),       analyzeGamma  }
	, {typeid(jive::theta_op),       analyzeTheta  }
	, {typeid(jive::phi::operation), analyzePhi    }
	});

	auto & op = node.operation();
	JLM_ASSERT(nodes.find(typeid(op)) != nodes.end());
	nodes[typeid(op)](*this, node);
}

void
Steensgaard::Analyze(jive::region & region)
{
	using namespace jive;

	topdown_traverser traverser(&region);
	for (auto & node : traverser) {
		if (auto smpnode = dynamic_cast<const simple_node*>(node)) {
			Analyze(*smpnode);
			continue;
		}

		JLM_ASSERT(is<structural_op>(node));
		auto structnode = static_cast<const structural_node*>(node);
		Analyze(*structnode);
	}
}

void
Steensgaard::Analyze(const jive::graph & graph)
{
	auto add_imports = [](const jive::graph & graph, locationset & lset)
	{
		auto region = graph.root();
		for (size_t n = 0; n < region->narguments(); n++) {
			auto argument = region->argument(n);
			if (!jive::is<ptrtype>(argument->type()))
				continue;
			/* FIXME: we should not add function imports */
			auto imploc = lset.insert(argument);
			auto ptr = lset.insert(argument, false);
			ptr->set_pointsto(imploc);
		}
	};

	add_imports(graph, lset_);
	Analyze(*graph.root());
}

std::unique_ptr<ptg>
Steensgaard::Analyze(const rvsdg_module & module)
{
	ResetState();

	Analyze(*module.graph());
//	std::cout << lset_.to_dot() << std::flush;
	auto ptg = ConstructPointsToGraph(lset_);
//	std::cout << ptg::to_dot(*ptg) << std::flush;

	return ptg;
}

std::unique_ptr<ptg>
Steensgaard::ConstructPointsToGraph(const locationset & lset) const
{
	auto ptg = ptg::create();

	/*
		Create points-to graph nodes
	*/
	std::vector<ptg::memnode*> memNodes;
	std::unordered_map<location*, ptg::node*> map;
	std::unordered_map<const disjointset<location*>::set*, std::vector<ptg::memnode*>> allocators;
	for (auto & set : lset) {
		for (auto & loc : set) {
			if (auto regloc = dynamic_cast<jlm::aa::regloc*>(loc)) {
				map[loc] = ptg::regnode::create(ptg.get(), regloc->output());
				continue;
			}

			if (auto memloc = dynamic_cast<jlm::aa::memloc*>(loc)) {
				auto node = ptg::allocator::create(ptg.get(), memloc->node());
				allocators[&set].push_back(node);
				memNodes.push_back(node);
				map[loc] = node;
				continue;
			}

			if (auto l = dynamic_cast<imploc*>(loc)) {
				auto node = ptg::impnode::create(ptg.get(), l->argument());
				allocators[&set].push_back(node);
				memNodes.push_back(node);
				map[loc] = node;
				continue;
			}

			if (auto unnamedLocation = dynamic_cast<DummyLocation*>(loc)) {
				continue;
			}

			JLM_UNREACHABLE("Unhandled location node.");
		}
	}

	/*
		Create points-to graph edges
	*/
	for (auto & set : lset) {
		bool pointsToUnknown = lset.set(*set.begin()).value()->unknown();

		for (auto & loc : set) {
			if (dynamic_cast<DummyLocation*>(loc))
				continue;

			if (pointsToUnknown) {
				map[loc]->add_edge(&ptg->memunknown());
			}

			auto pt = set.value()->pointsto();
			if (pt == nullptr)
				continue;

			auto & ptset = lset.set(pt);
			for (auto & allocator : allocators[&ptset])
				map[loc]->add_edge(allocator);
		}
	}

	return ptg;
}

void
Steensgaard::ResetState()
{
	lset_.clear();
}

}}
