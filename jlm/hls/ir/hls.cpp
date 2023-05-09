/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/ir/hls.hpp>

jive::structural_output *jlm::hls::loop_node::add_loopvar(jive::output *origin, jive::output **buffer) {
    auto input = jive::structural_input::create(this, origin, origin->type());
    auto output = jive::structural_output::create(this, origin->type());

    auto argument_in = jive::argument::create(subregion(), input, origin->type());
    auto argument_loop = add_backedge(origin->type());

    auto mux = hls::mux_op::create(*predicate_buffer(), {argument_in, argument_loop}, false, true)[0];
    auto buf = hls::buffer_op::create(*mux, 2)[0];
    auto branch = hls::branch_op::create(*predicate()->origin(), *buf, true);
    if (buffer != nullptr) {
        *buffer = buf;
    }
    jive::result::create(subregion(), branch[0], output, origin->type());
    auto result_loop = argument_loop->result();
    result_loop->divert_to(branch[1]);
    return output;
}

jlm::hls::loop_node *jlm::hls::loop_node::copy(jive::region *region, jive::substitution_map &smap) const {
    auto nf = graph()->node_normal_form(typeid(jive::operation));
    nf->set_mutable(false);

    jive::substitution_map rmap;
    auto loop = create(region, false);

    for (size_t i = 0; i < ninputs(); ++i) {
        auto in_origin = smap.lookup(input(i)->origin());
        auto inp = jive::structural_input::create(loop, in_origin, in_origin->type());
        rmap.insert(input(i), loop->input(i));
        auto oarg = input(i)->arguments.begin().ptr();
        auto narg = jive::argument::create(loop->subregion(), inp, oarg->port());
        rmap.insert(oarg, narg);
    }
    for (size_t i = 0; i < noutputs(); ++i) {
        auto out = jive::structural_output::create(loop, output(i)->type());
        rmap.insert(output(i), out);
        smap.insert(output(i), out);
    }
    for (size_t i = 0; i < subregion()->narguments(); ++i) {
        auto arg = subregion()->argument(i);
        if (auto ba = dynamic_cast<jlm::hls::backedge_argument*>(arg)) {
            auto na = loop->add_backedge(arg->type());
            rmap.insert(ba, na);
        }
    }

    subregion()->copy(loop->subregion(), rmap, false, false);
    loop->_predicate_buffer = rmap.lookup(_predicate_buffer);
    // redirect backedges
    for (size_t i = 0; i < subregion()->narguments(); ++i) {
        auto arg = subregion()->argument(i);
        if (auto ba = dynamic_cast<jlm::hls::backedge_argument*>(arg)) {
            auto na = dynamic_cast<jlm::hls::backedge_argument*>(rmap.lookup(ba));
            na->result()->divert_to(rmap.lookup(ba->result()->origin()));
        }
    }
    for (size_t i = 0; i < noutputs(); ++i) {
        auto outp = output(i);
        auto res = outp->results.begin().ptr();
        auto origin = rmap.lookup(res->origin());
        jive::result::create(loop->subregion(), origin, loop->output(i), res->port());
    }
    nf->set_mutable(true);
    return loop;
}

jlm::hls::backedge_argument *jlm::hls::loop_node::add_backedge(const jive::type &type) {
    auto argument_loop = backedge_argument::create(subregion(), type);
    auto result_loop = backedge_result::create(argument_loop);
    argument_loop->result_ = result_loop;
    result_loop->argument_ = argument_loop;
    return argument_loop;
}

jlm::hls::loop_node *jlm::hls::loop_node::create(jive::region *parent, bool init) {
    auto ln = new loop_node(parent);
    if (init) {
        auto predicate = jive::control_false(ln->subregion());
        auto pred_arg = ln->add_backedge(jive::ctltype(2));
        pred_arg->result()->divert_to(predicate);
        ln->_predicate_buffer = hls::predicate_buffer_op::create(*pred_arg)[0];
    }
    return ln;
}

void jlm::hls::loop_node::set_predicate(jive::output *p) {
    auto node = jive::node_output::node(predicate()->origin());
    predicate()->origin()->divert_users(p);
    if (node && !node->has_users())
        remove(node);
}
