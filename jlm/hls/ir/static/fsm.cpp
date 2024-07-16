/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/ir/static/fsm.hpp>
#include <jlm/hls/ir/static/loop.hpp>

#include <jlm/rvsdg/notifiers.hpp>

namespace jlm::static_hls
{

fsm_state *
fsm_state::create(fsm_node_temp * parent_fsm_node, size_t index)
{
    // The region is being created with a pointer to the parent node and an index
    // But the region is not being added to the parent node's subregions list
    auto fs = new fsm_state(parent_fsm_node, index);
    return fs;
};

void
fsm_state::enable_reg(jlm::rvsdg::node* node)
{
    JLM_ASSERT(jlm::rvsdg::is<reg_op>(node->operation()));
    JLM_ASSERT(dynamic_cast<jlm::rvsdg::structural_output*>(node->input(0)->origin()));
    auto fsm_output = static_cast<jlm::rvsdg::structural_output*>(node->input(0)->origin()); 
    
    for (auto &result : fsm_output->results)
    {
        if (result.region() != this) continue;

        auto ctrl_const = jlm::rvsdg::control_constant(this, 2, 1);

        auto old_origin = result.origin();
        result.divert_to(ctrl_const);
        jlm::rvsdg::remove(static_cast<jlm::rvsdg::node_output*>(old_origin)->node());
    }
};

void
fsm_state::add_ctl_result(size_t nalternatives, jlm::rvsdg::structural_output* structural_output)
{
    auto ctrl_const = jlm::rvsdg::control_constant(this, nalternatives, 0);

    //TODO connecting these regions result to a dummy empty structural node like this is not clean and may break asumptions made in the jlm code base
    //! This is just a temporary workaround
    // note: calling create() will add the result to the results list of the structural_output
    jlm::rvsdg::result::create(this, ctrl_const, structural_output, jlm::rvsdg::ctltype(nalternatives));
};

void
fsm_state::set_mux_ctl(jlm::rvsdg::input* result, size_t alternatives)
{
    muxes_ctl_[result] = alternatives;
};

void
fsm_state::apply_mux_ctl()
{
    for (auto &mux_ctl : muxes_ctl_)
    {
        auto result = mux_ctl.first;
        auto mux_output = result->origin();
        JLM_ASSERT(jlm::rvsdg::is<jlm::rvsdg::node_output>(mux_output));

        auto mux = static_cast<jlm::rvsdg::node_output*>(mux_output)->node();
        auto fsm_output = mux->input(0)->origin();
        JLM_ASSERT(jlm::rvsdg::is<jlm::rvsdg::structural_output>(fsm_output));

        for (auto &result: static_cast<jlm::rvsdg::structural_output*>(fsm_output)->results)
        {
            if (result.region() != this) continue;
            
            auto mux_op = dynamic_cast<const jlm::static_hls::mux_op*>(&mux->operation());
            if (!mux_op) JLM_UNREACHABLE("SHLS: Mux operation not found");

            auto ctrl_const = jlm::rvsdg::control_constant(this, mux_op->nalternatives(), mux_ctl.second);
            auto old_origin = result.origin();
            result.divert_to(ctrl_const);
            jlm::rvsdg::remove(static_cast<jlm::rvsdg::node_output*>(old_origin)->node());
        }

        // auto ctrl_const = jlm::rvsdg::control_constant(this, mux_ctl.second, 0);
        // mux_ctl.first->divert_to(ctrl_const);
    }
};

void
fsm_node_temp::print_states() const
{
    std::unordered_map<jlm::rvsdg::region*, size_t> region_to_state;
    for (size_t i=0; i<states_.size(); i++)
    {
        std::cout << "S" << i << ": ";
        auto state = states_[i];

        for (size_t result_id=0; result_id<state->nresults(); result_id++)
        {
            auto result = state->result(result_id);
            auto node_output = dynamic_cast<jlm::rvsdg::node_output*>(*result->output()->begin());
            if (!node_output) continue;

            auto node = node_output->node();

            if (dynamic_cast<const jlm::static_hls::reg_op*>(&node->operation()))
            {
                auto ctl = static_cast<jlm::rvsdg::node_output*>(result->origin())->node();
                if (!jlm::rvsdg::is_ctlconstant_op(ctl->operation())) continue;

                // If the register store is not on skip it
                auto clt_val = static_cast<const jlm::rvsdg::ctlconstant_op&>(ctl->operation()).value().alternative();
                if (clt_val == 0) continue;

                std::cout << "R" << result_id << " ";
                
            }
        }

    }
};

// FIXME: This function is not implemented
fsm_node_temp *
fsm_node_temp::copy(jlm::rvsdg::region * region, jlm::rvsdg::substitution_map & smap) const {
    JLM_UNREACHABLE("SHLS: fsm_node_temp::copy() is not implemented");
    return nullptr;
    // auto ln = new fsm_node_temp(region);
    // return ln;
};

fsm_node_temp::~fsm_node_temp()
{
    std::cout << "*** Deleting fsm_node_temp ***" << std::endl;
    for (auto state : states_)
    {
        delete state;
    }
};

fsm_node_builder::~fsm_node_builder()
{
    std::cout << "*** Deleting fsm_node_builder ***" << std::endl;
    // // states_.clear();
    // for (auto state : states_) {
    //     delete state;
    // }
    // delete fsm_node_temp_;
};

fsm_node_builder *
fsm_node_builder::create(jlm::rvsdg::region * parent)
{
    // FSM should be create in the control region of a loop node
    JLM_ASSERT(jlm::rvsdg::is<jlm::static_hls::loop_op>(parent->node()->operation()));
    JLM_ASSERT(parent->index() == 0);

    auto fn = new fsm_node_builder(parent);
    return fn;
};

jlm::rvsdg::structural_output *
fsm_node_temp::add_ctl_output(size_t nalternatives)
{
    auto structural_output = jlm::rvsdg::structural_output::create(this, jlm::rvsdg::ctltype(nalternatives));
    for (auto state : states_)
    {
        state->add_ctl_result(nalternatives, structural_output);
    }
    std::cout << "Added register output to FSM node with results size " << structural_output->results.size() << ", empty " << structural_output->results.empty() << std::endl;
    return structural_output;
};

// jlm::rvsdg::structural_output *
// fsm_node_temp::add_mux_ouput()
// {
//     auto structural_output = jlm::rvsdg::structural_output::create(this, jlm::rvsdg::ctltype(1));
//     for (auto state : states_)
//     {
//         state->add_ctl_result(1, structural_output);
//     }
//     std::cout << "Added mux output to FSM node with results size " << structural_output->results.size() << ", empty " << structural_output->results.empty() << std::endl;
//     return structural_output;
// };

fsm_state *
fsm_node_temp::add_state()
{
    auto state = fsm_state::create(this, states_.size());

    for (size_t i = 0; i < noutputs(); i++)
    {
        state->add_ctl_result(2, output(i));
    }

    states_.push_back(state);
    return state;
};

void
fsm_node_builder::generate_gamma(jlm::rvsdg::output *predicate)
{
    gamma_ = jlm::rvsdg::gamma_node::create(predicate, fsm_node_temp_->states_.size());

    for (size_t i = 0; i < fsm_node_temp_->noutputs(); i++)
    {
        std::vector<jlm::rvsdg::output *> results_origins;
        for (auto &result : fsm_node_temp_->output(i)->results)
        {
            auto region = gamma_->subregion(result.region()->index());
            auto new_ctrl_const = static_cast<jlm::rvsdg::node_output*>(result.origin())->node()->copy(region, {});
            results_origins.push_back(new_ctrl_const->output(0));
        }
        gamma_->add_exitvar(results_origins);
    }

    for (size_t i = 0; i < fsm_node_temp_->noutputs(); i++)
    {
        fsm_node_temp_->output(i)->divert_users(gamma_->output(i));
    }

    delete fsm_node_temp_;
};

} // namespace jlm::static_hls