/*
 * Copyright 2024 Louis Maurin <louis7maurin@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/hls/ir/static/loop.hpp>

namespace jlm::static_hls
{
size_t instances_count = 0;

std::vector<jlm::rvsdg::output *>
loop_node::get_users(jlm::rvsdg::node_input * input) const
{
    auto node = get_mux(input);
    if (!node)
    {
        std::cout << "input is a loop input";
        return {};
    }

    std::vector<jlm::rvsdg::output *> users;

    for (size_t i = 0; i < node->ninputs(); i++)
    {   
        users.push_back(node->input(i)->origin());
    }
    return users;
};

jlm::rvsdg::node*
loop_node::get_mux(jlm::rvsdg::node_input * node) const
{
    auto control_result = get_origin_result(node);
    if (!control_result)
    {
        return nullptr;
    }

    auto node_output = dynamic_cast<jlm::rvsdg::node_output*>(control_result->origin());
    if (!node_output) JLM_UNREACHABLE("SHLS: call of function loop_node::get_mux() with invalid argument");
    JLM_ASSERT(jlm::rvsdg::is<jlm::static_hls::mux_op>(node_output->node()->operation()));
    return node_output->node();
};

loop_node *
loop_node::create(jlm::rvsdg::region * parent)
{
    auto ln = new loop_node(parent);
    ln->fsm_ = fsm_node_builder::create(ln->control_subregion());
    return ln;
};

jlm::rvsdg::structural_output *
loop_node::add_loopvar(jlm::rvsdg::theta_input * theta_input)
{
    auto input = jlm::rvsdg::structural_input::create(this, theta_input->origin(), theta_input->origin()->type());
    auto output = jlm::rvsdg::structural_output::create(this, theta_input->origin()->type());

    auto argument_in = jlm::rvsdg::argument::create(control_subregion(), input, theta_input->origin()->type());
    
    reg_smap_.insert(theta_input->argument(), argument_in);

    return output;
};

// FIXME: This function is not implemented
loop_node *
loop_node::copy(jlm::rvsdg::region * region, jlm::rvsdg::substitution_map & smap) const {
    JLM_UNREACHABLE("SHLS: loop_node::copy() is not implemented");
    return nullptr;
    // auto ln = new loop_node(region);
    // return ln;
};

void
loop_node::add_node(jlm::rvsdg::node * node) {
    std::cout << "Adding node: " << node->operation().debug_string() << std::endl;
    
    jlm::rvsdg::node *compute_node;

    auto fsm_state = fsm_->add_state();

    if (auto implemented_node = is_op_implemented(node->operation())) {
        compute_node = implemented_node;
        std::cout << "Node operation " << node->operation().debug_string() << " already implemented" << std::endl;

        //! *** Add the inputs connection to the muxes of the control region ****
        for (size_t i = 0; i < node->ninputs(); i++)
        {
            // Get the get mux connected to the already implemented node
            auto mux = get_mux(compute_node->input(i));
            if (!mux) JLM_UNREACHABLE("SHLS loop_node::add_node() : mux not found");

            auto input_new_origin = reg_smap_.lookup(node->input(i)->origin());
            if (!input_new_origin) JLM_UNREACHABLE("SHLS loop_node::add_node() : node input origin not found in reg_smap_");

            mux = mux_add_input(mux, input_new_origin);

            auto mux_op = dynamic_cast<const jlm::static_hls::mux_op*>(&mux->operation());
            if (!mux_op) JLM_UNREACHABLE("Mux operation not found");

            fsm_state->set_mux_ctl(*(mux->output(0)->begin()), mux_op->nalternatives()-1);
        }
    
    // If the node is not implemented yet
    } else {
        //! *** Create args in compute region and result and mux in control region for each node input ****
        std::vector<jlm::rvsdg::output *> inputs_args;
        for (size_t i = 0; i < node->ninputs(); i++)
        {
            //! Create arg in compute region
            auto input_arg = backedge_argument::create(compute_subregion(), node->input(i)->type());
            inputs_args.push_back(input_arg);

            auto input_new_origin = reg_smap_.lookup(node->input(i)->origin());
            if (!input_new_origin) JLM_UNREACHABLE(util::strfmt("SHLS loop_node::add_node() : node input origin not found in reg_smap_ for node with op ", node->operation().debug_string()).c_str());
            
            //! This will create a mux without a predicate which is added afterwards
            auto mux = jlm::static_hls::mux_op::create({input_new_origin});

            //! Create corresponding result in control region
            auto res = backedge_result::create(mux->output(0));
            res->argument_ = input_arg;
            input_arg->result_ = res;

            auto mux_op = dynamic_cast<const jlm::static_hls::mux_op*>(&mux->operation());
            if (!mux_op) JLM_UNREACHABLE("Mux operation not found");

            fsm_state->set_mux_ctl(res, mux_op->nalternatives()-1);
        }

        // Copy the node into the compute subregion of the loop
        compute_node = node->copy(compute_subregion(), inputs_args);
    }

    //! Create a backedge and register for each of the node outputs
    for (size_t i = 0; i < node->noutputs(); i++)
    {
        auto backedge_result = add_backedge(compute_node->output(i));
        auto reg_store_origin = fsm_->add_register_ouput();

        auto reg = reg_op::create(*reg_store_origin, *backedge_result->argument(), jlm::util::strfmt(compute_node->operation().debug_string(), ":", i));
        fsm_state->enable_reg(reg);
        
        reg_smap_.insert(node->output(i), reg->output(0));
    }
};

void
loop_node::add_loopback_arg(jlm::rvsdg::theta_input * theta_input)
{
    auto new_arg_out = reg_smap_.lookup(theta_input->argument());
    if (!new_arg_out) JLM_UNREACHABLE("SHLS: loop_node::add_loopback_arg() cannot find the argument in reg_smap_");

    auto new_arg = dynamic_cast<jlm::rvsdg::argument*>(new_arg_out);
    if (!new_arg) JLM_UNREACHABLE("SHLS: loop_node::add_loopback_arg() cannot cast to arg");

    auto fsm_state = fsm_->add_state();

    std::cout << "new_arg nusers: " << new_arg->nusers() << std::endl;

    //! Need to iterate through the old users because the users are modified during the loop
    std::vector<jlm::rvsdg::input *> old_users(new_arg->begin(), new_arg->end());

    for (auto user : old_users)
    {
        auto mux_in = dynamic_cast<jlm::rvsdg::node_input*>(user);
        if (!mux_in) JLM_UNREACHABLE("SHLS: loop_node::add_loopback_arg() cannot cast to node_input");

        auto mux = mux_in->node();

        auto mux_op = dynamic_cast<const jlm::static_hls::mux_op*>(&mux->operation());
        if (!mux_op) JLM_UNREACHABLE("SHLS: loop_node::add_loopback_arg() mux operation not found");

        auto new_result_origin = reg_smap_.lookup(theta_input->result()->origin());

        mux = mux_add_input(mux, new_result_origin);
        fsm_state->set_mux_ctl(*(mux->output(0)->begin()), mux_op->nalternatives()-1);
    };
};

//TODO optimize this function by using a set of operations
jlm::rvsdg::node *
loop_node::is_op_implemented(const jlm::rvsdg::operation& op) const noexcept {
    for (size_t ind_res=0 ; ind_res < compute_subregion()->nresults(); ind_res++) {
        if (static_cast<jlm::rvsdg::node_output*>(compute_subregion()->result(ind_res)->origin())->node()->operation() == op)
        {
            return static_cast<jlm::rvsdg::node_output*>(compute_subregion()->result(ind_res)->origin())->node();
        }
    }
    return nullptr;
};

backedge_result *
loop_node::add_backedge(jlm::rvsdg::output* origin)
{
  auto result_loop = backedge_result::create(origin);
  auto argument_loop = backedge_argument::create(control_subregion(), origin->type());
  argument_loop->result_ = result_loop;
  result_loop->argument_ = argument_loop;
  return result_loop;
};

jlm::rvsdg::result *
loop_node::get_origin_result(jlm::rvsdg::node_input * input) const
{
    auto arg = dynamic_cast<backedge_argument *>(input->origin());
    if (!arg) return nullptr; // This is a loop_var
    return arg->result();
};

void
loop_node::print_nodes_registers() const
{
    std::cout << "**** Printing nodes and registers ****" << std::endl;
    for (size_t i=0; i<compute_subregion()->nresults(); i++)
    {
        auto node_ouput = static_cast<jlm::rvsdg::node_output*>(compute_subregion()->result(i)->origin());
        
        auto node = node_ouput->node();
        std::cout << "node " << node->operation().debug_string() << " | ouput " << node_ouput->index() << " = ";
        
        auto backedge_result = dynamic_cast<jlm::static_hls::backedge_result*>(compute_subregion()->result(i));
        if (!backedge_result) 
        {
            JLM_UNREACHABLE("SHLS: loop_node::print_nodes_registers() cannot cast to backedge_result");
        }
        auto arg_user = backedge_result->argument()->begin();

        auto node_in = dynamic_cast<jlm::rvsdg::node_input*>(*arg_user);
        if (!node_in)
        {
            std::cout << "arg_user is not a node_input" << std::endl;
            continue;
        }
        std::cout << node_in->node()->operation().debug_string() << std::endl;

        // auto reg_out = reg_smap_.lookup(node.output(i));
        // if (!reg_out) 
        // {
        //   std::cout << "node ouput " << i << " not in reg_smap_ " << std::endl;
        //   continue;
        // }
        // auto node_out = dynamic_cast<jlm::rvsdg::node_output*>(reg_out);
        // if (!node_out)
        // {
        //   std::cout << "node ouput " << i << " substitute is not a node_ouput" << std::endl;
        //   continue;
        // }
        // std::cout << "node ouput " << i << " : " << node_out->node()->operation().debug_string() << std::endl;
    }
};

void
loop_node::remove_single_input_muxes()
{
    for (size_t i=0; i<control_subregion()->nresults(); i++)
    {
        auto node_output = dynamic_cast<jlm::rvsdg::node_output*>(control_subregion()->result(i)->origin());
        if (!node_output) JLM_UNREACHABLE("SHLS: loop_node::remove_single_input_muxes() cannot cast to node_output");

        auto mux = node_output->node();
        JLM_ASSERT(jlm::rvsdg::is<jlm::static_hls::mux_op>(mux->operation()));

        auto mux_op = dynamic_cast<const jlm::static_hls::mux_op*>(&mux->operation());
        if (!mux_op) JLM_UNREACHABLE("Mux operation not found");

        if (mux_op->nalternatives() == 1)
        {
            size_t input_id = 0;
            if (mux_op->has_predicate()) input_id = 1;
            mux->output(0)->divert_users(mux->input(input_id)->origin());
            remove(mux);
        }
    }
};

void
loop_node::connect_muxes()
{
    for (size_t i=0; i<control_subregion()->nresults(); i++)
    {
        auto node_output = dynamic_cast<jlm::rvsdg::node_output*>(control_subregion()->result(i)->origin());
        if (!node_output) JLM_UNREACHABLE("SHLS: loop_node::connect_muxes() cannot cast to node_output");

        auto mux = node_output->node();
        JLM_ASSERT(jlm::rvsdg::is<jlm::static_hls::mux_op>(mux->operation()));

        auto mux_op = dynamic_cast<const jlm::static_hls::mux_op*>(&mux->operation());
        if (!mux_op) JLM_UNREACHABLE("Mux operation not found");

        auto mux_ctl_origin = fsm_->add_ctl_output(mux_op->nalternatives());
        mux_connect_predicate(mux, mux_ctl_origin);
    }
};

void
loop_node::finalize()
{
    //FIXME this is a temporary solution with an argument
    auto arg = jlm::rvsdg::argument::create(control_subregion(), nullptr, jlm::rvsdg::ctltype(fsm_->nalternatives()));
    connect_muxes();
    fsm_->apply_mux_ctl();
    fsm_->generate_gamma(arg);
    remove_single_input_muxes();
};

} // namespace jlm::static_hls