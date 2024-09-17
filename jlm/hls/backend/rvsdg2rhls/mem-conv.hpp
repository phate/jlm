/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_MEM_CONV_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_MEM_CONV_HPP

#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

typedef std::vector<std::tuple<
    std::vector<jlm::rvsdg::simple_node *>,
    std::vector<jlm::rvsdg::simple_node *>,
    std::vector<jlm::rvsdg::simple_node *>>>
    port_load_store_decouple;

void
trace_pointer_arguments(const llvm::lambda::node * ln, port_load_store_decouple & port_nodes);

void
MemoryConverter(llvm::RvsdgModule & rm);

/**
 * @param lambda The lambda node for wich the load and store operations are to be connected to
 * response (arguemnts) ports
 * @param argumentIndex The index of the reponse (argument) port to be connected
 * @param smap The substitution map for the lambda node
 * @param originalLoadNodes The load nodes to be connected to the reponse port
 * @param originalStoreNodes The store nodes to be connected to the reponse port
 * @param originalDecoupledNodes The decouple nodes to be connected to the reponse port
 * @result The request output to which the memory operations are connected
 */
jlm::rvsdg::output *
ConnectRequestResponseMemPorts(
    const llvm::lambda::node * lambda,
    size_t argumentIndex,
    jlm::rvsdg::substitution_map & smap,
    const std::vector<jlm::rvsdg::simple_node *> & originalLoadNodes,
    const std::vector<jlm::rvsdg::simple_node *> & originalStoreNodes,
    const std::vector<jlm::rvsdg::simple_node *> & originalDecoupledNodes);

jlm::rvsdg::simple_node *
ReplaceLoad(
    jlm::rvsdg::substitution_map & smap,
    const jlm::rvsdg::simple_node * originalLoad,
    jlm::rvsdg::output * response);

jlm::rvsdg::simple_node *
ReplaceStore(jlm::rvsdg::substitution_map & smap, const jlm::rvsdg::simple_node * originalStore);

jlm::rvsdg::output *
route_response(rvsdg::Region * target, jlm::rvsdg::output * response);

jlm::rvsdg::output *
route_request(rvsdg::Region * target, jlm::rvsdg::output * request);

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_MEM_CONV_HPP
