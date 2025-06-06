/*
 * Copyright 2021 David Metz <david.c.metz@ntnu.no>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_BACKEND_HLS_RVSDG2RHLS_MEM_CONV_HPP
#define JLM_BACKEND_HLS_RVSDG2RHLS_MEM_CONV_HPP

#include "jlm/llvm/ir/operators/IntegerOperations.hpp"
#include <jlm/llvm/ir/operators/lambda.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>

namespace jlm::hls
{

typedef std::vector<std::tuple<
    std::vector<jlm::rvsdg::SimpleNode *>,
    std::vector<jlm::rvsdg::SimpleNode *>,
    std::vector<jlm::rvsdg::SimpleNode *>>>
    port_load_store_decouple;

/**
 * Traces all pointer arguments of a lambda node and finds all memory operations.
 * Pointers read from memory is not traced, i.e., the output of load operations is not traced.
 * @param lambda The lambda node for which to trace all pointer arguments
 * @param portNodes A vector where each element contains all memory operations traced from a pointer
 */
void
TracePointerArguments(const rvsdg::LambdaNode * lambda, port_load_store_decouple & portNodes);

jlm::rvsdg::SimpleNode *
find_decouple_response(
    const jlm::rvsdg::LambdaNode * lambda,
    const jlm::llvm::IntegerConstantOperation * request_constant);

void
MemoryConverter(llvm::RvsdgModule & rm);

/**
 * @param lambda The lambda node for wich the load and store operations are to be connected to
 * response (argument) ports
 * @param argumentIndex The index of the reponse (argument) port to be connected
 * @param smap The substitution map for the lambda node
 * @param originalLoadNodes The load nodes to be connected to the reponse port
 * @param originalStoreNodes The store nodes to be connected to the reponse port
 * @param originalDecoupledNodes The decouple nodes to be connected to the reponse port
 * @result The request output to which the memory operations are connected
 */
jlm::rvsdg::Output *
ConnectRequestResponseMemPorts(
    const rvsdg::LambdaNode * lambda,
    size_t argumentIndex,
    rvsdg::SubstitutionMap & smap,
    const std::vector<jlm::rvsdg::SimpleNode *> & originalLoadNodes,
    const std::vector<jlm::rvsdg::SimpleNode *> & originalStoreNodes,
    const std::vector<jlm::rvsdg::SimpleNode *> & originalDecoupledNodes);

jlm::rvsdg::SimpleNode *
ReplaceLoad(
    rvsdg::SubstitutionMap & smap,
    const jlm::rvsdg::SimpleNode * originalLoad,
    jlm::rvsdg::Output * response);

jlm::rvsdg::SimpleNode *
ReplaceStore(
    rvsdg::SubstitutionMap & smap,
    const jlm::rvsdg::SimpleNode * originalStore,
    rvsdg::Output * response);

} // namespace jlm::hls

#endif // JLM_BACKEND_HLS_RVSDG2RHLS_MEM_CONV_HPP
