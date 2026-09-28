/*
 * Copyright 2026 Magnus Sjalander <work@sjalander.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_MLIR_TESTRVSDGS_HPP
#define JLM_MLIR_TESTRVSDGS_HPP

#include <jlm/llvm/TestRvsdgs.hpp>

namespace jlm::mlir
{

/** \brief LoadNonVolatileTest class
 *
 * This class sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   uint32_t f(uint32_t * p)
 *   {
 *     return *p;
 *   }
 * \endcode
 *
 * It uses a single memory state and no I/O state.
 *
 * \see LoadVolatileTest
 */
class LoadNonVolatileTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  jlm::rvsdg::SimpleNode * load;
};

/** \brief LoadVolatileTest class
 *
 * This class sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   uint32_t f(uint32_t * p)
 *   {
 *     return * volatile p;
 *   }
 * \endcode
 *
 * In contrast to the non-volatile case, the load requires an I/O state that
 * sequentializes it with respect to other volatile memory accesses.
 *
 * \see LoadNonVolatileTest
 */
class LoadVolatileTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  jlm::rvsdg::SimpleNode * load;
};

/** \brief StoreNonVolatileTest class
 *
 * This class sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   void f(uint32_t * p, uint32_t v)
 *   {
 *     *p = v;
 *   }
 * \endcode
 *
 * It uses a single memory state and no I/O state.
 *
 * \see StoreVolatileTest
 */
class StoreNonVolatileTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  jlm::rvsdg::SimpleNode * store;
};

/** \brief StoreVolatileTest class
 *
 * This class sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   void f(uint32_t * p, uint32_t v)
 *   {
 *     * volatile p = v;
 *   }
 * \endcode
 *
 * In contrast to the non-volatile case, the store requires an I/O state that
 * sequentializes it with respect to other volatile memory accesses.
 *
 * \see StoreNonVolatileTest
 */
class StoreVolatileTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  jlm::rvsdg::SimpleNode * store;
};

}

#endif
