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

/** \brief MemoryHoistBarrierTest class
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
 * The load's address is routed through a
 * \ref jlm::llvm::MemoryHoistBarrierOperation together with an I/O state.
 */
class MemoryHoistBarrierTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  jlm::rvsdg::SimpleNode * memoryHoistBarrier;
};

/** \brief IntegerConversionTest class
 *
 * This class sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   int32_t f(int32_t p)
 *   {
 *     return (int32_t)(int64_t)p;
 *   }
 * \endcode
 *
 * The sign extension widens the 32-bit argument to 64 bits, which is what makes it a real
 * conversion instead of a no-op, and the truncation narrows it back. Both operations are
 * chained so that the conversion is reachable from the lambda result.
 */
class IntegerConversionTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  jlm::rvsdg::SimpleNode * sext;

  jlm::rvsdg::SimpleNode * trunc;
};

/** \brief FloatBinaryTest class
 *
 * This class sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   void f(double a, double b, double * r1, double * r2, double * r3)
 *   {
 *     double s = a + b;
 *     double d = a - b;
 *     double m = a * b;
 *     double q = s / d;
 *     double r = m % s;
 *     *r1 = s + d;
 *     *r2 = m + q;
 *     *r3 = *r2 + r;
 *   }
 * \endcode
 *
 * All five \ref jlm::llvm::fpop values go through the same converter branch, so they are exercised
 * together. The lambda returns three of the values so that every binary operation is directly
 * reachable from a lambda result.
 *
 * \see FloatConversionTest
 */
class FloatBinaryTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;
};

/** \brief FloatConversionTest class
 *
 * This class sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   double f(double a, float b, int64_t i)
 *   {
 *     double e = (double)b;   // floating point extension
 *     double n = -(e * 2.0);  // negation of a product with a floating point constant
 *     double t = (double)i;   // signed int to floating point
 *     double q = a * a + t;   // fused multiply-add
 *     return n + t, t + q;    // two results
 *   }
 * \endcode
 *
 * In contrast to the \ref FloatBinaryTest, every operation here takes its own branch in both
 * converters: a floating point constant, negation, floating point extension, signed conversion
 * and the multiply-add intrinsic.
 *
 * \see FloatBinaryTest
 */
class FloatConversionTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;
};

/** \brief ComparisonTest class
 *
 * This class sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   int32_t f(double a, double b, int32_t * x, int32_t * y)
 *   {
 *     int32_t p = (a < b) ? 1 : 0;
 *     int32_t q = (x < y) ? 1 : 0;
 *     return p, q;  // two results
 *   }
 * \endcode
 *
 * It combines a floating point comparison, a pointer comparison and the two selects that consume
 * their results. Neither comparison is covered by any other roundtrip graph.
 */
class ComparisonTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  jlm::rvsdg::SimpleNode * fcmp;

  jlm::rvsdg::SimpleNode * ptrcmp;
};

/** \brief IOBarrierTest class
 *
 * This class sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   uint32_t f(uint32_t x)
 *   {
 *     opaque(); // performs an externally visible side-effect
 *     return x + 1;
 *   }
 * \endcode
 *
 * The barrier sequentializes the addition after the side-effect by routing the operand through it.
 * The I/O state that carries the dependency is threaded through the lambda. Since this operation
 * has no LLVM counterpart it is only defined by the RVSDG, so it needs a graph of its own.
 *
 * \see StoreVolatileTest
 */
class IOBarrierTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  jlm::rvsdg::SimpleNode * barrier;

  jlm::rvsdg::SimpleNode * add;
};

/** \brief ConstantDeltaTest class
 *
 * This class sets up an RVSDG representing the following declarations:
 *
 * \code{.c}
 *   __attribute__((section("mysection"))) int mutableGlobal = 1;
 *   __attribute__((section("mysection"))) const int constantGlobal = 1;
 * \endcode
 *
 * Both carry a section name, which distinguishes them from the deltas in the other graphs, and
 * only the second one is marked constant. Both attributes take part in the operation's equality.
 *
 * \see jlm::llvm::DeltaTest1
 */
class ConstantDeltaTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::DeltaNode * mutableDelta;

  jlm::rvsdg::DeltaNode * constantDelta;
};

/** \brief GlobalArrayTest class
 *
 * This class sets up an RVSDG representing the following declarations:
 *
 * \code{.c}
 *   uint32_t initArray[2] = { 1, 2 };
 *   uint32_t zeroArray[2] = { 0, 0 };
 * \endcode
 *
 * The first array is initialized element-wise and the second one is zero-initialized, so each is
 * built by a different operation. Both are held in a delta to give them static storage duration.
 */
class GlobalArrayTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::DeltaNode * initDelta;

  jlm::rvsdg::DeltaNode * zeroDelta;

  jlm::rvsdg::SimpleNode * constantDataArray;

  jlm::rvsdg::SimpleNode * constantAggregateZero;
};

/** \brief WideMemoryNodesTest class
 *
 * This class sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   void * f(int64_t s)
 *   {
 *     uint64_t v;
 *     return malloc(s), &v;
 *   }
 * \endcode
 *
 * Both allocations carry wider types than the ones the memory graphs in \ref jlm::llvm::RvsdgTest
 * use: the malloc size is a 64-bit integer where they use a 32-bit one, and the alloca allocates a
 * 64-bit value rather than a pointer or a 32-bit value. Both widths live in the operations, so this
 * is what puts a 64-bit size and a 64-bit allocated type under the comparison.
 */
class WideMemoryNodesTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  jlm::rvsdg::SimpleNode * malloc;

  jlm::rvsdg::SimpleNode * alloca;
};

/** \brief RootRegionNodesTest class
 *
 * This class sets up an RVSDG whose root region holds simple nodes directly, rather than inside a
 * lambda or a delta:
 *
 * \code{.c}
 *   int64_t constant = 16;
 *   va_list * argumentList = va_arg_list((int64_t)1, (int64_t)2);
 * \endcode
 *
 * Every other graph reaches the conversion through a lambda, so nothing else in the suite exercises
 * a node in the root region itself. That placement is a separate mapping path: the root region
 * becomes an \c Omega node, whose arguments are imports and whose terminator carries the export
 * names. The constants here are 64 bits wide, which is also what the per-operation tests used.
 */
class RootRegionNodesTest final : public jlm::llvm::RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::SimpleNode * constant;

  jlm::rvsdg::SimpleNode * variadicArgumentList;
};

}

#endif
