/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#ifndef JLM_LLVM_TESTRVSDGS_HPP
#define JLM_LLVM_TESTRVSDGS_HPP

#include <jlm/llvm/ir/operators/delta.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/lambda.hpp>

namespace jlm::rvsdg
{
class Graph;
class GammaNode;
class LambdaNode;
class PhiNode;
class ThetaNode;
}

namespace jlm::llvm
{

/**
 * \brief RvsdgTest class
 */
class RvsdgTest
{
public:
  virtual ~RvsdgTest() = default;

  jlm::llvm::LlvmRvsdgModule &
  module()
  {
    InitializeTest();
    return *module_;
  }

  const rvsdg::Graph &
  graph()
  {
    return module().Rvsdg();
  }

  /**
   * Needs to be called to create the RVSDG module provided by the class.
   * Will automatically be called by the module() and graph() accessors.
   */
  void
  InitializeTest()
  {
    if (module_ == nullptr)
      module_ = SetupRvsdg();
  }

private:
  /**
   * \brief Create RVSDG for this test.
   */
  virtual std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() = 0;

  std::unique_ptr<jlm::llvm::LlvmRvsdgModule> module_;
};

/** \brief StoreTest1 class
 *
 * This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   void f()
 *   {
 *     uint32_t d;
 *     uint32_t * c;
 *     uint32_t ** b;
 *     uint32_t *** a;
 *
 *     a = &b;
 *     b = &c;
 *     c = &d;
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class StoreTest1 final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  rvsdg::SimpleNode * size;

  rvsdg::SimpleNode * alloca_a;
  rvsdg::SimpleNode * alloca_b;
  rvsdg::SimpleNode * alloca_c;
  rvsdg::SimpleNode * alloca_d;
};

/** \brief StoreTest2 class
 *
 * This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   void f()
 *   {
 *     uint32_t a, b;
 *     uint32_t * x, * y;
 *     uint32_t ** p;
 *
 *     x = &a;
 *     y = &b;
 *     p = &x;
 *     p = &y;
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class StoreTest2 final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  rvsdg::SimpleNode * size;

  rvsdg::SimpleNode * alloca_a;
  rvsdg::SimpleNode * alloca_b;
  rvsdg::SimpleNode * alloca_x;
  rvsdg::SimpleNode * alloca_y;
  rvsdg::SimpleNode * alloca_p;
};

/** \brief LoadTest1 class
 *
 * This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   uint32_t f(uint32_t ** p)
 *   {
 *     uint32_t * x = *p;
 *     uint32_t a = *x;
 *     return a;
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class LoadTest1 final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  rvsdg::Node * load_p;
  rvsdg::Node * load_x;
};

/** \brief LoadTest2 class
 *
 * This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   void f()
 *   {
 *     // Represented using allocas
 *     uint32_t a, b;
 *     uint32_t * x, * y;
 *     uint32_t ** p;
 *
 *     x = &a;
 *     y = &b;
 *     p = &x;
 *
 *     // Represented as virtual registers
 *     uint32_t * load_x = *p;
 *     uint32_t load_a = *load_x;
 *     y = load_a;
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class LoadTest2 final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  rvsdg::Node * size;

  rvsdg::SimpleNode * alloca_a;
  rvsdg::SimpleNode * alloca_b;
  rvsdg::SimpleNode * alloca_x;
  rvsdg::SimpleNode * alloca_y;
  rvsdg::SimpleNode * alloca_p;

  rvsdg::SimpleNode * load_x;
  rvsdg::SimpleNode * load_a;
};

/** \brief LoadFromUndefTest class
 *
 * This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   int f()
 *   {
 *     int * x;
 *     return *x;
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class LoadFromUndefTest final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  Lambda() const noexcept
  {
    return *Lambda_;
  }

  [[nodiscard]] const rvsdg::Node *
  UndefValueNode() const noexcept
  {
    return UndefValueNode_;
  }

private:
  jlm::rvsdg::LambdaNode * Lambda_{};
  rvsdg::Node * UndefValueNode_{};
};

/** \brief GetElementPtrTest class
 *
 * This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   struct point {
 *     uint32_t x;
 *     uint32_t y;
 *   };
 *
 *   uint32_t f(const struct point * p)
 *   {
 *     return p->x + p->y;
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class GetElementPtrTest final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  rvsdg::Node * getElementPtrX;
  rvsdg::Node * getElementPtrY;
};

/** \brief BitCastTest class
 *
 * This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   uint16_t * f(uint32_t * p)
 *   {
 *     return (uint16_t*)p;
 *   }
 * \endcode
 */
class BitCastTest final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  rvsdg::Node * bitCast;
};

/** \brief Bits2PtrTest class
 *
 * This function sets up an RVSDG representing the following code snippet:
 *
 * \code{.c}
 *   static void*
 *   bit2ptr(ptrdiff_t i)
 *   {
 *     return (void*)i;
 *   }
 *
 *   void
 *   test(ptrdiff_t i)
 *   {
 *     bit2ptr(i);
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory operations.
 */
class Bits2PtrTest final : public RvsdgTest
{
public:
  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  GetLambdaBits2Ptr() const noexcept
  {
    return *LambdaBits2Ptr_;
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  GetLambdaTest() const noexcept
  {
    return *LambdaTest_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  GetCallNode() const noexcept
  {
    return *CallNode_;
  }

  [[nodiscard]] const rvsdg::Node &
  GetBitsToPtrNode() const noexcept
  {
    return *BitsToPtrNode_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::LambdaNode * LambdaBits2Ptr_{};
  jlm::rvsdg::LambdaNode * LambdaTest_{};

  rvsdg::Node * BitsToPtrNode_{};

  rvsdg::SimpleNode * CallNode_{};
};

/** \brief ConstantPointerNullTest class
 *
 * This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   void f(uint32_t ** i)
 *   {
 *	    *i = NULL;
 *   }
 * \endcode
 */
class ConstantPointerNullTest final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  rvsdg::Node * constantPointerNullNode;
};

/** \brief CallTest1 class
 *
 * This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *	  static uint32_t
 *	  f(uint32_t * x, uint32_t * y)
 *	  {
 *	    return *x + *y;
 *	  }
 *
 *	  static uint32_t
 *	  g(uint32_t * x, uint32_t * y)
 *	  {
 *	    return *x - *y;
 *	  }
 *
 *	  uint32_t
 *	  h()
 *	  {
 *	    uint32_t x = 5, y = 6, z = 7;
 *	    return f(&x, &y) + g(&z, &z);
 *	  }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations within each function.
 */
class CallTest1 final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  [[nodiscard]] const rvsdg::SimpleNode &
  CallF() const noexcept
  {
    return *CallF_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  CallG() const noexcept
  {
    return *CallG_;
  }

  jlm::rvsdg::LambdaNode * lambda_f;
  jlm::rvsdg::LambdaNode * lambda_g;
  jlm::rvsdg::LambdaNode * lambda_h;

  rvsdg::SimpleNode * alloca_x;
  rvsdg::SimpleNode * alloca_y;
  rvsdg::SimpleNode * alloca_z;

private:
  rvsdg::SimpleNode * CallF_{};
  rvsdg::SimpleNode * CallG_{};
};

/** \brief CallTest2 class
 *
 * This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *	  static uint32_t *
 *	  create(size_t n)
 *	  {
 *	    return (uint32_t*)malloc(n * sizeof(uint32_t));
 *	  }
 *
 *	  static void
 *	  destroy(uint32_t * p)
 *	  {
 *	    free(p);
 *	  }
 *
 *	  void
 *	  test()
 *	  {
 *		  uint32_t * p1 = create(6);
 *		  uint32_t * p2 = create(7);
 *
 *	    destroy(p1);
 *		  destroy(p2);
 *	  }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations within each function.
 */
class CallTest2 final : public RvsdgTest
{
public:
  [[nodiscard]] const rvsdg::SimpleNode &
  CallCreate1() const noexcept
  {
    return *CallCreate1_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  CallCreate2() const noexcept
  {
    return *CallCreate2_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  CallDestroy1() const noexcept
  {
    return *CallDestroy1_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  CallDestroy2() const noexcept
  {
    return *CallDestroy2_;
  }

  jlm::rvsdg::LambdaNode * lambda_create;
  jlm::rvsdg::LambdaNode * lambda_destroy;
  jlm::rvsdg::LambdaNode * lambda_test;

  rvsdg::SimpleNode * malloc;
  rvsdg::SimpleNode * free;

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  rvsdg::SimpleNode * CallCreate1_{};
  rvsdg::SimpleNode * CallCreate2_{};

  rvsdg::SimpleNode * CallDestroy1_{};
  rvsdg::SimpleNode * CallDestroy2_{};
};

/** \brief IndirectCallTest1 class
 *
 * This function sets up an RVSDG representing the following function:
 *
 *	\code{.c}
 *	  static uint32_t
 *	  four()
 *	  {
 *	    return 4;
 *	  }
 *
 *	  static uint32_t
 *	  three()
 *	  {
 *	    return 3;
 *	  }
 *
 *	  static uint32_t
 *	  indcall(uint32_t (*f)())
 *	  {
 *	    return (*f)();
 *	  }
 *
 *	  uint32_t
 *	  test()
 *	  {
 *	    return indcall(&four) + indcall(&three);
 *	  }
 *	\endcode
 *
 *	It uses a single memory state to sequentialize the respective memory
 * operations within each function.
 */
class IndirectCallTest1 final : public RvsdgTest
{
public:
  [[nodiscard]] const rvsdg::SimpleNode &
  CallIndcall() const noexcept
  {
    return *CallIndcall_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  CallThree() const noexcept
  {
    return *CallThree_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  CallFour() const noexcept
  {
    return *CallFour_;
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  GetLambdaThree() const noexcept
  {
    return *LambdaThree_;
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  GetLambdaFour() const noexcept
  {
    return *LambdaFour_;
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  GetLambdaIndcall() const noexcept
  {
    return *LambdaIndcall_;
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  GetLambdaTest() const noexcept
  {
    return *LambdaTest_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  rvsdg::SimpleNode * CallIndcall_{};
  rvsdg::SimpleNode * CallThree_{};
  rvsdg::SimpleNode * CallFour_{};

  jlm::rvsdg::LambdaNode * LambdaThree_{};
  jlm::rvsdg::LambdaNode * LambdaFour_{};
  jlm::rvsdg::LambdaNode * LambdaIndcall_{};
  jlm::rvsdg::LambdaNode * LambdaTest_{};
};

/** \brief IndirectCallTest2 class
 *
 * This function sets up an RVSDG representing the following program:
 *
 * \code{.c}
 *   static int32_t g1 = 1;
 *   static int32_t g2 = 2;
 *
 *   static int32_t
 *   three()
 *   {
 *     return 3;
 *   }
 *
 *   static int32_t
 *   four()
 *   {
 *     return 4;
 *   }
 *
 *   static int32_t
 *   i(int32_t(*f)())
 *   {
 *     return f();
 *   }
 *
 *   static int32_t
 *   x(int32_t * p)
 *   {
 *     *p = 5;
 *     return i(&three);
 *   }
 *
 *   static int32_t
 *   y(int32_t * p)
 *   {
 *     *p = 6;
 *     return i(&four);
 *   }
 *
 *   int32_t
 *   test()
 *   {
 *     int32_t px;
 *     int32_t py;
 *     int32_t sum = x(&px) + y(&py);
 *     sum += g1 + g2;
 *
 *     return sum + px + py;
 *   }
 *
 *   int32_t
 *   test2()
 *   {
 *     int32_t pz;
 *     return x(&pz);
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations within each function.
 */
class IndirectCallTest2 final : public RvsdgTest
{
public:
  [[nodiscard]] jlm::rvsdg::DeltaNode &
  GetDeltaG1() const noexcept
  {
    return *DeltaG1_;
  }

  [[nodiscard]] jlm::rvsdg::DeltaNode &
  GetDeltaG2() const noexcept
  {
    return *DeltaG2_;
  }

  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaThree() const noexcept
  {
    return *LambdaThree_;
  }

  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaFour() const noexcept
  {
    return *LambdaFour_;
  }

  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaI() const noexcept
  {
    return *LambdaI_;
  }

  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaX() const noexcept
  {
    return *LambdaX_;
  }

  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaY() const noexcept
  {
    return *LambdaY_;
  }

  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaTest() const noexcept
  {
    return *LambdaTest_;
  }

  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaTest2() const noexcept
  {
    return *LambdaTest2_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetIndirectCall() const noexcept
  {
    return *IndirectCall_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetCallIWithThree() const noexcept
  {
    return *CallIWithThree_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetCallIWithFour() const noexcept
  {
    return *CallIWithFour_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetTestCallX() const noexcept
  {
    return *TestCallX_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetTest2CallX() const noexcept
  {
    return *Test2CallX_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetCallY() const noexcept
  {
    return *CallY_;
  }

  [[nodiscard]] jlm::rvsdg::SimpleNode &
  GetAllocaPx() const noexcept
  {
    return *AllocaPx_;
  }

  [[nodiscard]] jlm::rvsdg::SimpleNode &
  GetAllocaPy() const noexcept
  {
    return *AllocaPy_;
  }

  [[nodiscard]] jlm::rvsdg::SimpleNode &
  GetAllocaPz() const noexcept
  {
    return *AllocaPz_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::DeltaNode * DeltaG1_{};
  jlm::rvsdg::DeltaNode * DeltaG2_{};

  jlm::rvsdg::LambdaNode * LambdaThree_{};
  jlm::rvsdg::LambdaNode * LambdaFour_{};
  jlm::rvsdg::LambdaNode * LambdaI_{};
  jlm::rvsdg::LambdaNode * LambdaX_{};
  jlm::rvsdg::LambdaNode * LambdaY_{};
  jlm::rvsdg::LambdaNode * LambdaTest_{};
  jlm::rvsdg::LambdaNode * LambdaTest2_{};

  rvsdg::SimpleNode * IndirectCall_{};
  rvsdg::SimpleNode * CallIWithThree_{};
  rvsdg::SimpleNode * CallIWithFour_{};
  rvsdg::SimpleNode * TestCallX_{};
  rvsdg::SimpleNode * Test2CallX_{};
  rvsdg::SimpleNode * CallY_{};

  jlm::rvsdg::SimpleNode * AllocaPx_{};
  jlm::rvsdg::SimpleNode * AllocaPy_{};
  jlm::rvsdg::SimpleNode * AllocaPz_{};
};

/**
 * This function sets up an RVSDG representing the following program:
 *
 * \code{.c}
 *   int*
 *   g(const char * path, const char * mode);
 *
 *   int*
 *   f(const char * path, const char * mode)
 *   {
 *     return g(path, mode);
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory operations within each
 * function.
 */
class ExternalCallTest1 final : public RvsdgTest
{
public:
  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  LambdaF() const noexcept
  {
    return *LambdaF_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  CallG() const noexcept
  {
    return *CallG_;
  }

  [[nodiscard]] const jlm::rvsdg::GraphImport &
  ExternalGArgument() const noexcept
  {
    return *ExternalGArgument_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::LambdaNode * LambdaF_{};

  rvsdg::SimpleNode * CallG_{};

  jlm::rvsdg::GraphImport * ExternalGArgument_{};
};

/**
 * This function sets up an RVSDG representing the following program:
 *
 * \code{.c}
 *   #include <stdint.h>
 *
 *   typedef struct myStruct
 *   {
 *     uint32_t i;
 *     uint32_t ** p1;
 *     uint32_t ** p2;
 *   } myStruct;
 *
 *   extern void
 *   f(myStruct * s);
 *
 *   void
 *   g()
 *   {
 *     myStruct s;
 *     f(&s);
 *     uint32_t * tmp = *s.p1;
 *     *s.p1 = *s.p2;
 *     *s.p2 = tmp;
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory operations within each
 * function.
 */
class ExternalCallTest2 final : public RvsdgTest
{
public:
  [[nodiscard]] jlm::rvsdg::LambdaNode &
  LambdaG()
  {
    JLM_ASSERT(LambdaG_ != nullptr);
    return *LambdaG_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  CallF()
  {
    JLM_ASSERT(CallF_ != nullptr);
    return *CallF_;
  }

  [[nodiscard]] jlm::rvsdg::RegionArgument &
  ExternalF()
  {
    JLM_ASSERT(ExternalFArgument_ != nullptr);
    return *ExternalFArgument_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::LambdaNode * LambdaG_ = {};

  rvsdg::SimpleNode * CallF_ = {};

  jlm::rvsdg::RegionArgument * ExternalFArgument_ = {};
};

/** \brief GammaTest class
 *
 * This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   uint32_t f(uint32_t c, uint32_t * p1, uint32_t * p2, uint32_t * p3, uint32_t * p4)
 *   {
 *		  uint32_t * tmp1, * tmp2;
 *     if (c == 0) {
 *		    tmp1 = p1;
 *       tmp2 = p2;
 *     } else {
 *		    tmp1 = p3;
 *       tmp2 = p4;
 *     }
 *		  return *tmp1 + *tmp2;
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class GammaTest final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;

  rvsdg::GammaNode * gamma;
};

/** \brief GammaTest2 class
 *
 * This class sets up an RVSDG representing the following code:
 *
 * \code{.c}
 *   static uint32_t
 *   f(uint32_t c, uint32_t* x, uint32_t* y)
 *   {
 *     uint32_t a;
 *     uint32_t* z = NULL;
 *
 *     if (c == 0)
 *     {
 *       a = *x;
 *       *z = 1;
 *     }
 *     else
 *     {
 *       a = *y;
 *       *z = 2;
 *     }
 *
 *     return a + *z;
 *   }
 *
 *   uint32_t
 *   g()
 *   {
 *     uint32_t x = 1;
 *     uint32_t y = 2;
 *
 *     return f(0, &x, &y);
 *   }
 *
 *   uint32_t
 *   h()
 *   {
 *     uint32_t x = 3;
 *     uint32_t y = 4;
 *
 *     return f(1, &x, &y);
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory operations.
 */
class GammaTest2 final : public RvsdgTest
{
public:
  [[nodiscard]] rvsdg::LambdaNode &
  GetLambdaF() const noexcept
  {
    return *LambdaF_;
  }

  [[nodiscard]] rvsdg::LambdaNode &
  GetLambdaG() const noexcept
  {
    return *LambdaG_;
  }

  [[nodiscard]] rvsdg::LambdaNode &
  GetLambdaH() const noexcept
  {
    return *LambdaH_;
  }

  [[nodiscard]] rvsdg::GammaNode &
  GetGamma() const noexcept
  {
    return *Gamma_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetCallFromG() const noexcept
  {
    return *CallFromG_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetCallFromH() const noexcept
  {
    return *CallFromH_;
  }

  [[nodiscard]] rvsdg::Node &
  GetAllocaXFromG() const noexcept
  {
    return *AllocaXFromG_;
  }

  [[nodiscard]] rvsdg::Node &
  GetAllocaYFromG() const noexcept
  {
    return *AllocaYFromG_;
  }

  [[nodiscard]] rvsdg::Node &
  GetAllocaXFromH() const noexcept
  {
    return *AllocaXFromH_;
  }

  [[nodiscard]] rvsdg::Node &
  GetAllocaYFromH() const noexcept
  {
    return *AllocaYFromH_;
  }

  [[nodiscard]] rvsdg::Node &
  GetAllocaZ() const noexcept
  {
    return *AllocaZ_;
  }

private:
  std::unique_ptr<llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  rvsdg::LambdaNode * LambdaF_{};
  rvsdg::LambdaNode * LambdaG_{};
  rvsdg::LambdaNode * LambdaH_{};

  rvsdg::GammaNode * Gamma_{};

  rvsdg::SimpleNode * CallFromG_{};
  rvsdg::SimpleNode * CallFromH_{};

  rvsdg::Node * AllocaXFromG_{};
  rvsdg::Node * AllocaYFromG_{};
  rvsdg::Node * AllocaXFromH_{};
  rvsdg::Node * AllocaYFromH_{};
  rvsdg::Node * AllocaZ_{};
};

/** \brief ThetaTest class
 *
 * This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   void f(uint32_t l, uint32_t  a[], uint32_t c)
 *   {
 *		  uint32_t n = 0;
 *		  do {
 *		    a[n++] = c;
 *		  } while (n < l);
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class ThetaTest final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * lambda;
  jlm::rvsdg::ThetaNode * theta;
  rvsdg::Node * gep;
};

/** \brief DeltaTest1 class
 *
 * This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   static uint32_t f;
 *
 *   static uint32_t
 *   g(uint32_t * v)
 *   {
 *     return *v;
 *   }
 *
 *   uint32_t
 *   h()
 *   {
 *     f = 5;
 *     return g(&f);
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class DeltaTest1 final : public RvsdgTest
{
public:
  [[nodiscard]] const rvsdg::SimpleNode &
  CallG() const noexcept
  {
    return *CallG_;
  }

  jlm::rvsdg::LambdaNode * lambda_g;
  jlm::rvsdg::LambdaNode * lambda_h;

  jlm::rvsdg::DeltaNode * delta_f;

  rvsdg::Node * constantFive;

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  rvsdg::SimpleNode * CallG_{};
};

/** \brief DeltaTest2 class
 *
 *	This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   static uint32_t d1 = 0;
 *   static uint32_t d2 = 0;
 *
 *   static void
 *   f1()
 *   {
 *     d1 = 2;
 *   }
 *
 *   void
 *   f2()
 *   {
 *     d1 = 5;
 *     f1();
 *     d2 = 42;
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class DeltaTest2 final : public RvsdgTest
{
public:
  [[nodiscard]] const rvsdg::SimpleNode &
  CallF1() const noexcept
  {
    return *CallF1_;
  }

  jlm::rvsdg::LambdaNode * lambda_f1;
  jlm::rvsdg::LambdaNode * lambda_f2;

  jlm::rvsdg::DeltaNode * delta_d1;
  jlm::rvsdg::DeltaNode * delta_d2;

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  rvsdg::SimpleNode * CallF1_{};
};

/** \brief DeltaTest3 class
 *
 *	This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   static int32_t g1 = 1L;
 *   static int32_t *g2 = &g1;
 *
 *   static int16_t
 *   f()
 *   {
 *     g2 = g2;
 *     return g1;
 *   }
 *
 *   int16_t
 *   test()
 *   {
 *     return f();
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class DeltaTest3 final : public RvsdgTest
{
public:
  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  LambdaF() const noexcept
  {
    return *LambdaF_;
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  LambdaTest() const noexcept
  {
    return *LambdaTest_;
  }

  [[nodiscard]] const jlm::rvsdg::DeltaNode &
  DeltaG1() const noexcept
  {
    return *DeltaG1_;
  }

  [[nodiscard]] const jlm::rvsdg::DeltaNode &
  DeltaG2() const noexcept
  {
    return *DeltaG2_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  CallF() const noexcept
  {
    return *CallF_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::LambdaNode * LambdaF_{};
  jlm::rvsdg::LambdaNode * LambdaTest_{};

  jlm::rvsdg::DeltaNode * DeltaG1_{};
  jlm::rvsdg::DeltaNode * DeltaG2_{};

  rvsdg::SimpleNode * CallF_{};
};

/** \brief ImportTest class
 *
 *	This function sets up an RVSDG representing the following function:
 *
 * \code{.c}
 *   extern uint32_t d1 = 0;
 *   extern uint32_t d2 = 0;
 *
 *   static void
 *   f1()
 *   {
 *     d1 = 5;
 *   }
 *
 *   void
 *   f2()
 *   {
 *     d1 = 2;
 *     f1();
 *     d2 = 21;
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class ImportTest final : public RvsdgTest
{
public:
  [[nodiscard]] const rvsdg::SimpleNode &
  CallF1() const noexcept
  {
    return *CallF1_;
  }

  jlm::rvsdg::LambdaNode * lambda_f1;
  jlm::rvsdg::LambdaNode * lambda_f2;

  jlm::rvsdg::GraphImport * import_d1;
  jlm::rvsdg::GraphImport * import_d2;

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  rvsdg::SimpleNode * CallF1_{};
};

/** \brief PhiTest1 class
 *
 *	This function sets up an RVSDG representing the following code snippet:
 *
 * \code{.c}
 *	  static void
 *	  fib(uint64_t n, uint64_t result[])
 *	  {
 *	    if (n < 2) {
 *	      result[n] = n;
 *	      return;
 *	    }
 *
 *	    fib(n-1, result);
 *	    fib(n-2, result);
 *	    result[n] = result[n-1] + result[n-2];
 *	  }
 *
 *	  void
 *	  test()
 *	  {
 *	    uint64_t n = 10;
 *	    uint64_t results[n];
 *
 *	    fib(n, results);
 *	  }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class PhiTest1 final : public RvsdgTest
{
public:
  [[nodiscard]] const rvsdg::SimpleNode &
  CallFib() const noexcept
  {
    return *CallFib_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  CallFibm1() const noexcept
  {
    return *CallFibm1_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  CallFibm2() const noexcept
  {
    return *CallFibm2_;
  }

  jlm::rvsdg::LambdaNode * lambda_fib;
  jlm::rvsdg::LambdaNode * lambda_test;

  rvsdg::GammaNode * gamma;

  jlm::rvsdg::PhiNode * phi;

  rvsdg::SimpleNode * alloca;

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  rvsdg::SimpleNode * CallFibm1_{};
  rvsdg::SimpleNode * CallFibm2_{};

  rvsdg::SimpleNode * CallFib_{};
};

/** \brief PhiTest2 class
 *
 * This function sets up an RVSDG representing the following code snippet:
 *
 * \code{.c}
 *  static int32_t
 *  a(int32_t*);
 *
 *  static int32_t
 *  b(int32_t*);
 *
 *  static int32_t
 *  c(int32_t*);
 *
 *  static int32_t
 *  d(int32_t*);
 *
 *  static int32_t
 *  eight()
 *  {
 *    return 8;
 *  }
 *
 *  static int32_t
 *  i(int32_t(*f)())
 *  {
 *    return f();
 *  }
 *
 *  static int32_t
 *  a(int32_t * x)
 *  {
 *    *x = 1;
 *    int32_t pa;
 *    return b(&pa) + d(&pa);
 *  }
 *
 *  static int32_t
 *  b(int32_t * x)
 *  {
 *    *x = 2;
 *    int32_t pb;
 *    return i(&eight) + c(&pb);
 *  }
 *
 *  static int32_t
 *  c(int32_t * x)
 *  {
 *    *x = 3;
 *    int32_t pc;
 *    return a(&pc) + *x;
 *  }
 *
 *  static int32_t
 *  d(int32_t * x)
 *  {
 *    *x = 4;
 *    int32_t pd;
 *    return a(&pd);
 *  }
 *
 *  int32_t
 *  test()
 *  {
 *    int32_t pTest;
 *    return a(&pTest);
 *  }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory operations.
 */
class PhiTest2 final : public RvsdgTest
{
public:
  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaEight() const noexcept
  {
    return *LambdaEight_;
  }

  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaI() const noexcept
  {
    return *LambdaI_;
  }

  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaA() const noexcept
  {
    return *LambdaA_;
  }

  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaB() const noexcept
  {
    return *LambdaB_;
  }

  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaC() const noexcept
  {
    return *LambdaC_;
  }

  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaD() const noexcept
  {
    return *LambdaD_;
  }

  [[nodiscard]] jlm::rvsdg::LambdaNode &
  GetLambdaTest() const noexcept
  {
    return *LambdaTest_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetCallAFromTest() const noexcept
  {
    return *CallAFromTest_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetCallAFromC() const noexcept
  {
    return *CallAFromC_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetCallAFromD() const noexcept
  {
    return *CallAFromD_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetCallB() const noexcept
  {
    return *CallB_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetCallC() const noexcept
  {
    return *CallC_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetCallD() const noexcept
  {
    return *CallD_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetCallI() const noexcept
  {
    return *CallI_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetIndirectCall() const noexcept
  {
    return *IndirectCall_;
  }

  [[nodiscard]] jlm::rvsdg::SimpleNode &
  GetPTestAlloca() const noexcept
  {
    return *PTestAlloca_;
  }

  [[nodiscard]] jlm::rvsdg::SimpleNode &
  GetPaAlloca() const noexcept
  {
    return *PaAlloca_;
  }

  [[nodiscard]] jlm::rvsdg::SimpleNode &
  GetPbAlloca() const noexcept
  {
    return *PbAlloca_;
  }

  [[nodiscard]] jlm::rvsdg::SimpleNode &
  GetPcAlloca() const noexcept
  {
    return *PcAlloca_;
  }

  [[nodiscard]] jlm::rvsdg::SimpleNode &
  GetPdAlloca() const noexcept
  {
    return *PdAlloca_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::LambdaNode * LambdaEight_{};
  jlm::rvsdg::LambdaNode * LambdaI_{};
  jlm::rvsdg::LambdaNode * LambdaA_{};
  jlm::rvsdg::LambdaNode * LambdaB_{};
  jlm::rvsdg::LambdaNode * LambdaC_{};
  jlm::rvsdg::LambdaNode * LambdaD_{};
  jlm::rvsdg::LambdaNode * LambdaTest_{};

  rvsdg::SimpleNode * CallAFromTest_{};
  rvsdg::SimpleNode * CallAFromC_{};
  rvsdg::SimpleNode * CallAFromD_{};
  rvsdg::SimpleNode * CallB_{};
  rvsdg::SimpleNode * CallC_{};
  rvsdg::SimpleNode * CallD_{};
  rvsdg::SimpleNode * CallI_{};
  rvsdg::SimpleNode * IndirectCall_{};

  jlm::rvsdg::SimpleNode * PTestAlloca_{};
  jlm::rvsdg::SimpleNode * PaAlloca_{};
  jlm::rvsdg::SimpleNode * PbAlloca_{};
  jlm::rvsdg::SimpleNode * PcAlloca_{};
  jlm::rvsdg::SimpleNode * PdAlloca_{};
};

/**
 * This function sets up an RVSDG representing the following code snippet:
 *
 * \code{.c}
 *   #include <stdlib.h>
 *
 *   struct myStruct {
 *     struct myStruct *other;
 *   };
 *
 *   struct myStruct myArray[] = {
 *     {NULL},
 *     {&myArray[0]}
 *   };
 * \endcode
 */
class PhiWithDeltaTest final : public RvsdgTest
{
  [[nodiscard]] const jlm::rvsdg::DeltaNode &
  GetDelta() const noexcept
  {
    JLM_ASSERT(Delta_ != nullptr);
    return *Delta_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::DeltaNode * Delta_ = {};
};

/** \brief ExternalMemoryTest class
 *
 * This function sets up an RVSDG representing the following code snippet:
 *
 * \code{.c}
 *  void
 *  f(int32_t * x, int32_t * y)
 *  {
 *      *x = 1;
 *      *y = 2;
 *  }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class ExternalMemoryTest final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * LambdaF;
};

/** \brief EscapedMemoryTest1 class
 *
 * This class sets up an RVSDG representing the following code snippet:
 *
 * \code{.c}
 *  static int a = 1;
 *  static int *x = &a;
 *  int **y = &x;
 *
 *  static int b = 2;
 *
 *  int
 *  test(int **p)
 *  {
 *    b = 5;
 *
 *    return **p;
 *  }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory operations.
 */
class EscapedMemoryTest1 final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * LambdaTest;

  jlm::rvsdg::DeltaNode * DeltaA;
  jlm::rvsdg::DeltaNode * DeltaB;
  jlm::rvsdg::DeltaNode * DeltaX;
  jlm::rvsdg::DeltaNode * DeltaY;

  rvsdg::SimpleNode * LoadNode1;
};

/** \brief EscapedMemoryTest2 class
 *
 * This class sets up an RVSDG representing the following code snippet:
 *
 * \code{.c}
 *  #include <stdlib.h>
 *
 *  extern void ExternalFunction1(void*);
 *  extern int* ExternalFunction2();
 *
 *  void*
 *  ReturnAddress()
 *  {
 *    return malloc(8);
 *  }
 *
 *  void
 *  CallExternalFunction1()
 *  {
 *    void* address = malloc(8);
 *    ExternalFunction1(address);
 *  }
 *
 *  int
 *  CallExternalFunction2()
 *  {
 *    return *ExternalFunction2();
 *  }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory operations.
 */
class EscapedMemoryTest2 final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * ReturnAddressFunction;
  jlm::rvsdg::LambdaNode * CallExternalFunction1;
  jlm::rvsdg::LambdaNode * CallExternalFunction2;

  rvsdg::SimpleNode * ExternalFunction1Call;
  rvsdg::SimpleNode * ExternalFunction2Call;

  rvsdg::SimpleNode * ReturnAddressMalloc;
  rvsdg::SimpleNode * CallExternalFunction1Malloc;

  jlm::rvsdg::GraphImport * ExternalFunction1Import;
  jlm::rvsdg::GraphImport * ExternalFunction2Import;

  rvsdg::SimpleNode * LoadNode;
};

/** \brief EscapedMemoryTest3 class
 *
 * This class sets up an RVSDG representing the following code snippet:
 *
 * \code{.c}
 *  extern int32_t* externalFunction();
 *  int32_t global = 4;
 *
 *  int32_t
 *  test()
 *  {
 *    return *externalFunction();
 *  }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory operations.
 */
class EscapedMemoryTest3 final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

public:
  jlm::rvsdg::LambdaNode * LambdaTest;

  jlm::rvsdg::DeltaNode * DeltaGlobal;

  jlm::rvsdg::GraphImport * ImportExternalFunction;

  rvsdg::SimpleNode * CallExternalFunction;

  rvsdg::SimpleNode * LoadNode;
};

/** \brief MemcpyTest class
 *
 * This class sets up an RVSDG representing the following code snippet:
 *
 * \code{.c}
 *  #include <string.h>
 *
 *  int globalArray[5];
 *
 *  int
 *  f()
 *  {
 *    globalArray[2] = 6;
 *    return globalArray[2];
 *  }
 *
 *  int
 *  g()
 *  {
 *    int localArray[5] = {0, 1, 2, 3, 4};
 *    memcpy(globalArray, localArray, sizeof(int)*5);
 *    return f();
 *  }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory operations.
 */
class MemcpyTest final : public RvsdgTest
{
public:
  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  LambdaF() const noexcept
  {
    return *LambdaF_;
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  LambdaG() const noexcept
  {
    return *LambdaG_;
  }

  [[nodiscard]] const jlm::rvsdg::DeltaNode &
  LocalArray() const noexcept
  {
    return *LocalArray_;
  }

  [[nodiscard]] const jlm::rvsdg::DeltaNode &
  GlobalArray() const noexcept
  {
    return *GlobalArray_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  CallF() const noexcept
  {
    return *CallF_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  Memcpy() const noexcept
  {
    return *Memcpy_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::LambdaNode * LambdaF_{};
  jlm::rvsdg::LambdaNode * LambdaG_{};

  jlm::rvsdg::DeltaNode * LocalArray_{};
  jlm::rvsdg::DeltaNode * GlobalArray_{};

  rvsdg::SimpleNode * CallF_{};

  rvsdg::SimpleNode * Memcpy_{};
};

/**
 * This class sets up an RVSDG representing the following code snippet:
 *
 * \code{.c}
 *   #include <string.h>
 *
 *   typedef struct structB
 *   {
 *     int * array[32];
 *   } structB;
 *
 *   typedef struct structA
 *   {
 *     int x;
 *     structB * b;
 *   } structA;
 *
 *   static void
 *   g(structB * s1, structB * s2)
 *   {
 *     memcpy(*s2->array, *s1->array, sizeof(int) * 32);
 *   }
 *
 *   void
 *   f(structA * s1, structA * s2)
 *   {
 *     g(s1->b, s2->b);
 *   }
 * \endcode
 */
class MemcpyTest2 final : public RvsdgTest
{
public:
  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  LambdaF() const noexcept
  {
    JLM_ASSERT(LambdaF_ != nullptr);
    return *LambdaF_;
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  LambdaG() const noexcept
  {
    JLM_ASSERT(LambdaG_ != nullptr);
    return *LambdaG_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  CallG() const noexcept
  {
    JLM_ASSERT(CallG_ != nullptr);
    return *CallG_;
  }

  [[nodiscard]] const rvsdg::Node &
  Memcpy() const noexcept
  {
    JLM_ASSERT(Memcpy_ != nullptr);
    return *Memcpy_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::LambdaNode * LambdaF_ = {};
  jlm::rvsdg::LambdaNode * LambdaG_ = {};

  rvsdg::SimpleNode * CallG_ = {};

  rvsdg::Node * Memcpy_ = {};
};

/**
 * This class sets up an RVSDG representing the following code snippet:
 *
 * \code{.c}
 *   #include <stdint.h>
 *   #include <string.h>
 *
 *   typedef struct {
 *     uint8_t * buf;
 *   } myStruct;
 *
 *   void
 *   f(myStruct * p)
 *   {
 *     myStruct s = *p;
 *     memcpy(s.buf, s.buf - 5, 3);
 *   }
 * \endcode
 */
class MemcpyTest3 final : public RvsdgTest
{
public:
  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  Lambda() const noexcept
  {
    JLM_ASSERT(Lambda_ != nullptr);
    return *Lambda_;
  }

  [[nodiscard]] const rvsdg::Node &
  Alloca() const noexcept
  {
    JLM_ASSERT(Alloca_ != nullptr);
    return *Alloca_;
  }

  [[nodiscard]] const rvsdg::Node &
  Memcpy() const noexcept
  {
    JLM_ASSERT(Memcpy_ != nullptr);
    return *Memcpy_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::LambdaNode * Lambda_ = {};

  rvsdg::Node * Alloca_ = {};

  rvsdg::Node * Memcpy_ = {};
};

/** \brief LinkedListTest class
 *
 * This class sets up an RVSDG representing the following code snippet:
 *
 * \code{.c}
 *  struct list
 *  {
 *    struct list * next;
 *  } * myList;
 *
 *  struct list*
 *  next()
 *  {
 *    struct list * tmp = myList;
 *    tmp = tmp->next;
 *    return tmp;
 *  }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory operations.
 */
class LinkedListTest final : public RvsdgTest
{
public:
  [[nodiscard]] const rvsdg::SimpleNode &
  GetAlloca() const noexcept
  {
    return *Alloca_;
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  GetLambdaNext() const noexcept
  {
    return *LambdaNext_;
  }

  [[nodiscard]] const jlm::rvsdg::DeltaNode &
  GetDeltaMyList() const noexcept
  {
    return *DeltaMyList_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::DeltaNode * DeltaMyList_{};

  jlm::rvsdg::LambdaNode * LambdaNext_{};

  rvsdg::SimpleNode * Alloca_{};
};

/** \brief RVSDG module with one of each memory node type.
 *
 * The class sets up an RVSDG module corresponding to the code:
 *
 * \code{.c}
 *   int* global;
 *   extern int imported;
 *
 *   void f()
 *   {
 *     int* alloca:
 *     alloca = malloc(4);
 *     *alloca = imported;
 *     global = alloca;
 *   }
 * \endcode
 *
 * It provides getters for all the memory node creating RVSDG nodes, and their outputs.
 */
class AllMemoryNodesTest final : public RvsdgTest
{
public:
  [[nodiscard]] const jlm::rvsdg::DeltaNode &
  GetDeltaNode() const noexcept
  {
    JLM_ASSERT(Delta_);
    return *Delta_;
  }

  [[nodiscard]] const rvsdg::Output &
  GetDeltaOutput() const noexcept
  {
    JLM_ASSERT(Delta_);
    return Delta_->output();
  }

  [[nodiscard]] const llvm::LlvmGraphImport &
  GetImportOutput() const noexcept
  {
    JLM_ASSERT(Import_);
    return *Import_;
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  GetLambdaNode() const noexcept
  {
    JLM_ASSERT(Lambda_);
    return *Lambda_;
  }

  [[nodiscard]] const rvsdg::Output &
  GetLambdaOutput() const noexcept
  {
    JLM_ASSERT(Lambda_);
    return *Lambda_->output();
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  GetAllocaNode() const noexcept
  {
    JLM_ASSERT(Alloca_);
    return *Alloca_;
  }

  [[nodiscard]] const jlm::rvsdg::Output &
  GetAllocaOutput() const noexcept
  {
    JLM_ASSERT(Alloca_);
    return *Alloca_->output(0);
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  GetMallocNode() const noexcept
  {
    JLM_ASSERT(Malloc_);
    return *Malloc_;
  }

  [[nodiscard]] const jlm::rvsdg::Output &
  GetMallocOutput() const noexcept
  {
    JLM_ASSERT(Malloc_);
    return *Malloc_->output(0);
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::DeltaNode * Delta_ = {};

  jlm::llvm::LlvmGraphImport * Import_ = {};

  jlm::rvsdg::LambdaNode * Lambda_ = {};

  rvsdg::SimpleNode * Alloca_ = {};

  rvsdg::SimpleNode * Malloc_ = {};
};

/** \brief RVSDG module with an arbitrary amount of alloca nodes.
 *
 * The class sets up an RVSDG module corresponding to the code:
 *
 * \code{.c}
 *   void f()
 *   {
 *     uint32_t a;
 *     uint32_t b;
 *     uint32_t c;
 *     ...
 *   }
 * \endcode
 *
 * It provides getters for the alloca nodes themselves, and for their outputs.
 */
class NAllocaNodesTest final : public RvsdgTest
{
public:
  explicit NAllocaNodesTest(size_t numAllocaNodes)
      : NumAllocaNodes_(numAllocaNodes)
  {}

  [[nodiscard]] size_t
  GetNumAllocaNodes() const noexcept
  {
    return NumAllocaNodes_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  GetAllocaNode(size_t index) const noexcept
  {
    JLM_ASSERT(index < AllocaNodes_.size());
    return *AllocaNodes_[index];
  }

  [[nodiscard]] const jlm::rvsdg::Output &
  GetAllocaOutput(size_t index) const noexcept
  {
    JLM_ASSERT(index < AllocaNodes_.size());
    return *AllocaNodes_[index]->output(0);
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  GetFunction() const noexcept
  {
    JLM_ASSERT(Function_);
    return *Function_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  size_t NumAllocaNodes_;

  std::vector<const rvsdg::SimpleNode *> AllocaNodes_ = {};

  jlm::rvsdg::LambdaNode * Function_{};
};

/** \brief RVSDG module with a static function escaping through another function.
 *
 * The class sets up an RVSDG module corresponding to the code:
 *
 * \code{.c}
 *   static uint32_t global;
 *
 *   static uint32_t* localFunc(uint32_t* param)
 *   {
 *     return &global;
 *   }
 *
 *   typedef uint32_t* localFuncSignature(uint32_t*);
 *
 *   localFuncSignature* exportedFunc()
 *   {
 *     return localFunc;
 *   }
 * \endcode
 *
 * It provides getters for the alloca nodes themselves, and for their outputs.
 */
class EscapingLocalFunctionTest final : public RvsdgTest
{
public:
  [[nodiscard]] const jlm::rvsdg::DeltaNode &
  GetGlobal() const noexcept
  {
    JLM_ASSERT(Global_);
    return *Global_;
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  GetLocalFunction() const noexcept
  {
    JLM_ASSERT(LocalFunc_);
    return *LocalFunc_;
  }

  [[nodiscard]] const jlm::rvsdg::Output &
  GetLocalFunctionRegister() const noexcept
  {
    JLM_ASSERT(LocalFuncRegister_);
    return *LocalFuncRegister_;
  }

  [[nodiscard]] const jlm::rvsdg::Output &
  GetLocalFunctionParam() const noexcept
  {
    JLM_ASSERT(LocalFuncParam_);
    return *LocalFuncParam_;
  }

  [[nodiscard]] const rvsdg::Node &
  GetLocalFunctionParamAllocaNode() const noexcept
  {
    JLM_ASSERT(LocalFuncParamAllocaNode_);
    return *LocalFuncParamAllocaNode_;
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  GetExportedFunction() const noexcept
  {
    JLM_ASSERT(ExportedFunc_);
    return *ExportedFunc_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::DeltaNode * Global_ = {};
  jlm::rvsdg::LambdaNode * LocalFunc_ = {};
  jlm::rvsdg::Output * LocalFuncParam_ = {};
  jlm::rvsdg::Output * LocalFuncRegister_ = {};
  rvsdg::Node * LocalFuncParamAllocaNode_ = {};
  jlm::rvsdg::LambdaNode * ExportedFunc_ = {};
};

/** \brief RVSDG module containing a static function that is called with the wrong number of
 * arguments.
 *
 * LLVM permits such code, albeit with issuing the warning: too many arguments in call to 'g'
 *
 * The class sets up an RVSDG module corresponding to the following code:
 *
 * \code{.c}
 *   static unsigned int
 *   g()
 *   {
 *     return 5;
 *   }
 *
 *   int
 *   main()
 *   {
 *     unsigned int x = 6;
 *     return g(x);
 *   }
 * \endcode
 */
class LambdaCallArgumentMismatch final : public RvsdgTest
{
public:
  [[nodiscard]] const rvsdg::LambdaNode &
  GetLambdaMain() const noexcept
  {
    return *LambdaMain_;
  }

  [[nodiscard]] const rvsdg::LambdaNode &
  GetLambdaG() const noexcept
  {
    return *LambdaG_;
  }

  [[nodiscard]] const rvsdg::SimpleNode &
  GetCall() const noexcept
  {
    return *Call_;
  }

private:
  std::unique_ptr<llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  rvsdg::LambdaNode * LambdaG_ = {};
  rvsdg::LambdaNode * LambdaMain_ = {};
  rvsdg::SimpleNode * Call_ = {};
};

/** \brief RVSDG module with a call to free(NULL).
 *
 * The class sets up an RVSDG module corresponding to the code:
 *
 * \code{.c}
 *   int
 *   main()
 *   {
 *     int* x = NULL;
 *     free(x);
 *     return 0;
 *   }
 * \endcode
 *
 */
class FreeNullTest final : public RvsdgTest
{
public:
  [[nodiscard]] rvsdg::LambdaNode &
  LambdaMain() const noexcept
  {
    return *LambdaMain_;
  }

private:
  std::unique_ptr<llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  rvsdg::LambdaNode * LambdaMain_{};
};

/**
 * The class sets up an RVSDG module corresponding to the code:
 *
 * \code{.c}
 *   #include <stdint.h>
 *
 *   uint32_t* h(uint32_t, ...);
 *
 *
 *   static void
 *   f(uint32_t * i)
 *   {
 *     uint32_t* x = h(1, i);
 *     *x = 3;
 *   }
 *
 *   void
 *   g()
 *   {
 *     uint32_t i = 5;
 *     f(&i);
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory operations.
 */
class VariadicFunctionTest1 final : public RvsdgTest
{
public:
  [[nodiscard]] rvsdg::LambdaNode &
  GetLambdaF() const noexcept
  {
    JLM_ASSERT(LambdaF_ != nullptr);
    return *LambdaF_;
  }

  [[nodiscard]] rvsdg::LambdaNode &
  GetLambdaG() const noexcept
  {
    JLM_ASSERT(LambdaG_ != nullptr);
    return *LambdaG_;
  }

  [[nodiscard]] rvsdg::RegionArgument &
  GetImportH() const noexcept
  {
    JLM_ASSERT(ImportH_ != nullptr);
    return *ImportH_;
  }

  [[nodiscard]] rvsdg::SimpleNode &
  GetCallH() const noexcept
  {
    JLM_ASSERT(CallH_ != nullptr);
    return *CallH_;
  }

  [[nodiscard]] rvsdg::Node &
  GetAllocaNode() const noexcept
  {
    JLM_ASSERT(AllocaNode_ != nullptr);
    return *AllocaNode_;
  }

private:
  std::unique_ptr<llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  rvsdg::LambdaNode * LambdaF_ = {};
  rvsdg::LambdaNode * LambdaG_ = {};

  rvsdg::RegionArgument * ImportH_ = {};

  rvsdg::SimpleNode * CallH_ = {};

  rvsdg::Node * AllocaNode_ = {};
};

/**
 * This class sets up an RVSDG representing the following code:
 *
 * \code{.c}
 *   #include <stdarg.h>
 *   #include <stdio.h>
 *   #include <stdint.h>
 *
 *   static int
 *   fst(int n, ...)
 *   {
 *     va_list arguments;
 *     va_start(arguments, n);
 *     int tmp = va_arg(arguments, int);
 *     va_end(arguments);
 *
 *     return tmp;
 *   }
 *
 *   int
 *   g()
 *   {
 *     return fst(3, 0, 1, 2);
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory operations within each
 * function. The code produced by the compiler for variadic functions is architecture specific. This
 * function sets up the code that was produced for x64.
 */
class VariadicFunctionTest2 final : public RvsdgTest
{
public:
  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  GetLambdaFst() const noexcept
  {
    JLM_ASSERT(LambdaFst_ != nullptr);
    return *LambdaFst_;
  }

  [[nodiscard]] const jlm::rvsdg::LambdaNode &
  GetLambdaG() const noexcept
  {
    JLM_ASSERT(LambdaG_ != nullptr);
    return *LambdaG_;
  }

  [[nodiscard]] rvsdg::Node &
  GetAllocaNode() const noexcept
  {
    JLM_ASSERT(AllocaNode_ != nullptr);
    return *AllocaNode_;
  }

private:
  std::unique_ptr<jlm::llvm::LlvmRvsdgModule>
  SetupRvsdg() override;

  jlm::rvsdg::LambdaNode * LambdaFst_ = {};
  jlm::rvsdg::LambdaNode * LambdaG_ = {};

  rvsdg::Node * AllocaNode_ = {};
};

}

#endif
