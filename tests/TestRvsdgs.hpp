/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jlm/llvm/ir/operators.hpp>
#include <jlm/llvm/ir/RvsdgModule.hpp>
#include <jlm/rvsdg/gamma.hpp>
#include <jlm/rvsdg/theta.hpp>

namespace jlm::tests
{

/**
 * \brief RvsdgTest class
 */
class RvsdgTest
{
public:
  virtual ~RvsdgTest() = default;

  jlm::llvm::RvsdgModule &
  module()
  {
    InitializeTest();
    return *module_;
  }

  const jlm::rvsdg::graph &
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
  virtual std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() = 0;

  std::unique_ptr<jlm::llvm::RvsdgModule> module_;
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::llvm::lambda::node * lambda;

  jlm::rvsdg::node * size;

  jlm::rvsdg::node * alloca_a;
  jlm::rvsdg::node * alloca_b;
  jlm::rvsdg::node * alloca_c;
  jlm::rvsdg::node * alloca_d;
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::llvm::lambda::node * lambda;

  jlm::rvsdg::node * size;

  jlm::rvsdg::node * alloca_a;
  jlm::rvsdg::node * alloca_b;
  jlm::rvsdg::node * alloca_x;
  jlm::rvsdg::node * alloca_y;
  jlm::rvsdg::node * alloca_p;
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::llvm::lambda::node * lambda;

  jlm::rvsdg::node * load_p;
  jlm::rvsdg::node * load_x;
};

/** \brief LoadTest2 class
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
 *     y = *p;
 *   }
 * \endcode
 *
 * It uses a single memory state to sequentialize the respective memory
 * operations.
 */
class LoadTest2 final : public RvsdgTest
{
private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::llvm::lambda::node * lambda;

  jlm::rvsdg::node * size;

  jlm::rvsdg::node * alloca_a;
  jlm::rvsdg::node * alloca_b;
  jlm::rvsdg::node * alloca_x;
  jlm::rvsdg::node * alloca_y;
  jlm::rvsdg::node * alloca_p;

  jlm::rvsdg::node * load_x;
  jlm::rvsdg::node * load_a;
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  [[nodiscard]] const jlm::llvm::lambda::node &
  Lambda() const noexcept
  {
    return *Lambda_;
  }

  [[nodiscard]] const jlm::rvsdg::node *
  UndefValueNode() const noexcept
  {
    return UndefValueNode_;
  }

private:
  jlm::llvm::lambda::node * Lambda_;
  jlm::rvsdg::node * UndefValueNode_;
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::llvm::lambda::node * lambda;

  jlm::rvsdg::node * getElementPtrX;
  jlm::rvsdg::node * getElementPtrY;
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::llvm::lambda::node * lambda;

  jlm::rvsdg::node * bitCast;
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
  [[nodiscard]] const jlm::llvm::lambda::node &
  GetLambdaBits2Ptr() const noexcept
  {
    return *LambdaBits2Ptr_;
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  GetLambdaTest() const noexcept
  {
    return *LambdaTest_;
  }

  [[nodiscard]] const jlm::llvm::CallNode &
  GetCallNode() const noexcept
  {
    return *CallNode_;
  }

  [[nodiscard]] const jlm::rvsdg::node &
  GetBitsToPtrNode() const noexcept
  {
    return *BitsToPtrNode_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::lambda::node * LambdaBits2Ptr_;
  jlm::llvm::lambda::node * LambdaTest_;

  jlm::rvsdg::node * BitsToPtrNode_;

  jlm::llvm::CallNode * CallNode_;
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::llvm::lambda::node * lambda;

  jlm::rvsdg::node * constantPointerNullNode;
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  [[nodiscard]] const jlm::llvm::CallNode &
  CallF() const noexcept
  {
    return *CallF_;
  }

  [[nodiscard]] const jlm::llvm::CallNode &
  CallG() const noexcept
  {
    return *CallG_;
  }

  jlm::llvm::lambda::node * lambda_f;
  jlm::llvm::lambda::node * lambda_g;
  jlm::llvm::lambda::node * lambda_h;

  jlm::rvsdg::node * alloca_x;
  jlm::rvsdg::node * alloca_y;
  jlm::rvsdg::node * alloca_z;

private:
  jlm::llvm::CallNode * CallF_;
  jlm::llvm::CallNode * CallG_;
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
  [[nodiscard]] const jlm::llvm::CallNode &
  CallCreate1() const noexcept
  {
    return *CallCreate1_;
  }

  [[nodiscard]] const jlm::llvm::CallNode &
  CallCreate2() const noexcept
  {
    return *CallCreate2_;
  }

  [[nodiscard]] const jlm::llvm::CallNode &
  CallDestroy1() const noexcept
  {
    return *CallDestroy1_;
  }

  [[nodiscard]] const jlm::llvm::CallNode &
  CallDestroy2() const noexcept
  {
    return *CallDestroy2_;
  }

  jlm::llvm::lambda::node * lambda_create;
  jlm::llvm::lambda::node * lambda_destroy;
  jlm::llvm::lambda::node * lambda_test;

  jlm::rvsdg::node * malloc;
  jlm::rvsdg::node * free;

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::CallNode * CallCreate1_;
  jlm::llvm::CallNode * CallCreate2_;

  jlm::llvm::CallNode * CallDestroy1_;
  jlm::llvm::CallNode * CallDestroy2_;
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
  [[nodiscard]] const jlm::llvm::CallNode &
  CallIndcall() const noexcept
  {
    return *CallIndcall_;
  }

  [[nodiscard]] const jlm::llvm::CallNode &
  CallThree() const noexcept
  {
    return *CallThree_;
  }

  [[nodiscard]] const jlm::llvm::CallNode &
  CallFour() const noexcept
  {
    return *CallFour_;
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  GetLambdaThree() const noexcept
  {
    return *LambdaThree_;
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  GetLambdaFour() const noexcept
  {
    return *LambdaFour_;
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  GetLambdaIndcall() const noexcept
  {
    return *LambdaIndcall_;
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  GetLambdaTest() const noexcept
  {
    return *LambdaTest_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::CallNode * CallIndcall_;
  jlm::llvm::CallNode * CallThree_;
  jlm::llvm::CallNode * CallFour_;

  jlm::llvm::lambda::node * LambdaThree_;
  jlm::llvm::lambda::node * LambdaFour_;
  jlm::llvm::lambda::node * LambdaIndcall_;
  jlm::llvm::lambda::node * LambdaTest_;
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
  [[nodiscard]] jlm::llvm::delta::node &
  GetDeltaG1() const noexcept
  {
    return *DeltaG1_;
  }

  [[nodiscard]] jlm::llvm::delta::node &
  GetDeltaG2() const noexcept
  {
    return *DeltaG2_;
  }

  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaThree() const noexcept
  {
    return *LambdaThree_;
  }

  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaFour() const noexcept
  {
    return *LambdaFour_;
  }

  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaI() const noexcept
  {
    return *LambdaI_;
  }

  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaX() const noexcept
  {
    return *LambdaX_;
  }

  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaY() const noexcept
  {
    return *LambdaY_;
  }

  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaTest() const noexcept
  {
    return *LambdaTest_;
  }

  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaTest2() const noexcept
  {
    return *LambdaTest2_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetIndirectCall() const noexcept
  {
    return *IndirectCall_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetCallIWithThree() const noexcept
  {
    return *CallIWithThree_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetCallIWithFour() const noexcept
  {
    return *CallIWithFour_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetTestCallX() const noexcept
  {
    return *TestCallX_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetTest2CallX() const noexcept
  {
    return *Test2CallX_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetCallY() const noexcept
  {
    return *CallY_;
  }

  [[nodiscard]] jlm::rvsdg::simple_node &
  GetAllocaPx() const noexcept
  {
    return *AllocaPx_;
  }

  [[nodiscard]] jlm::rvsdg::simple_node &
  GetAllocaPy() const noexcept
  {
    return *AllocaPy_;
  }

  [[nodiscard]] jlm::rvsdg::simple_node &
  GetAllocaPz() const noexcept
  {
    return *AllocaPz_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::delta::node * DeltaG1_;
  jlm::llvm::delta::node * DeltaG2_;

  jlm::llvm::lambda::node * LambdaThree_;
  jlm::llvm::lambda::node * LambdaFour_;
  jlm::llvm::lambda::node * LambdaI_;
  jlm::llvm::lambda::node * LambdaX_;
  jlm::llvm::lambda::node * LambdaY_;
  jlm::llvm::lambda::node * LambdaTest_;
  jlm::llvm::lambda::node * LambdaTest2_;

  jlm::llvm::CallNode * IndirectCall_;
  jlm::llvm::CallNode * CallIWithThree_;
  jlm::llvm::CallNode * CallIWithFour_;
  jlm::llvm::CallNode * TestCallX_;
  jlm::llvm::CallNode * Test2CallX_;
  jlm::llvm::CallNode * CallY_;

  jlm::rvsdg::simple_node * AllocaPx_;
  jlm::rvsdg::simple_node * AllocaPy_;
  jlm::rvsdg::simple_node * AllocaPz_;
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
  [[nodiscard]] const jlm::llvm::lambda::node &
  LambdaF() const noexcept
  {
    return *LambdaF_;
  }

  [[nodiscard]] const jlm::llvm::CallNode &
  CallG() const noexcept
  {
    return *CallG_;
  }

  [[nodiscard]] const jlm::rvsdg::RegionArgument &
  ExternalGArgument() const noexcept
  {
    return *ExternalGArgument_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::lambda::node * LambdaF_;

  jlm::llvm::CallNode * CallG_;

  jlm::rvsdg::RegionArgument * ExternalGArgument_;
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
  [[nodiscard]] jlm::llvm::lambda::node &
  LambdaG()
  {
    JLM_ASSERT(LambdaG_ != nullptr);
    return *LambdaG_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::lambda::node * LambdaG_ = {};

  jlm::llvm::CallNode * CallF_ = {};

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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::llvm::lambda::node * lambda;

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
  [[nodiscard]] llvm::lambda::node &
  GetLambdaF() const noexcept
  {
    return *LambdaF_;
  }

  [[nodiscard]] llvm::lambda::node &
  GetLambdaG() const noexcept
  {
    return *LambdaG_;
  }

  [[nodiscard]] llvm::lambda::node &
  GetLambdaH() const noexcept
  {
    return *LambdaH_;
  }

  [[nodiscard]] rvsdg::GammaNode &
  GetGamma() const noexcept
  {
    return *Gamma_;
  }

  [[nodiscard]] llvm::CallNode &
  GetCallFromG() const noexcept
  {
    return *CallFromG_;
  }

  [[nodiscard]] llvm::CallNode &
  GetCallFromH() const noexcept
  {
    return *CallFromH_;
  }

  [[nodiscard]] rvsdg::node &
  GetAllocaXFromG() const noexcept
  {
    return *AllocaXFromG_;
  }

  [[nodiscard]] rvsdg::node &
  GetAllocaYFromG() const noexcept
  {
    return *AllocaYFromG_;
  }

  [[nodiscard]] rvsdg::node &
  GetAllocaXFromH() const noexcept
  {
    return *AllocaXFromH_;
  }

  [[nodiscard]] rvsdg::node &
  GetAllocaYFromH() const noexcept
  {
    return *AllocaYFromH_;
  }

  [[nodiscard]] rvsdg::node &
  GetAllocaZ() const noexcept
  {
    return *AllocaZ_;
  }

private:
  std::unique_ptr<llvm::RvsdgModule>
  SetupRvsdg() override;

  llvm::lambda::node * LambdaF_;
  llvm::lambda::node * LambdaG_;
  llvm::lambda::node * LambdaH_;

  rvsdg::GammaNode * Gamma_;

  llvm::CallNode * CallFromG_;
  llvm::CallNode * CallFromH_;

  rvsdg::node * AllocaXFromG_;
  rvsdg::node * AllocaYFromG_;
  rvsdg::node * AllocaXFromH_;
  rvsdg::node * AllocaYFromH_;
  rvsdg::node * AllocaZ_;
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::llvm::lambda::node * lambda;
  jlm::rvsdg::ThetaNode * theta;
  jlm::rvsdg::node * gep;
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
  [[nodiscard]] const jlm::llvm::CallNode &
  CallG() const noexcept
  {
    return *CallG_;
  }

  jlm::llvm::lambda::node * lambda_g;
  jlm::llvm::lambda::node * lambda_h;

  jlm::llvm::delta::node * delta_f;

  jlm::rvsdg::node * constantFive;

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::CallNode * CallG_;
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
  [[nodiscard]] const jlm::llvm::CallNode &
  CallF1() const noexcept
  {
    return *CallF1_;
  }

  jlm::llvm::lambda::node * lambda_f1;
  jlm::llvm::lambda::node * lambda_f2;

  jlm::llvm::delta::node * delta_d1;
  jlm::llvm::delta::node * delta_d2;

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::CallNode * CallF1_;
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
  [[nodiscard]] const jlm::llvm::lambda::node &
  LambdaF() const noexcept
  {
    return *LambdaF_;
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  LambdaTest() const noexcept
  {
    return *LambdaTest_;
  }

  [[nodiscard]] const jlm::llvm::delta::node &
  DeltaG1() const noexcept
  {
    return *DeltaG1_;
  }

  [[nodiscard]] const jlm::llvm::delta::node &
  DeltaG2() const noexcept
  {
    return *DeltaG2_;
  }

  [[nodiscard]] const jlm::llvm::CallNode &
  CallF() const noexcept
  {
    return *CallF_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::lambda::node * LambdaF_;
  jlm::llvm::lambda::node * LambdaTest_;

  jlm::llvm::delta::node * DeltaG1_;
  jlm::llvm::delta::node * DeltaG2_;

  jlm::llvm::CallNode * CallF_;
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
  [[nodiscard]] const jlm::llvm::CallNode &
  CallF1() const noexcept
  {
    return *CallF1_;
  }

  jlm::llvm::lambda::node * lambda_f1;
  jlm::llvm::lambda::node * lambda_f2;

  jlm::rvsdg::RegionArgument * import_d1;
  jlm::rvsdg::RegionArgument * import_d2;

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::CallNode * CallF1_;
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
  [[nodiscard]] const jlm::llvm::CallNode &
  CallFib() const noexcept
  {
    return *CallFib_;
  }

  [[nodiscard]] const jlm::llvm::CallNode &
  CallFibm1() const noexcept
  {
    return *CallFibm1_;
  }

  [[nodiscard]] const jlm::llvm::CallNode &
  CallFibm2() const noexcept
  {
    return *CallFibm2_;
  }

  jlm::llvm::lambda::node * lambda_fib;
  jlm::llvm::lambda::node * lambda_test;

  rvsdg::GammaNode * gamma;

  jlm::llvm::phi::node * phi;

  jlm::rvsdg::node * alloca;

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::CallNode * CallFibm1_;
  jlm::llvm::CallNode * CallFibm2_;

  jlm::llvm::CallNode * CallFib_;
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
  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaEight() const noexcept
  {
    return *LambdaEight_;
  }

  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaI() const noexcept
  {
    return *LambdaI_;
  }

  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaA() const noexcept
  {
    return *LambdaA_;
  }

  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaB() const noexcept
  {
    return *LambdaB_;
  }

  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaC() const noexcept
  {
    return *LambdaC_;
  }

  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaD() const noexcept
  {
    return *LambdaD_;
  }

  [[nodiscard]] jlm::llvm::lambda::node &
  GetLambdaTest() const noexcept
  {
    return *LambdaTest_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetCallAFromTest() const noexcept
  {
    return *CallAFromTest_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetCallAFromC() const noexcept
  {
    return *CallAFromC_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetCallAFromD() const noexcept
  {
    return *CallAFromD_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetCallB() const noexcept
  {
    return *CallB_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetCallC() const noexcept
  {
    return *CallC_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetCallD() const noexcept
  {
    return *CallD_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetCallI() const noexcept
  {
    return *CallI_;
  }

  [[nodiscard]] jlm::llvm::CallNode &
  GetIndirectCall() const noexcept
  {
    return *IndirectCall_;
  }

  [[nodiscard]] jlm::rvsdg::simple_node &
  GetPTestAlloca() const noexcept
  {
    return *PTestAlloca_;
  }

  [[nodiscard]] jlm::rvsdg::simple_node &
  GetPaAlloca() const noexcept
  {
    return *PaAlloca_;
  }

  [[nodiscard]] jlm::rvsdg::simple_node &
  GetPbAlloca() const noexcept
  {
    return *PbAlloca_;
  }

  [[nodiscard]] jlm::rvsdg::simple_node &
  GetPcAlloca() const noexcept
  {
    return *PcAlloca_;
  }

  [[nodiscard]] jlm::rvsdg::simple_node &
  GetPdAlloca() const noexcept
  {
    return *PdAlloca_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::lambda::node * LambdaEight_;
  jlm::llvm::lambda::node * LambdaI_;
  jlm::llvm::lambda::node * LambdaA_;
  jlm::llvm::lambda::node * LambdaB_;
  jlm::llvm::lambda::node * LambdaC_;
  jlm::llvm::lambda::node * LambdaD_;
  jlm::llvm::lambda::node * LambdaTest_;

  jlm::llvm::CallNode * CallAFromTest_;
  jlm::llvm::CallNode * CallAFromC_;
  jlm::llvm::CallNode * CallAFromD_;
  jlm::llvm::CallNode * CallB_;
  jlm::llvm::CallNode * CallC_;
  jlm::llvm::CallNode * CallD_;
  jlm::llvm::CallNode * CallI_;
  jlm::llvm::CallNode * IndirectCall_;

  jlm::rvsdg::simple_node * PTestAlloca_;
  jlm::rvsdg::simple_node * PaAlloca_;
  jlm::rvsdg::simple_node * PbAlloca_;
  jlm::rvsdg::simple_node * PcAlloca_;
  jlm::rvsdg::simple_node * PdAlloca_;
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
  [[nodiscard]] const jlm::llvm::delta::node &
  GetDelta() const noexcept
  {
    JLM_ASSERT(Delta_ != nullptr);
    return *Delta_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::delta::node * Delta_ = {};
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::llvm::lambda::node * LambdaF;
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::llvm::lambda::node * LambdaTest;

  jlm::llvm::delta::node * DeltaA;
  jlm::llvm::delta::node * DeltaB;
  jlm::llvm::delta::node * DeltaX;
  jlm::llvm::delta::node * DeltaY;

  jlm::llvm::LoadNonVolatileNode * LoadNode1;
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::llvm::lambda::node * ReturnAddressFunction;
  jlm::llvm::lambda::node * CallExternalFunction1;
  jlm::llvm::lambda::node * CallExternalFunction2;

  jlm::llvm::CallNode * ExternalFunction1Call;
  jlm::llvm::CallNode * ExternalFunction2Call;

  jlm::rvsdg::node * ReturnAddressMalloc;
  jlm::rvsdg::node * CallExternalFunction1Malloc;

  jlm::rvsdg::RegionArgument * ExternalFunction1Import;
  jlm::rvsdg::RegionArgument * ExternalFunction2Import;

  jlm::llvm::LoadNonVolatileNode * LoadNode;
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
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::llvm::lambda::node * LambdaTest;

  jlm::llvm::delta::node * DeltaGlobal;

  jlm::rvsdg::RegionArgument * ImportExternalFunction;

  jlm::llvm::CallNode * CallExternalFunction;

  jlm::llvm::LoadNonVolatileNode * LoadNode;
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
 *    int localArray[5] = {0, 1, 2, 3 , 4};
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
  [[nodiscard]] const jlm::llvm::lambda::node &
  LambdaF() const noexcept
  {
    return *LambdaF_;
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  LambdaG() const noexcept
  {
    return *LambdaG_;
  }

  [[nodiscard]] const jlm::llvm::delta::node &
  LocalArray() const noexcept
  {
    return *LocalArray_;
  }

  [[nodiscard]] const jlm::llvm::delta::node &
  GlobalArray() const noexcept
  {
    return *GlobalArray_;
  }

  [[nodiscard]] const jlm::llvm::CallNode &
  CallF() const noexcept
  {
    return *CallF_;
  }

  [[nodiscard]] const jlm::rvsdg::node &
  Memcpy() const noexcept
  {
    return *Memcpy_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::lambda::node * LambdaF_;
  jlm::llvm::lambda::node * LambdaG_;

  jlm::llvm::delta::node * LocalArray_;
  jlm::llvm::delta::node * GlobalArray_;

  jlm::llvm::CallNode * CallF_;

  jlm::rvsdg::node * Memcpy_;
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
  [[nodiscard]] const jlm::llvm::lambda::node &
  LambdaF() const noexcept
  {
    JLM_ASSERT(LambdaF_ != nullptr);
    return *LambdaF_;
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  LambdaG() const noexcept
  {
    JLM_ASSERT(LambdaG_ != nullptr);
    return *LambdaG_;
  }

  [[nodiscard]] const jlm::llvm::CallNode &
  CallG() const noexcept
  {
    JLM_ASSERT(CallG_ != nullptr);
    return *CallG_;
  }

  [[nodiscard]] const jlm::rvsdg::node &
  Memcpy() const noexcept
  {
    JLM_ASSERT(Memcpy_ != nullptr);
    return *Memcpy_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::lambda::node * LambdaF_ = {};
  jlm::llvm::lambda::node * LambdaG_ = {};

  jlm::llvm::CallNode * CallG_ = {};

  jlm::rvsdg::node * Memcpy_ = {};
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
  [[nodiscard]] const jlm::llvm::lambda::node &
  Lambda() const noexcept
  {
    JLM_ASSERT(Lambda_ != nullptr);
    return *Lambda_;
  }

  [[nodiscard]] const jlm::rvsdg::node &
  Alloca() const noexcept
  {
    JLM_ASSERT(Alloca_ != nullptr);
    return *Alloca_;
  }

  [[nodiscard]] const jlm::rvsdg::node &
  Memcpy() const noexcept
  {
    JLM_ASSERT(Memcpy_ != nullptr);
    return *Memcpy_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::lambda::node * Lambda_ = {};

  jlm::rvsdg::node * Alloca_ = {};

  jlm::rvsdg::node * Memcpy_ = {};
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
  [[nodiscard]] const jlm::rvsdg::node &
  GetAlloca() const noexcept
  {
    return *Alloca_;
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  GetLambdaNext() const noexcept
  {
    return *LambdaNext_;
  }

  [[nodiscard]] const jlm::llvm::delta::node &
  GetDeltaMyList() const noexcept
  {
    return *DeltaMyList_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::delta::node * DeltaMyList_;

  jlm::llvm::lambda::node * LambdaNext_;

  jlm::rvsdg::node * Alloca_;
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
  [[nodiscard]] const jlm::llvm::delta::node &
  GetDeltaNode() const noexcept
  {
    JLM_ASSERT(Delta_);
    return *Delta_;
  }

  [[nodiscard]] const jlm::llvm::delta::output &
  GetDeltaOutput() const noexcept
  {
    JLM_ASSERT(Delta_);
    return *Delta_->output();
  }

  [[nodiscard]] const llvm::GraphImport &
  GetImportOutput() const noexcept
  {
    JLM_ASSERT(Import_);
    return *Import_;
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  GetLambdaNode() const noexcept
  {
    JLM_ASSERT(Lambda_);
    return *Lambda_;
  }

  [[nodiscard]] const jlm::llvm::lambda::output &
  GetLambdaOutput() const noexcept
  {
    JLM_ASSERT(Lambda_);
    return *Lambda_->output();
  }

  [[nodiscard]] const jlm::rvsdg::node &
  GetAllocaNode() const noexcept
  {
    JLM_ASSERT(Alloca_);
    return *Alloca_;
  }

  [[nodiscard]] const jlm::rvsdg::output &
  GetAllocaOutput() const noexcept
  {
    JLM_ASSERT(Alloca_);
    return *Alloca_->output(0);
  }

  [[nodiscard]] const jlm::rvsdg::node &
  GetMallocNode() const noexcept
  {
    JLM_ASSERT(Malloc_);
    return *Malloc_;
  }

  [[nodiscard]] const jlm::rvsdg::output &
  GetMallocOutput() const noexcept
  {
    JLM_ASSERT(Malloc_);
    return *Malloc_->output(0);
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::delta::node * Delta_ = {};

  jlm::llvm::GraphImport * Import_ = {};

  jlm::llvm::lambda::node * Lambda_ = {};

  jlm::rvsdg::node * Alloca_ = {};

  jlm::rvsdg::node * Malloc_ = {};
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

  [[nodiscard]] const jlm::rvsdg::node &
  GetAllocaNode(size_t index) const noexcept
  {
    JLM_ASSERT(index < AllocaNodes_.size());
    return *AllocaNodes_[index];
  }

  [[nodiscard]] const jlm::rvsdg::output &
  GetAllocaOutput(size_t index) const noexcept
  {
    JLM_ASSERT(index < AllocaNodes_.size());
    return *AllocaNodes_[index]->output(0);
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  GetFunction() const noexcept
  {
    JLM_ASSERT(Function_);
    return *Function_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  size_t NumAllocaNodes_;

  std::vector<const rvsdg::node *> AllocaNodes_ = {};

  jlm::llvm::lambda::node * Function_;
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
  [[nodiscard]] const jlm::llvm::delta::node &
  GetGlobal() const noexcept
  {
    JLM_ASSERT(Global_);
    return *Global_;
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  GetLocalFunction() const noexcept
  {
    JLM_ASSERT(LocalFunc_);
    return *LocalFunc_;
  }

  [[nodiscard]] const jlm::rvsdg::output &
  GetLocalFunctionRegister() const noexcept
  {
    JLM_ASSERT(LocalFuncRegister_);
    return *LocalFuncRegister_;
  }

  [[nodiscard]] const jlm::rvsdg::RegionArgument &
  GetLocalFunctionParam() const noexcept
  {
    JLM_ASSERT(LocalFuncParam_);
    return *LocalFuncParam_;
  }

  [[nodiscard]] const jlm::rvsdg::node &
  GetLocalFunctionParamAllocaNode() const noexcept
  {
    JLM_ASSERT(LocalFuncParamAllocaNode_);
    return *LocalFuncParamAllocaNode_;
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  GetExportedFunction() const noexcept
  {
    JLM_ASSERT(ExportedFunc_);
    return *ExportedFunc_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::delta::node * Global_ = {};
  jlm::llvm::lambda::node * LocalFunc_ = {};
  jlm::rvsdg::RegionArgument * LocalFuncParam_ = {};
  jlm::rvsdg::output * LocalFuncRegister_ = {};
  jlm::rvsdg::node * LocalFuncParamAllocaNode_ = {};
  jlm::llvm::lambda::node * ExportedFunc_ = {};
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
  [[nodiscard]] const llvm::lambda::node &
  GetLambdaMain() const noexcept
  {
    return *LambdaMain_;
  }

  [[nodiscard]] const llvm::lambda::node &
  GetLambdaG() const noexcept
  {
    return *LambdaG_;
  }

  [[nodiscard]] const llvm::CallNode &
  GetCall() const noexcept
  {
    return *Call_;
  }

private:
  std::unique_ptr<llvm::RvsdgModule>
  SetupRvsdg() override;

  llvm::lambda::node * LambdaG_ = {};
  llvm::lambda::node * LambdaMain_ = {};
  llvm::CallNode * Call_ = {};
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
  [[nodiscard]] llvm::lambda::node &
  LambdaMain() const noexcept
  {
    return *LambdaMain_;
  }

private:
  std::unique_ptr<llvm::RvsdgModule>
  SetupRvsdg() override;

  llvm::lambda::node * LambdaMain_;
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
  [[nodiscard]] llvm::lambda::node &
  GetLambdaF() const noexcept
  {
    JLM_ASSERT(LambdaF_ != nullptr);
    return *LambdaF_;
  }

  [[nodiscard]] llvm::lambda::node &
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

  [[nodiscard]] llvm::CallNode &
  GetCallH() const noexcept
  {
    JLM_ASSERT(CallH_ != nullptr);
    return *CallH_;
  }

  [[nodiscard]] rvsdg::node &
  GetAllocaNode() const noexcept
  {
    JLM_ASSERT(AllocaNode_ != nullptr);
    return *AllocaNode_;
  }

private:
  std::unique_ptr<llvm::RvsdgModule>
  SetupRvsdg() override;

  llvm::lambda::node * LambdaF_ = {};
  llvm::lambda::node * LambdaG_ = {};

  rvsdg::RegionArgument * ImportH_ = {};

  llvm::CallNode * CallH_ = {};

  rvsdg::node * AllocaNode_ = {};
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
  [[nodiscard]] const jlm::llvm::lambda::node &
  GetLambdaFst() const noexcept
  {
    JLM_ASSERT(LambdaFst_ != nullptr);
    return *LambdaFst_;
  }

  [[nodiscard]] const jlm::llvm::lambda::node &
  GetLambdaG() const noexcept
  {
    JLM_ASSERT(LambdaG_ != nullptr);
    return *LambdaG_;
  }

  [[nodiscard]] rvsdg::node &
  GetAllocaNode() const noexcept
  {
    JLM_ASSERT(AllocaNode_ != nullptr);
    return *AllocaNode_;
  }

private:
  std::unique_ptr<jlm::llvm::RvsdgModule>
  SetupRvsdg() override;

  jlm::llvm::lambda::node * LambdaFst_ = {};
  jlm::llvm::lambda::node * LambdaG_ = {};

  rvsdg::node * AllocaNode_ = {};
};

}
