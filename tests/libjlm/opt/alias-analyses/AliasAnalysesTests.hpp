/*
 * Copyright 2020 Nico Reißmann <nico.reissmann@gmail.com>
 * See COPYING for terms of redistribution.
 */

#include <jive/rvsdg/theta.hpp>

#include <jlm/ir/RvsdgModule.hpp>
#include <jlm/ir/operators.hpp>

/**
* \brief AliasAnalysisTest class
*/
class AliasAnalysisTest {
public:
	jlm::RvsdgModule &
	module()
	{
    if (module_ == nullptr)
      module_ = SetupRvsdg();

		return *module_;
	}

	const jive::graph &
	graph()
	{
		return module().Rvsdg();
	}

private:
  /**
   * \brief Create RVSDG for this test.
   */
	virtual std::unique_ptr<jlm::RvsdgModule>
	SetupRvsdg() = 0;

	std::unique_ptr<jlm::RvsdgModule> module_;
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
class StoreTest1 final : public AliasAnalysisTest {
private:
	std::unique_ptr<jlm::RvsdgModule>
	SetupRvsdg() override;

public:
	jlm::lambda::node * lambda;

	jive::node * size;

	jive::node * alloca_a;
	jive::node * alloca_b;
	jive::node * alloca_c;
	jive::node * alloca_d;
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
class StoreTest2 final : public AliasAnalysisTest {
private:
	std::unique_ptr<jlm::RvsdgModule>
	SetupRvsdg() override;

public:
	jlm::lambda::node * lambda;

	jive::node * size;

	jive::node * alloca_a;
	jive::node * alloca_b;
	jive::node * alloca_x;
	jive::node * alloca_y;
	jive::node * alloca_p;
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
class LoadTest1 final : public AliasAnalysisTest {
private:
	std::unique_ptr<jlm::RvsdgModule>
	SetupRvsdg() override;

public:
	jlm::lambda::node * lambda;

	jive::node * load_p;
	jive::node * load_x;
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
class LoadTest2 final : public AliasAnalysisTest {
private:
	std::unique_ptr<jlm::RvsdgModule>
	SetupRvsdg() override;

public:
	jlm::lambda::node * lambda;

	jive::node * size;

	jive::node * alloca_a;
	jive::node * alloca_b;
	jive::node * alloca_x;
	jive::node * alloca_y;
	jive::node * alloca_p;

	jive::node * load_x;
	jive::node * load_a;
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
class LoadFromUndefTest final : public AliasAnalysisTest {
private:
  std::unique_ptr<jlm::RvsdgModule>
  SetupRvsdg() override;

public:
  [[nodiscard]] const jlm::lambda::node &
  Lambda() const noexcept
  {
    return *Lambda_;
  }

  [[nodiscard]] const jive::node *
  UndefValueNode() const noexcept
  {
    return UndefValueNode_;
  }

private:
  jlm::lambda::node * Lambda_;
  jive::node * UndefValueNode_;
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
class GetElementPtrTest final : public AliasAnalysisTest {
private:
	std::unique_ptr<jlm::RvsdgModule>
	SetupRvsdg() override;

public:
	jlm::lambda::node * lambda;

	jive::node * getElementPtrX;
	jive::node * getElementPtrY;
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
class BitCastTest final : public AliasAnalysisTest {
private:
	std::unique_ptr<jlm::RvsdgModule>
	SetupRvsdg() override;

public:
	jlm::lambda::node * lambda;

	jive::node * bitCast;
};

/** \brief Bits2PtrTest class
*
* This function sets up an RVSDG representing the following code snippet:
*
* \code{.c}
*   static void*
*   bits2ptr(ptrdiff_t i)
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
class Bits2PtrTest final : public AliasAnalysisTest {
private:
  std::unique_ptr<jlm::RvsdgModule>
	SetupRvsdg() override;

public:
	jlm::lambda::node * lambda_bits2ptr;
	jlm::lambda::node * lambda_test;

	jive::node * bits2ptr;

	jive::node * call;
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
class ConstantPointerNullTest final : public AliasAnalysisTest {
private:
	std::unique_ptr<jlm::RvsdgModule>
	SetupRvsdg() override;

public:
	jlm::lambda::node * lambda;

	jive::node * constantPointerNullNode;
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
class CallTest1 final : public AliasAnalysisTest {
private:
	std::unique_ptr<jlm::RvsdgModule>
	SetupRvsdg() override;

public:
  [[nodiscard]] const jlm::CallNode &
  CallF() const noexcept
  {
    return *CallF_;
  }

  [[nodiscard]] const jlm::CallNode &
  CallG() const noexcept
  {
    return *CallG_;
  }

	jlm::lambda::node * lambda_f;
	jlm::lambda::node * lambda_g;
	jlm::lambda::node * lambda_h;

	jive::node * alloca_x;
	jive::node * alloca_y;
	jive::node * alloca_z;

private:
  jlm::CallNode * CallF_;
  jlm::CallNode * CallG_;
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
class CallTest2 final : public AliasAnalysisTest {
public:
  [[nodiscard]] const jlm::CallNode &
  CallCreate1() const noexcept
  {
    return *CallCreate1_;
  }

  [[nodiscard]] const jlm::CallNode &
  CallCreate2() const noexcept
  {
    return *CallCreate2_;
  }

  [[nodiscard]] const jlm::CallNode &
  CallDestroy1() const noexcept
  {
    return *CallDestroy1_;
  }

  [[nodiscard]] const jlm::CallNode &
  CallDestroy2() const noexcept
  {
    return *CallDestroy2_;
  }

	jlm::lambda::node * lambda_create;
	jlm::lambda::node * lambda_destroy;
	jlm::lambda::node * lambda_test;

	jive::node * malloc;
	jive::node * free;

private:
  std::unique_ptr<jlm::RvsdgModule>
  SetupRvsdg() override;

	jlm::CallNode * CallCreate1_;
	jlm::CallNode * CallCreate2_;

	jlm::CallNode * CallDestroy1_;
	jlm::CallNode * CallDestroy2_;
};

/** \brief IndirectCallTest class
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
class IndirectCallTest final : public AliasAnalysisTest {
public:
  [[nodiscard]] const jlm::CallNode &
  CallIndcall() const noexcept
  {
    return *CallIndcall_;
  }

  [[nodiscard]] const jlm::CallNode &
  CallThree() const noexcept
  {
    return *CallThree_;
  }

  [[nodiscard]] const jlm::CallNode &
  CallFour() const noexcept
  {
    return *CallFour_;
  }

	jlm::lambda::node * lambda_three;
	jlm::lambda::node * lambda_four;
	jlm::lambda::node * lambda_indcall;
	jlm::lambda::node * lambda_test;

private:
  std::unique_ptr<jlm::RvsdgModule>
  SetupRvsdg() override;

	jlm::CallNode * CallIndcall_;
	jlm::CallNode * CallThree_;
	jlm::CallNode * CallFour_;
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
class GammaTest final : public AliasAnalysisTest {
private:
	std::unique_ptr<jlm::RvsdgModule>
	SetupRvsdg() override;

public:
	jlm::lambda::node * lambda;

	jive::gamma_node * gamma;
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
class ThetaTest final : public AliasAnalysisTest {
private:
	std::unique_ptr<jlm::RvsdgModule>
	SetupRvsdg() override;

public:
	jlm::lambda::node * lambda;
	jive::theta_node * theta;
	jive::node * gep;
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
class DeltaTest1 final : public AliasAnalysisTest {
public:
  [[nodiscard]] const jlm::CallNode &
  CallG() const noexcept
  {
    return *CallG_;
  }

	jlm::lambda::node * lambda_g;
	jlm::lambda::node * lambda_h;

	jlm::delta::node * delta_f;

  jive::node * constantFive;

private:
  std::unique_ptr<jlm::RvsdgModule>
  SetupRvsdg() override;

  jlm::CallNode * CallG_;
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
class DeltaTest2 final : public AliasAnalysisTest {
public:
  [[nodiscard]] const jlm::CallNode &
  CallF1() const noexcept
  {
    return *CallF1_;
  }

	jlm::lambda::node * lambda_f1;
	jlm::lambda::node * lambda_f2;

	jlm::delta::node * delta_d1;
	jlm::delta::node * delta_d2;

private:
  std::unique_ptr<jlm::RvsdgModule>
  SetupRvsdg() override;

	jlm::CallNode * CallF1_;
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
class ImportTest final : public AliasAnalysisTest {
public:
  [[nodiscard]] const jlm::CallNode &
  CallF1() const noexcept
  {
    return *CallF1_;
  }

	jlm::lambda::node * lambda_f1;
	jlm::lambda::node * lambda_f2;

	jive::argument * import_d1;
	jive::argument * import_d2;

private:
  std::unique_ptr<jlm::RvsdgModule>
  SetupRvsdg() override;

  jlm::CallNode * CallF1_;
};

/** \brief PhiTest class
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
class PhiTest final : public AliasAnalysisTest {
public:
  [[nodiscard]] const jlm::CallNode &
  CallFib() const noexcept
  {
    return *CallFib_;
  }

  [[nodiscard]] const jlm::CallNode &
  CallFibm1() const noexcept
  {
    return *CallFibm1_;
  }

  [[nodiscard]] const jlm::CallNode &
  CallFibm2() const noexcept
  {
    return *CallFibm2_;
  }

	jlm::lambda::node * lambda_fib;
	jlm::lambda::node * lambda_test;

	jive::gamma_node * gamma;

	jlm::phi::node * phi;

	jive::node * alloca;

private:
  std::unique_ptr<jlm::RvsdgModule>
  SetupRvsdg() override;

  jlm::CallNode * CallFibm1_;
  jlm::CallNode * CallFibm2_;

  jlm::CallNode * CallFib_;
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
class ExternalMemoryTest final : public AliasAnalysisTest {
private:
  std::unique_ptr<jlm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::lambda::node * LambdaF;

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
class EscapedMemoryTest1 final : public AliasAnalysisTest {
private:
  std::unique_ptr<jlm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::lambda::node * LambdaTest;

  jlm::delta::node * DeltaA;
  jlm::delta::node * DeltaB;
  jlm::delta::node * DeltaX;
  jlm::delta::node * DeltaY;

  jlm::LoadNode * LoadNode1;
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
class EscapedMemoryTest2 final : public AliasAnalysisTest {
private:
  std::unique_ptr<jlm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::lambda::node * ReturnAddressFunction;
  jlm::lambda::node * CallExternalFunction1;
  jlm::lambda::node * CallExternalFunction2;

  jlm::CallNode * ExternalFunction1Call;
  jlm::CallNode * ExternalFunction2Call;

  jive::node * ReturnAddressMalloc;
  jive::node * CallExternalFunction1Malloc;

  jive::argument * ExternalFunction1Import;
  jive::argument * ExternalFunction2Import;

  jlm::LoadNode * LoadNode;
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
class EscapedMemoryTest3 final : public AliasAnalysisTest {
private:
  std::unique_ptr<jlm::RvsdgModule>
  SetupRvsdg() override;

public:
  jlm::lambda::node * LambdaTest;

  jlm::delta::node * DeltaGlobal;

  jive::argument * ImportExternalFunction;

  jlm::CallNode * CallExternalFunction;

  jlm::LoadNode * LoadNode;
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
class MemcpyTest final : public AliasAnalysisTest
{
public:
  [[nodiscard]] const jlm::lambda::node &
  LambdaF() const noexcept
  {
    return *LambdaF_;
  }

  [[nodiscard]] const jlm::lambda::node &
  LambdaG() const noexcept
  {
    return *LambdaG_;
  }

  [[nodiscard]] const jlm::delta::node &
  LocalArray() const noexcept
  {
    return *LocalArray_;
  }

  [[nodiscard]] const jlm::delta::node &
  GlobalArray() const noexcept
  {
    return *GlobalArray_;
  }

  [[nodiscard]] const jlm::CallNode &
  CallF() const noexcept
  {
    return *CallF_;
  }

  [[nodiscard]] const jive::node &
  Memcpy() const noexcept
  {
    return *Memcpy_;
  }

private:
  std::unique_ptr<jlm::RvsdgModule>
  SetupRvsdg() override;

  jlm::lambda::node * LambdaF_;
  jlm::lambda::node * LambdaG_;

  jlm::delta::node * LocalArray_;
  jlm::delta::node * GlobalArray_;

  jlm::CallNode * CallF_;

  jive::node * Memcpy_;
};