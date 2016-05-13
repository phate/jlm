; ModuleID = 'test-constants'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test_constantInt(i32 %x) nounwind uwtable readnone optsize {
entry:
  %z = add i32 %x, -1
	ret i32 %z
}

define float @test_constantFP() nounwind uwtable {
entry:
  ret float 1.5
}

define noalias i32* @test_constantPointerNull() nounwind uwtable readnone {
entry:
  ret i32* null
}

define i32 @test_undefValue() nounwind uwtable readnone optsize {
entry:
  ret i32 undef
}

@constant = global i32 42, align 4

define i32 @test_globalVariable() nounwind uwtable readonly {
entry:
  %0 = load i32* @constant, align 4, !tbaa !0
  ret i32 %0
}

define {i64, i32} @test_constantAggregateZeroStruct() nounwind uwtable readonly {
entry:
  ret {i64, i32} zeroinitializer
}

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
