; ModuleID = 'main.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test_getelementptr(i32* nocapture %p) nounwind uwtable readonly {
entry:
  %0 = load i32* %p, align 4, !tbaa !0
  %arrayidx1 = getelementptr inbounds i32* %p, i64 1
  %1 = load i32* %arrayidx1, align 4, !tbaa !0
  %add = add i32 %1, %0
  ret i32 %add
}

!0 = !{!"int", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
