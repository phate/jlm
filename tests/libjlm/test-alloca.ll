; ModuleID = 'main.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test_alloca(i32 %x, i32 %y) nounwind uwtable {
entry:
  %mem = alloca [2 x i32], align 4
  %arrayidx = getelementptr inbounds [2 x i32]* %mem, i32 0, i64 0
  store i32 %x, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds [2 x i32]* %mem, i32 0, i64 1
  store i32 %y, i32* %arrayidx1, align 4
  %0 = load i32* %arrayidx, align 4
  %1 = load i32* %arrayidx1, align 4
  %add = add i32 %0, %1
  ret i32 %add
}
