; ModuleID = 'test-phi'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test_phi(i32 %x, i32 %y, i32 %z) nounwind uwtable readnone optsize {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %entry
  %x.addr.0 = phi i32 [ %x, %entry ], [ %z, %while.cond ]
  %cmp = icmp eq i32 %x.addr.0, %y
  br i1 %cmp, label %while.cond, label %while.end

while.end:                                        ; preds = %while.cond
  ret i32 %x.addr.0
}
