; ModuleID = 'main.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @fac(i32 %n) nounwind uwtable readnone {
entry:
  %cmp3 = icmp eq i32 %n, 0
  br i1 %cmp3, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %fac.05 = phi i32 [ %mul, %while.body ], [ 1, %entry ]
  %n.addr.04 = phi i32 [ %sub, %while.body ], [ %n, %entry ]
  %mul = mul i32 %fac.05, %n.addr.04
  %sub = sub i32 %n.addr.04, 1
  %cmp = icmp eq i32 %sub, 0
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %fac.0.lcssa = phi i32 [ 1, %entry ], [ %mul, %while.body ]
  ret i32 %fac.0.lcssa
}
