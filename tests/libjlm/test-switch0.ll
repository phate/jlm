; ModuleID = 'main.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test_switch0(i32 %n) nounwind uwtable readnone {
entry:
  switch i32 %n, label %sw.default [
    i32 7, label %sw.epilog
    i32 8, label %sw.bb1
  ]

sw.bb1:                                           ; preds = %entry
  br label %sw.epilog

sw.default:                                       ; preds = %entry
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.default, %sw.bb1
  %x.0 = phi i32 [ 11, %sw.default ], [ 6, %sw.bb1 ], [ 5, %entry ]
  ret i32 %x.0
}
