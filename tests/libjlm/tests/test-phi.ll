; ModuleID = 'inliner_condition_test.cpp'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind readnone uwtable
define i32 @main() {
  br label %1

; <label>:1                                       ; preds = %9, %0
  %e.07 = phi i32 [ 0, %0 ], [ %10, %9 ]
  %f.06 = phi i32 [ 0, %0 ], [ %7, %9 ]
  %i.05 = phi i32 [ 0, %0 ], [ %11, %9 ]
  %g.04 = phi i32 [ 0, %0 ], [ %6, %9 ]
  %2 = add nsw i32 42, %e.07
  %3 = add nsw i32 %2, %f.06
  br label %4

; <label>:4                                       ; preds = %4, %1
  %j.03 = phi i32 [ 0, %1 ], [ %8, %4 ]
  %f.12 = phi i32 [ %3, %1 ], [ %7, %4 ]
  %g.11 = phi i32 [ %g.04, %1 ], [ %6, %4 ]
  %5 = add nsw i32 42, %g.11
  %6 = add nsw i32 %5, %g.11
  %7 = add nsw i32 %5, %f.12
  %8 = add nsw i32 %j.03, 1
  %exitcond = icmp eq i32 %8, 2
  br i1 %exitcond, label %9, label %4

; <label>:9                                       ; preds = %4
  %10 = add nsw i32 %2, %e.07
  %11 = add nsw i32 %i.05, 1
  %exitcond10 = icmp eq i32 %11, 2
  br i1 %exitcond10, label %12, label %1

; <label>:12                                      ; preds = %9
  %13 = add i32 %7, %6
  %14 = add i32 %13, %10
  ret i32 %14
}
