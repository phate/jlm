; ModuleID = 'inliner_condition_test.cpp'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; Function Attrs: nounwind readnone uwtable
define i32 @main() #0 {
  br label %1

; <label>:1                                       ; preds = %1, %0
  %e.03 = phi i32 [ 0, %0 ], [ %3, %1 ]
  %i.02 = phi i32 [ 0, %0 ], [ %5, %1 ]
  %f.01 = phi i32 [ 0, %0 ], [ %4, %1 ]
  %2 = tail call fastcc i32 @_ZL1yi(i32 %e.03)
  %3 = add nsw i32 %2, %e.03
  %4 = add nsw i32 %2, %f.01
  %5 = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %5, 2
  br i1 %exitcond, label %6, label %1

; <label>:6                                       ; preds = %1
  %7 = add nsw i32 %3, %4
  ret i32 %7
}

; Function Attrs: nounwind readnone uwtable
define internal fastcc i32 @_ZL1yi(i32 %b) #0 {
  %1 = add nsw i32 %b, 1
  ret i32 %1
}

attributes #0 = { nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
