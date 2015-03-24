; ModuleID = 'main.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @fib(i32 %n) nounwind uwtable readnone {
entry:
  %cmp = icmp ult i32 %n, 2
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %sub = add i32 %n, -1
  %call = tail call i32 @fib(i32 %sub)
  %sub1 = add i32 %n, -2
  %call2 = tail call i32 @fib(i32 %sub1)
  %add = add i32 %call2, %call
  ret i32 %add

return:                                           ; preds = %entry
  ret i32 %n
}
