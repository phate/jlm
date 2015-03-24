; ModuleID = 'main.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i32 %x) nounwind uwtable readnone {
entry:
  %add = add i32 %x, 1
  %call = tail call i32 @bar(i32 %add)
  ret i32 %call
}

define i32 @bar(i32 %x) nounwind uwtable readnone {
entry:
  %cmp = icmp eq i32 %x, 100
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %add = add i32 %x, 1
  %call = tail call i32 @foo(i32 %add)
  br label %return

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi i32 [ %call, %if.end ], [ %x, %entry ]
  ret i32 %retval.0
}

define i32 @test_phi() nounwind uwtable readnone {
entry:
  %call = tail call i32 @bar(i32 0)
  ret i32 %call
}
