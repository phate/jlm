; ModuleID = 'dummy.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @callee0() nounwind uwtable {
entry:
	ret void
}

define i32 @callee1(i32 %x) nounwind uwtable {
entry:
  ret i32 %x
}

define i32 @caller(i32 %x) nounwind uwtable {
entry:
  %call1 = call i32 @callee1(i32 %x)
	call void @callee0()
  ret i32 %call1
}
