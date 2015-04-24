; ModuleID = 'test-constantInt'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test_constantInt(i32 %x) nounwind uwtable readnone optsize {
entry:
  %z = add i32 %x, -1
	ret i32 %z
}
