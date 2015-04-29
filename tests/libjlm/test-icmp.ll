; ModuleID = 'dummy.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i1 @test_slt(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp slt i64 %x, %y
  ret i1 %cmp
}

define i1 @test_sle(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp sle i64 %x, %y
  ret i1 %cmp
}

define i1 @test_sge(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp sge i64 %x, %y
  ret i1 %cmp
}

define i1 @test_sgt(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp sgt i64 %x, %y
  ret i1 %cmp
}

define i1 @test_ult(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp ult i64 %x, %y
  ret i1 %cmp
}

define i1 @test_ule(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp ule i64 %x, %y
  ret i1 %cmp
}

define i1 @test_uge(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp uge i64 %x, %y
  ret i1 %cmp
}

define i1 @test_ugt(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp ugt i64 %x, %y
  ret i1 %cmp
}

define i1 @test_eq(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp eq i64 %x, %y
  ret i1 %cmp
}

define i1 @test_ne(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp ne i64 %x, %y
  ret i1 %cmp
}
