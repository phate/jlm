; ModuleID = 'dummy.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i64 @test_add(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %add = add i64 %y, %x
  ret i64 %add
}

define i64 @test_and(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %and = and i64 %y, %x
  ret i64 %and
}

define i64 @test_ashr(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %shr = ashr i64 %x, %y
  ret i64 %shr
}

define i64 @test_sub(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %sub = sub i64 %x, %y
  ret i64 %sub
}

define i64 @test_udiv(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %div = udiv i64 %x, %y
  ret i64 %div
}

define i64 @test_sdiv(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %div = sdiv i64 %x, %y
  ret i64 %div
}

define i64 @test_urem(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %rem = urem i64 %x, %y
  ret i64 %rem
}

define i64 @test_srem(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %rem = srem i64 %x, %y
  ret i64 %rem
}

define i64 @test_shl(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %shl = shl i64 %x, %y
  ret i64 %shl
}

define i64 @test_lshr(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %shr = lshr i64 %x, %y
  ret i64 %shr
}

define i64 @test_or(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %or = or i64 %y, %x
  ret i64 %or
}

define i64 @test_xor(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %xor = xor i64 %y, %x
  ret i64 %xor
}

define i64 @test_mul(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %mul = mul i64 %y, %x
  ret i64 %mul
}

define i64 @test_slt(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp slt i64 %x, %y
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

define i64 @test_sle(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp sle i64 %x, %y
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

define i64 @test_sge(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp sge i64 %x, %y
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

define i64 @test_sgt(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp sgt i64 %x, %y
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

define i64 @test_ult(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp ult i64 %x, %y
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

define i64 @test_ule(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp ule i64 %x, %y
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

define i64 @test_uge(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp uge i64 %x, %y
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

define i64 @test_ugt(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp ugt i64 %x, %y
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

define i64 @test_eq(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp eq i64 %x, %y
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

define i64 @test_ne(i64 %x, i64 %y) nounwind uwtable readnone optsize {
entry:
  %cmp = icmp ne i64 %x, %y
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}

