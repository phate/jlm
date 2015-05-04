; ModuleID = 'main.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @test(i32 %p1, i32 %p2, i32 %p3) nounwind uwtable {
entry:
  %p1.addr = alloca i32, align 4
  %p2.addr = alloca i32, align 4
  %p3.addr = alloca i32, align 4
  %r = alloca i32, align 4
  %i = alloca i32, align 4
  %f = alloca i32, align 4
  %k = alloca i32, align 4
  store i32 %p1, i32* %p1.addr, align 4
  store i32 %p2, i32* %p2.addr, align 4
  store i32 %p3, i32* %p3.addr, align 4
  store i32 0, i32* %r, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc5, %entry
  %0 = load i32* %i, align 4
  %1 = load i32* %p1.addr, align 4
  %cmp = icmp ult i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end7

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %f, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %2 = load i32* %f, align 4
  %3 = load i32* %p2.addr, align 4
  %cmp2 = icmp ult i32 %2, %3
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %4 = load i32* %r, align 4
  %inc = add i32 %4, 1
  store i32 %inc, i32* %r, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %5 = load i32* %f, align 4
  %inc4 = add i32 %5, 1
  store i32 %inc4, i32* %f, align 4
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc5

for.inc5:                                         ; preds = %for.end
  %6 = load i32* %i, align 4
  %inc6 = add i32 %6, 1
  store i32 %inc6, i32* %i, align 4
  br label %for.cond

for.end7:                                         ; preds = %for.cond
  store i32 0, i32* %k, align 4
  br label %for.cond8

for.cond8:                                        ; preds = %for.inc12, %for.end7
  %7 = load i32* %k, align 4
  %8 = load i32* %p3.addr, align 4
  %cmp9 = icmp ult i32 %7, %8
  br i1 %cmp9, label %for.body10, label %for.end14

for.body10:                                       ; preds = %for.cond8
  %9 = load i32* %r, align 4
  %inc11 = add i32 %9, 1
  store i32 %inc11, i32* %r, align 4
  br label %for.inc12

for.inc12:                                        ; preds = %for.body10
  %10 = load i32* %k, align 4
  %inc13 = add i32 %10, 1
  store i32 %inc13, i32* %k, align 4
  br label %for.cond8

for.end14:                                        ; preds = %for.cond8
  %11 = load i32* %r, align 4
  ret i32 %11
}
