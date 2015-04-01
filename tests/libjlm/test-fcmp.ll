; ModuleID = 'main.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


define i1 @test_fcmp_true_half(half %a, half %b) nounwind uwtable {
entry:
  %cmp = fcmp true half %a, %b
  ret i1 %cmp
}

define i1 @test_fcmp_true_float(float %a, float %b) nounwind uwtable {
entry:
  %cmp = fcmp true float %a, %b
  ret i1 %cmp
}

define i1 @test_fcmp_true_double(double %a, double %b) nounwind uwtable {
entry:
  %cmp = fcmp true double %a, %b
  ret i1 %cmp
}



define i1 @test_fcmp_false_half(half %a, half %b) nounwind uwtable {
entry:
  %cmp = fcmp false half %a, %b
  ret i1 %cmp
}

define i1 @test_fcmp_false_float(float %a, float %b) nounwind uwtable {
entry:
  %cmp = fcmp false float %a, %b
  ret i1 %cmp
}

define i1 @test_fcmp_false_double(double %a, double %b) nounwind uwtable {
entry:
  %cmp = fcmp false double %a, %b
  ret i1 %cmp
}
