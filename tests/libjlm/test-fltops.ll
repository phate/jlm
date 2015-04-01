; ModuleID = 'main.c'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define float @test_fadd_float(float %a, float %b) nounwind uwtable {
entry:
  %add = fadd float %a, %b
  ret float %add
}

define float @test_fsub_float(float %a, float %b) nounwind uwtable {
entry:
  %sub = fsub float %a, %b
  ret float %sub
}

define float @test_fmul_float(float %a, float %b) nounwind uwtable {
entry:
  %mul = fmul float %a, %b
  ret float %mul
}

define float @test_fdiv_float(float %a, float %b) nounwind uwtable {
entry:
  %div = fdiv float %a, %b
  ret float %div
}
