; ModuleID = 'test-casts'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32* @test_bitcast(i32* %x) nounwind uwtable readnone {
entry:
  %conv = bitcast i32* %x to i32*
  ret i32* %conv
}

define double @test_fpext(float %x) nounwind uwtable readnone {
entry:
  %conv = fpext float %x to double
  ret double %conv
}

define i32 @test_fptosi(float %x) nounwind uwtable readnone {
entry:
  %conv = fptosi float %x to i32
  ret i32 %conv
}

define i32 @test_fptoui(float %x) nounwind uwtable readnone {
entry:
  %conv = fptoui float %x to i32
  ret i32 %conv
}

define float @test_fptrunc(double %x) nounwind uwtable readnone {
entry:
  %conv = fptrunc double %x to float
  ret float %conv
}

define i64* @test_inttoptr(i64 %x) nounwind uwtable readnone {
entry:
  %0 = inttoptr i64 %x to i64*
  ret i64* %0
}

define i64 @test_ptrtoint(i64* %x) nounwind uwtable readnone {
entry:
  %0 = ptrtoint i64* %x to i64
  ret i64 %0
}

define i8 @test_sext(i4 %x) nounwind uwtable readnone {
entry:
  %conv = sext i4 %x to i8
  ret i8 %conv
}

define float @test_sitofp(i32 %x) nounwind uwtable readnone {
entry:
  %conv = sitofp i32 %x to float
  ret float %conv
}

define i32 @test_trunc(i64 %x) nounwind uwtable readnone optsize {
entry:
  %conv = trunc i64 %x to i32
  ret i32 %conv
}

define float @test_uitofp(i32 %x) nounwind uwtable readnone {
entry:
  %conv = uitofp i32 %x to float
  ret float %conv
}

define i64 @test_zext(i16 %x) nounwind uwtable readnone {
entry:
  %conv = zext i16 %x to i64
  ret i64 %conv
}
