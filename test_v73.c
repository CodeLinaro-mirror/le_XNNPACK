
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
void test() {
  HVX_Vector a, b;
  Q6_Vsf_vadd_VsfVsf(a, b);
}

