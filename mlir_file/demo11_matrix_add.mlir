module {
  func.func @matrix_add(%A: memref<2x2xi32>, %B: memref<2x2xi32>, %C: memref<2x2xi32>) {
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 2 {
        %a = affine.load %A[%i, %j] : memref<2x2xi32>
        %b = affine.load %B[%i, %j] : memref<2x2xi32>
        %sum = arith.addi %a, %b : i32
        affine.store %sum, %C[%i, %j] : memref<2x2xi32>
      }
    }
    return
  }
}
