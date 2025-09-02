module {
  func.func @affine_sum(%n: index) -> i32 {
    %c0 = arith.constant 10 : i32
    %sum = affine.for %i = 0 to %n iter_args(%sum_iter = %c0) -> i32 {
      %i_32 = arith.index_cast %i : index to i32
      %new_sum = arith.addi %sum_iter, %i_32 : i32
      affine.yield %new_sum : i32
    }
    return %sum : i32
  }
}
