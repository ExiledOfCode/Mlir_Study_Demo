module {
  func.func @sum_loop(%n_i32: i32) -> i32 {

    //定义
    %n = arith.index_cast %n_i32 : i32 to index

    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.index_cast %c0_i32 : i32 to index

    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.index_cast %c1_i32 : i32 to index

    %init_sum = arith.constant 0 : i32  // 初始累加值

    // 循环结构
    %sum = scf.for %i = %c0 to %n step %c1 iter_args(%sum_iter = %init_sum) -> i32 {
      // 循环体：每次加 1
      %new_sum = arith.addi %sum_iter, %c1_i32 : i32
      scf.yield %new_sum : i32  // 返回更新后的值
    }

    return %sum : i32
  }
}
