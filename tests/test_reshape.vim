vim9script

import '../autoload/autograd.vim' as ag

var Tensor = ag.Tensor


def TestSum()
  # case 1
  var x1 = Tensor.new([[20, 40, 60], [1, 4, 5]])
  var y1 = ag.Sum(x1)
  assert_equal([130.0], y1.data)
  assert_equal([1], y1.shape)

  ag.Backward(y1)
  var gx1: Tensor = x1.grad
  assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], gx1.data)
  assert_equal([2, 3], gx1.shape)

  # case 2
  var x2 = Tensor.new([[1, 2, 3], [4, 5, 6]])
  var y2 = ag.Sum(x2, 1)
  assert_equal([2], y2.shape)
  assert_equal([6.0, 15.0], y2.data)

  var y2_2 = ag.Sum(x2, 0)
  assert_equal([3], y2_2.shape)
  assert_equal([5.0, 7.0, 9.0], y2_2.data)

  var y2_3 = ag.Sum(x2, 1, 1)
  assert_equal([2, 1], y2_3.shape)
  assert_equal([6.0, 15.0], y2_3.data)

  x2.ClearGrad()
  ag.Backward(y2)
  gx1 = x1.grad
  assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], gx1.data)
  assert_equal([2, 3], gx1.shape)

  # case 3
  var x3 = Tensor.new([[[2, 3], [2, 4]], [[7, 8], [9, 6]]])
  assert_equal([2, 2, 2], x3.shape)

  var y3_0 = ag.Sum(x3, 0)
  assert_equal([2, 2], y3_0.shape)
  assert_equal([9.0, 11.0, 11.0, 10.0], y3_0.data)

  x3.ClearGrad()
  ag.Backward(y3_0)
  var gx3: Tensor = x3.grad

  assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], gx3.data)
  assert_equal([2, 2, 2], gx3.shape)

  # case 4
  var y3_1 = ag.Sum(x3, 0, 1)
  assert_equal([1, 2, 2], y3_1.shape)
  assert_equal([9.0, 11.0, 11.0, 10.0], y3_1.data)

  x3.ClearGrad()
  ag.Backward(y3_1)
  gx3 = x3.grad

  assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], gx3.data)
  assert_equal([2, 2, 2], gx3.shape)

  # case 5
  var y3_4 = ag.Sum(x3, [0, 1])
  assert_equal([2], y3_4.shape)
  assert_equal([20.0, 21.0], y3_4.data)

  x3.ClearGrad()
  ag.Backward(y3_4)
  gx3 = x3.grad

  assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], gx3.data)
  assert_equal([2, 2, 2], gx3.shape)

  # case 6
  var y3_5 = ag.Sum(x3, [0, 1], 1)
  assert_equal([1, 1, 2], y3_5.shape)
  assert_equal([20.0, 21.0], y3_5.data)

  x3.ClearGrad()
  ag.Backward(y3_5)
  gx3 = x3.grad

  assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], gx3.data)
  assert_equal([2, 2, 2], gx3.shape)

  # case 7
  var y3_6 = ag.Sum(x3, [1, 2])
  assert_equal([2], y3_6.shape)
  assert_equal([11.0, 30.0], y3_6.data)

  x3.ClearGrad()
  ag.Backward(y3_6)
  gx3 = x3.grad

  assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], gx3.data)
  assert_equal([2, 2, 2], gx3.shape)

  # case 8
  var y3_7 = ag.Sum(x3, [1, 2], 1)
  assert_equal([2, 1, 1], y3_7.shape)
  assert_equal([11.0, 30.0], y3_7.data)

  x3.ClearGrad()
  ag.Backward(y3_7)
  gx3 = x3.grad

  assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], gx3.data)
  assert_equal([2, 2, 2], gx3.shape)
enddef


def TestBroadcastTo()
  var x1 = Tensor.new([20])
  var y1 = ag.BroadcastTo(x1, [2, 2, 3])

  var y1_expect = [
    20.0, 20.0, 20.0,
    20.0, 20.0, 20.0,
    20.0, 20.0, 20.0,
    20.0, 20.0, 20.0
    ]
  assert_equal(y1_expect, y1.data)

  assert_equal([2, 2, 3], y1.shape)

  ag.Backward(y1)
  var gx1: Tensor = x1.grad
  assert_equal([12.0], gx1.data)
  assert_equal([1], x1.shape)


  var x2 = ag.Uniform(10.0, 100.0, [2, 3])
  var y2 = ag.BroadcastTo(x2, [2, 4, 2, 3])
  assert_equal([2, 4, 2, 3], y2.shape)
  assert_equal(flattennew(repeat(x2.data, 8)), y2.data)

  ag.Backward(y2)
  var gx2: Tensor = x2.grad
  assert_equal([2, 3], gx2.shape)
  assert_equal([8.0, 8.0, 8.0, 8.0, 8.0, 8.0], gx2.data)


  var x3 = ag.Uniform(10.0, 100.0, [1, 2, 3])
  var y3 = ag.BroadcastTo(x3, [3, 2, 3])
  assert_equal([3, 2, 3], y3.shape)
  assert_equal(flattennew(repeat(x3.data, 3)), y3.data)

  ag.Backward(y3)
  var gx3: Tensor = x3.grad
  assert_equal([1, 2, 3], gx3.shape)
  assert_equal([3.0, 3.0, 3.0, 3.0, 3.0, 3.0], gx3.data)


  var x4 = ag.Uniform(10.0, 100.0, [1, 1, 2])
  var y4 = ag.BroadcastTo(x4, [3, 4, 2])
  assert_equal([3, 4, 2], y4.shape)
  assert_equal(flattennew(repeat(x4.data, 12)), y4.data)

  ag.Backward(y4)
  var gx4: Tensor = x4.grad
  assert_equal([1, 1, 2], gx4.shape)
  assert_equal([12.0, 12.0], gx4.data)


  var x5 = ag.Randn(1, 3, 2, 1)
  var y5 = ag.BroadcastTo(x5, [1, 3, 2, 2])
  assert_equal([1, 3, 2, 2], y5.shape)

  ag.Backward(y5)
  var gx5: Tensor = x5.grad
  assert_equal([1, 3, 2, 1], gx5.shape)
  assert_equal([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], gx5.data)


  var x6 = ag.Randn(3)
  var y6 = ag.BroadcastTo(x6, [1, 4, 3])
  assert_equal([1, 4, 3], y6.shape)

  ag.Backward(y6)
  var gx6: Tensor = x6.grad
  assert_equal([3], gx6.shape)
  assert_equal([4.0, 4.0, 4.0], gx6.data)


  var x7 = ag.Randn(2, 3, 1, 1)
  var y7 = ag.BroadcastTo(x7, [2, 3, 5, 6])
  assert_equal([2, 3, 5, 6], y7.shape)

  ag.Backward(y7)
  var gx7: Tensor = x7.grad
  assert_equal([2, 3, 1, 1], gx7.shape)
  assert_equal([30.0, 30.0, 30.0, 30.0, 30.0, 30.0], gx7.data)
enddef


def TestTranspose()
  var x1 = Tensor.new([[1, 2, 3], [4, 5, 6]])
  assert_equal([2, 3], x1.shape)

  var y1 = ag.Transpose(x1)
  assert_equal([3, 2], y1.shape)
  assert_equal([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], y1.data)

  ag.Backward(y1)
  var gx1: Tensor = x1.grad
  assert_equal([2, 3], gx1.shape)
  assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], gx1.data)
enddef


def TestReshape()
  var x1 = ag.Randn(2, 3)
  assert_equal([2, 3], x1.shape)

  var y1 = ag.Reshape(x1, [1, 6])
  assert_equal([1, 6], y1.shape)
  assert_equal(x1.data, y1.data)

  ag.Backward(y1)
  var gx1: Tensor = x1.grad
  assert_equal([2, 3], gx1.shape)
  assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], gx1.data)

  var y2 = ag.Reshape(x1, [3, 2])
  assert_equal([3, 2], y2.shape)
  assert_equal(x1.data, y2.data)

  var y3 = ag.Reshape(x1, [6, 1])
  assert_equal([6, 1], y3.shape)
  assert_equal(x1.data, y3.data)
enddef


export def RunTestSuite()
  TestSum()
  TestBroadcastTo()
  TestTranspose()
  TestReshape()
enddef
