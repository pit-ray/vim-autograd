vim9script

import '../autoload/autograd.vim' as ag

var Tensor = ag.Tensor


def TestLog()
  var F = (xs) => ag.Log(xs[0])
  var x = ag.Uniform(0.0, 100.0, [2, 3])
  ag.GradCheck(F, [x])
enddef


def TestExp()
  var F = (xs) => ag.Exp(xs[0])
  var x = ag.Uniform(0.0, 10.0, [2, 3])
  ag.GradCheck(F, [x])
enddef


def TestSin()
  var F = (xs) => ag.Sin(xs[0])
  var x = ag.Uniform(0.0, ag.Pi() * 2, [2, 3])
  ag.GradCheck(F, [x])
enddef


def TestCos()
  var F = (xs) => ag.Cos(xs[0])
  var x = ag.Uniform(0.0, ag.Pi() * 2, [2, 3])
  ag.GradCheck(F, [x])
enddef


def TestTanh()
  var F = (xs) => ag.Tanh(xs[0])
  var x = ag.Rand(2, 3)
  ag.GradCheck(F, [x])
enddef


def TestAbs()
  var x1 = ag.Tensor.new([0.7, -1.2, 0.0, 2.3])

  var y1 = ag.Abs(x1)
  assert_equal([0.7, 1.2, 0.0, 2.3], y1.data)

  ag.Backward(y1)
  var gx1: Tensor = x1.grad
  assert_equal([1.0, -1.0, 0.0, 1.0], gx1.data)
enddef



def TestSign()
  var x1 = ag.Tensor.new([0.2, 1.5, -5.6, 0.0])

  var y1 = ag.Sign(x1)
  assert_equal([1.0, 1.0, -1.0, 0.0], y1.data)

  ag.Backward(y1)
  var gx1: Tensor = x1.grad
  assert_equal([0.0, 0.0, 0.0, 0.0], gx1.data)
enddef


def TestMatmul()
  var x0 = ag.Tensor.new([[2, 4, 5], [4, 5, 6]])
  assert_equal([2, 3], x0.shape)

  var x1 = ag.Tensor.new([[5, 7], [7, 8], [5, 7]])
  assert_equal([3, 2], x1.shape)

  var y0 = ag.Matmul(x0, x1)
  assert_equal([2, 2], y0.shape)
  assert_equal([63.0, 81.0, 85.0, 110.0], y0.data)

  var y1 = ag.Matmul(x1, x0)
  assert_equal([3, 3], y1.shape)
  assert_equal([
    38.0, 55.0, 67.0,
    46.0, 68.0, 83.0,
    38.0, 55.0, 67.0], y1.data)

  ag.Backward(y0)
  var gx0: Tensor = x0.grad
  var gx1: Tensor = x1.grad

  assert_equal([
    12.0, 15.0, 12.0,
    12.0, 15.0, 12.0], gx0.data)
  assert_equal([2, 3], gx0.shape)

  assert_equal([
    6.0, 6.0, 9.0,
    9.0, 11.0, 11.0], gx1.data)
  assert_equal([3, 2], gx1.shape)

  var x2 = ag.Tensor.new([2, 4, 5])
  var x3 = ag.Tensor.new([5, 7, 9])
  var y2 = ag.Matmul(x2, x3)
  assert_equal([1], y2.shape)
  assert_equal([83.0], y2.data)

  ag.Backward(y2)
  var gx2: Tensor = x2.grad
  var gx3: Tensor = x3.grad

  assert_equal([5.0, 7.0, 9.0], gx2.data)
  assert_equal([3], gx2.shape)

  assert_equal([2.0, 4.0, 5.0], gx3.data)
  assert_equal([3], gx3.shape)
enddef


def TestMax()
  # case 1
  var x1 = ag.Tensor.new([[2.0, 2.1, 1.9], [3.1, 2.0, 3.0]])
  var y1 = ag.Max(x1)
  assert_equal([3.1], y1.data)
  assert_equal([1], y1.shape)

  ag.Backward(y1)
  var gx1: Tensor = x1.grad
  assert_equal([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], gx1.data)
  assert_equal([2, 3], gx1.shape)

  # case 2
  var x2 = ag.Tensor.new([5, 7, 9])
  var y2 = ag.Pow(ag.Max(x2), 2)
  assert_equal([81.0], y2.data)
  assert_equal([1], y2.shape)

  ag.Backward(y2)
  var gx2: Tensor = x2.grad
  assert_equal([0.0, 0.0, 18.0], gx2.data)
  assert_equal([3], gx2.shape)
enddef


def TestMaximum()
  # case 1
  var x0 = ag.Tensor.new([1, 2, -1])
  var x1 = ag.Tensor.new([3, 0, 4])
  var y0 = ag.Maximum(x0, x1)
  assert_equal([3.0, 2.0, 4.0], y0.data)
  assert_equal([3], y0.shape)

  ag.Backward(y0)
  var gx0: Tensor = x0.grad
  var gx1: Tensor = x1.grad
  assert_equal([0.0, 1.0, 0.0], gx0.data)
  assert_equal([3], gx0.shape)
  assert_equal([1.0, 0.0, 1.0], gx1.data)
  assert_equal([3], gx1.shape)

  # case 2
  var x2 = ag.Tensor.new([4, -1, 9])
  var x3 = ag.Tensor.new([5])
  var y2 = ag.Maximum(x2, x3)
  assert_equal([5.0, 5.0, 9.0], y2.data)
  assert_equal([3], y2.shape)

  ag.Backward(y2)
  var gx2: Tensor = x2.grad
  var gx3: Tensor = x3.grad
  assert_equal([0.0, 0.0, 1.0], gx2.data)
  assert_equal([3], gx2.shape)
  assert_equal([2.0], gx3.data)
  assert_equal([1], gx3.shape)

  # case 3
  var x4 = ag.Tensor.new([[2, 4, 5], [6, -1, 5]])
  var x5 = ag.Tensor.new([3, 2, 5])
  var y4  = ag.Maximum(x4, x5)
  assert_equal([3.0, 4.0, 5.0, 6.0, 2.0, 5.0], y4.data)
  assert_equal([2, 3], y4.shape)

  ag.Backward(y4)
  var gx4: Tensor = x4.grad
  var gx5: Tensor = x5.grad
  assert_equal([0.0, 1.0, 1.0, 1.0, 0.0, 1.0], gx4.data)
  assert_equal([2, 3], gx4.shape)
  assert_equal([1.0, 1.0, 0.0], gx5.data)
  assert_equal([3], gx5.shape)
enddef


export def RunTestSuite()
  TestLog()
  TestExp()
  TestSin()
  TestCos()
  TestTanh()
  TestAbs()
  TestSign()
  TestMax()
  TestMatmul()
  TestMaximum()
enddef
