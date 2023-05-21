vim9script

import '../autoload/autograd.vim' as ag

var Tensor = ag.Tensor


def TestHigerOrderDifferential()
  var x = Tensor.new(2.0)
  assert_equal([2.0], x.data)

  # y = x^5 - 3*x^3 + 1
  var y: Tensor = ag.Add(ag.Sub(ag.Pow(x, 5), ag.Mul(ag.Pow(x, 3), 3)), 1)
  assert_equal([9.0], y.data)

  var gx: Tensor = ag.Grad(y, x, true)
  assert_equal([44.0], gx.data)

  gx = ag.Grad(gx, x, true)
  assert_equal([124.0], gx.data)

  gx = ag.Grad(gx, x, true)
  assert_equal([222.0], gx.data)

  gx = ag.Grad(gx, x, true)
  assert_equal([240.0], gx.data)

  ag.Backward(gx, true)
  gx = x.grad
  assert_equal([120.0], gx.data)
enddef


export def RunTestSuite()
  TestHigerOrderDifferential()
enddef
