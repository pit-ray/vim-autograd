vim9script

import '../autoload/autograd.vim' as ag

var Tensor = ag.Tensor


def GoldsteinPrice(inputs: list<any>): Tensor
  var x = ag.AsTensor(inputs[0])
  var y = ag.AsTensor(inputs[1])

  var t1 = ag.Pow(ag.Add(x, ag.Add(y, 1)), 2)
  var t2 = ag.Add(ag.Add(ag.Sub(ag.Add(ag.Add(ag.Mul(-14, x), 19), ag.Mul(3, ag.Pow(x, 2))), ag.Mul(14, y)), ag.Mul(x, ag.Mul(6, y))), ag.Mul(3, ag.Pow(y, 2)))
  var t3 = ag.Pow(ag.Sub(ag.Mul(x, 2), ag.Mul(3, y)), 2)
  var t4 = ag.Add(ag.Sub(ag.Add(ag.Add(ag.Add(ag.Mul(-32, x), 18), ag.Mul(12, ag.Pow(x, 2))), ag.Mul(48, y)), ag.Mul(x, ag.Mul(36, y))), ag.Mul(27, ag.Pow(y, 2)))
  return ag.Mul(ag.Add(ag.Mul(t1, t2), 1), ag.Add(ag.Mul(t3, t4), 30))
enddef


def TestGoldsteinPrice()
  var x = ag.Tensor.new(1.0)
  assert_equal([1.0], x.data)

  var y = ag.Tensor.new(1.0)
  assert_equal([1.0], y.data)

  var z = GoldsteinPrice([x, y])
  assert_equal([1876.0], z.data)

  ag.Backward(z)
  var xg: Tensor = x.grad
  var yg: Tensor = y.grad
  assert_equal([-5376.0], xg.data)
  assert_equal([8064.0], yg.data)
enddef


def TestGoldsteinPriceGradcheck()
  var x = ag.Uniform(0.0, 10.0, [2, 3])
  var y = ag.Uniform(0.0, 10.0, [2, 3])

  ag.GradCheck(GoldsteinPrice, [x, y])
enddef


export def RunTestSuite()
  TestGoldsteinPrice()
  TestGoldsteinPriceGradcheck()
enddef
