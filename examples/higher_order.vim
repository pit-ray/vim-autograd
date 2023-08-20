vim9script
scriptencoding utf-8

import '../autoload/autograd.vim' as ag
const Tensor = ag.Tensor


def F(x: Tensor): Tensor
  # y = x^5 - 2x^3 + 4x^2 + 6x + 5
  var t1 = ag.Pow(x, 5)
  var t2 = ag.Mul(-2, ag.Pow(x, 3))
  var t3 = ag.Mul(4, ag.Pow(x, 2))
  var t4 = ag.Mul(6, x)
  var t5 = 5
  var y = ag.Add(t1, ag.Add(t2, ag.Add(t3, ag.Add(t4, t5))))
  return y
enddef

def Main()
  var x = Tensor.new(2.0)
  var y = F(x)
  echo 'y  :' y.data

  x.SetName('x')
  y.SetName('y')
  ag.DumpGraph(y, '.autograd/example2.png')

  # gx1 = 5x^4 - 6x^2 + 8x + 6
  var gx1: Tensor = ag.Grad(y, x, true)
  echo 'gx1:' gx1.data

  # gx2 = 20x^3 - 12x + 8
  var gx2: Tensor = ag.Grad(gx1, x, true)
  echo 'gx2:' gx2.data

  # gx3 = 60x^2 - 12
  ag.Backward(gx2, true)
  var gx3: Tensor = x.grad
  echo 'gx3:' gx3.data
enddef

Main()
