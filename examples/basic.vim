vim9script

import '../autoload/autograd.vim' as ag
const Tensor = ag.Tensor


def F(x: Tensor): Tensor
  # y = x^5 - 2x^3
  var y = ag.Sub(
    ag.Pow(x, 5),
    ag.Mul(2, ag.Pow(x, 3)))
  return y
enddef


def Main()
  var x = Tensor.new(2.0)

  var y = F(x)
  ag.Backward(y)

  var x_grad: Tensor = x.grad
  echo x_grad.data

  x.SetName('x')
  y.SetName('y')
  ag.DumpGraph(y, '.autograd/example1.png')
enddef

Main()
