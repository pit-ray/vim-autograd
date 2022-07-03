vim9script
scriptencoding utf-8

import '../autoload/autograd_vim9.vim' as ag

def F(x: any): any
  # y = x^5 - 2x^3 + 4x^2 + 6x + 5
  var t1 = x.p(5)
  var t2 = x.p(3).m(2).n()
  var t3 = x.p(2).m(4)
  var t4 = x.m(6)
  var t5 = 5
  var y = t1.a(t2).a(t3).a(t4).a(t5)
  return y
enddef

def Main(): void
  var x = ag.Tensor(2.0)
  var y = F(x)
  echo 'y  :' y.data

  # gx1 = 5x^4 - 6x^2 + 8x + 6
  var gx1 = ag.Grad(y, x, 1)
  echo 'gx1:' gx1.data

  # gx2 = 20x^3 - 12x + 8
  var gx2 = ag.Grad(gx1, x, 1)
  echo 'gx2:' gx2.data

  # gx3 = 60x^2 - 12
  gx2.backward(1)
  echo 'gx3:' x.grad.data
enddef

Main()
