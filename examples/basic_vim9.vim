vim9script
scriptencoding utf-8

import '../autoload/autograd_vim9.vim' as ag

def F1(x: any): any
  # y = x^5 - 2x^3
  var y = ag.Sub(x.p(5), x.p(3).m(2))
  return y
enddef

def Main(): void
  var x = ag.Tensor(2.0)

  var y = F1(x)
  y.backward()

  echo x.grad.data

  var x.name = 'x'
  var y.name = 'y'
  ag.DumpGraph(y, '.autograd/example1.png')
enddef

Main()
