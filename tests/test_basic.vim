vim9script

import '../autoload/autograd.vim' as ag


def TestAdd()
  var F = (xs) => ag.Add(xs[0], xs[1])

  var x0 = ag.Uniform(0.0, 100.0, [2, 3])
  var x1 = ag.Uniform(0.0, 100.0, [2, 3])

  ag.GradCheck(F, [x0, x1])
enddef



def TestMul()
  var F = (xs) => ag.Mul(xs[0], xs[1])

  var x0 = ag.Uniform(0.0, 100.0, [2, 3])
  var x1 = ag.Uniform(0.0, 100.0, [2, 3])

  ag.GradCheck(F, [x0, x1])

  var x2 = ag.Uniform(0.0, 100.0, [2, 3])
  var x3 = ag.Tensor.new([10])
  ag.GradCheck(F, [x2, x3])
enddef


def TestSub()
  var F = (xs) => ag.Sub(xs[0], xs[1])

  var x0 = ag.Uniform(0.0, 100.0, [2, 3])
  var x1 = ag.Uniform(0.0, 100.0, [2, 3])

  ag.GradCheck(F, [x0, x1])
enddef


def TestDiv()
  var F = (xs) => ag.Div(xs[0], xs[1])

  var x0 = ag.Uniform(0.0, 100.0, [2, 3])
  var x1 = ag.Uniform(1.0, 100.0, [2, 3])

  ag.GradCheck(F, [x0, x1])
enddef


def TestPow()
  var F = (xs) => ag.Pow(xs[0], xs[1])

  var x0 = ag.Uniform(0.0, 10.0, [2, 3])
  var x1 = ag.Uniform(0.0, 10.0, [2, 3])

  ag.GradCheck(F, [x0, x1])
enddef


export def RunTestSuite()
  TestAdd()
  TestMul()
  TestSub()
  TestDiv()
  TestPow()
enddef
