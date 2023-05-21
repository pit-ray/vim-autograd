vim9script

import '../autoload/autograd.vim' as ag


def TestNoGrad()
  var x = ag.Tensor.new(4.0)

  ag.NoGrad(() => {
    var y = ag.Sub(ag.Pow(ag.Mul(2, x), 3), 10)
    assert_equal([502.0], y.data)
  })
enddef


export def RunTestSuite()
  TestNoGrad()
enddef
