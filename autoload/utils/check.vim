vim9script

import '../core/backward.vim'
import '../core/context.vim'
import '../core/engine.vim'
import '../core/tensor.vim'

import '../functions/add.vim' as add_module
import '../functions/div.vim'
import '../functions/sub.vim'
import '../functions/sum_to.vim'


def IsClose(
    a: float,
    b: float,
    rtol: float = 0.00001,
    atol: float = 0.00000001): bool
  return abs(a - b) <= (atol + rtol * abs(b))
enddef


def AllClose(
    a: any,
    b: any,
    rtol: float = 0.00001,
    atol: float = 0.00000001): bool
  var results: tensor.Tensor = engine.Elementwise(
    [a, b], (x, y): float => IsClose(x, y, rtol, atol) ? 1.0 : -1.0)

  for x in results.data
    if x < 0.0
      return false
    endif
  endfor
  return true
enddef


export def NumericalGrad(Fn: func, x: tensor.Tensor): tensor.Tensor
  var eps = tensor.Tensor.new(0.000001)
  var dx = tensor.Tensor.new(eps.data[0] * 2)

  var x0 = sub.Sub(x, eps)
  var x1 = add_module.Add(x, eps)

  var y0 = Fn(x0)
  var y1 = Fn(x1)

  var dy = sub.Sub(y1, y0)
  return sum_to.SumTo(div.Div(dy, dx), x.shape)
enddef


export def GradCheck(Fn: func, inputs: list<any>)
  var y: tensor.Tensor = Fn(inputs)

  for x: tensor.Tensor in inputs
    x.ClearGrad()
  endfor
  backward.Backward(y)

  var grads: list<tensor.Tensor>
  for x: tensor.Tensor in inputs
    grads->add(x.grad)
  endfor

  context.NoGrad(() => {
    var input_num = len(inputs)
    for i in range(input_num)
      var before_args = i > 0 ? inputs[: i - 1] : []
      var after_args = i < input_num - 1 ? inputs[i + 1 :] : []

      var num_grad = NumericalGrad(
        (x) => Fn(before_args + [x] + after_args),
        inputs[i])

      assert_true(AllClose(grads[i], num_grad))
    endfor
  })
enddef
