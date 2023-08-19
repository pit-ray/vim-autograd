vim9script

import '../core/callfunc.vim'
import '../core/engine.vim'
import '../core/function.vim'
import '../core/tensor.vim'

import './mul.vim'
import './sum_to.vim'

var Function = function.Function


class SubFunction extends Function
  def new()
    super.Init('sub')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      engine.Elementwise(xs, (lhs, rhs): float => lhs - rhs)
    ]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    var x0 = this.inputs[0]
    var x1 = this.inputs[1]

    var gx0 = gys[0]
    var gx1 = mul.Mul(gys[0], -1)

    if x0.shape == x1.shape
      return [gx0, gx1]
    endif

    return [
      sum_to.SumTo(gx0, x0.shape),
      sum_to.SumTo(gx1, x1.shape)
    ]
  enddef
endclass


export def Sub(x0: any, x1: any): tensor.Tensor
  return callfunc.CallFunction(SubFunction.new(), x0, x1)
enddef
