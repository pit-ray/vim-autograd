vim9script

import '../core/callfunc.vim'
import '../core/engine.vim'
import '../core/function.vim'
import '../core/tensor.vim'

import './sum_to.vim'

var Function = function.Function


class MulFunction extends Function
  def new()
    super.Init('mul')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      engine.Elementwise(xs, (lhs, rhs): float => lhs * rhs)
    ]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    var x0 = this.inputs[0]
    var x1 = this.inputs[1]

    var gx0 = Mul(x1, gys[0])
    var gx1 = Mul(x0, gys[0])

    if x0.shape == x1.shape
      return [gx0, gx1]
    endif

    return [
      sum_to.SumTo(gx0, x0.shape),
      sum_to.SumTo(gx1, x1.shape)
    ]
  enddef
endclass


export def Mul(x0: any, x1: any): tensor.Tensor
  return callfunc.CallFunction(MulFunction.new(), x0, x1)
enddef
