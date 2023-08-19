vim9script

import '../core/callfunc.vim'
import '../core/engine.vim'
import '../core/function.vim'
import '../core/tensor.vim'

import './mul.vim'
import './sum_to.vim'

var Function = function.Function


class MaximumFunction extends Function
  def new()
    super.Init('maximum')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      engine.Elementwise(
        xs, (lhs, rhs): float => lhs >= rhs ? lhs : rhs)
    ]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    var x0 = this.inputs[0]
    var x1 = this.inputs[1]

    var gx0_mask = engine.Elementwise(
      [x0, x1], (lhs, rhs): float => lhs >= rhs ? 1.0 : 0.0)
    var gx1_mask = engine.Elementwise(
      [x0, x1], (lhs, rhs): float => lhs < rhs ? 1.0 : 0.0)

    var gx0 = mul.Mul(gys[0], gx0_mask)
    var gx1 = mul.Mul(gys[0], gx1_mask)

    return [
      sum_to.SumTo(gx0, x0.shape),
      sum_to.SumTo(gx1, x1.shape)
    ]
  enddef
endclass


export def Maximum(a: any, b: any): tensor.Tensor
  return callfunc.CallFunction(MaximumFunction.new(), a, b)
enddef
