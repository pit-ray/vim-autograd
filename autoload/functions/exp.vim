vim9script

import '../core/callfunc.vim'
import '../core/engine.vim'
import '../core/function.vim'
import '../core/tensor.vim'

import './mul.vim'

var Function = function.Function


class ExpFunction extends Function
  def new()
    super.Init('exp')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      engine.Elementwise(xs, (hs): float => exp(hs))
    ]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    var y = this.outputs[0]
    return [mul.Mul(gys[0], y)]
  enddef
endclass


export def Exp(x: any): tensor.Tensor
  return callfunc.CallFunction(ExpFunction.new(), x)
enddef
