vim9script

import '../core/callfunc.vim'
import '../core/engine.vim'
import '../core/function.vim'
import '../core/tensor.vim'

import './cos.vim'
import './mul.vim'

const Function = function.Function


class SinFunction extends Function
  def new()
    super.Init('sin')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      engine.Elementwise(xs, (hs): float => sin(hs))
    ]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    var x = this.inputs[0]
    return [
      mul.Mul(gys[0], cos.Cos(x))
    ]
  enddef
endclass


export def Sin(x: any): tensor.Tensor
  return callfunc.CallFunction(SinFunction.new(), x)
enddef
