vim9script

import '../core/callfunc.vim'
import '../core/engine.vim'
import '../core/function.vim'
import '../core/tensor.vim'

import './cos.vim'
import './mul.vim'

var Function = function.Function
var HasCallableNode = function.HasCallableNode


class SinFunction extends Function implements HasCallableNode
  public this.name: string
  public this.gen: number
  public this.id: number
  public this.inputs: list<tensor.Tensor>
  public this.outputs: list<tensor.Tensor>

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
