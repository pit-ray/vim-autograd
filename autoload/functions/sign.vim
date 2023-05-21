vim9script

import '../core/callfunc.vim'
import '../core/engine.vim'
import '../core/function.vim'
import '../core/tensor.vim'

import './mul.vim'

var Function = function.Function
var HasCallableNode = function.HasCallableNode


class SignFunction extends Function implements HasCallableNode
  public this.name: string
  public this.gen: number
  public this.id: number
  public this.inputs: list<tensor.Tensor>
  public this.outputs: list<tensor.Tensor>

  def new()
    super.Init('sign')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      engine.Elementwise(
        xs, (hs): float => hs > 0.0 ? 1.0 : (hs < -1.0 ? -1.0 : 0.0)
      )
    ]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    return [mul.Mul(gys[0], 0.0)]
  enddef
endclass


export def Sign(x: any): tensor.Tensor
  return callfunc.CallFunction(SignFunction.new(), x)
enddef
