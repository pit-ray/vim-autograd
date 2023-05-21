vim9script

import '../core/callfunc.vim'
import '../core/engine.vim'
import '../core/function.vim'
import '../core/tensor.vim'

import './mul.vim'
import './sin.vim'

var Function = function.Function
var HasCallableNode = function.HasCallableNode


class CosFunction extends Function implements HasCallableNode
  public this.name: string
  public this.gen: number
  public this.id: number
  public this.inputs: list<tensor.Tensor>
  public this.outputs: list<tensor.Tensor>

  def new()
    super.Init('cos')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      engine.Elementwise(xs, (hs): float => cos(hs))
    ]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    var x = this.inputs[0]
    return [
      mul.Mul(gys[0], mul.Mul(sin.Sin(x), -1))
    ]
  enddef
endclass


export def Cos(x: any): tensor.Tensor
  return callfunc.CallFunction(CosFunction.new(), x)
enddef
