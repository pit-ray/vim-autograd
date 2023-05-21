vim9script

import '../core/callfunc.vim'
import '../core/engine.vim'
import '../core/function.vim'
import '../core/tensor.vim'

import './mul.vim'
import './pow.vim'
import './sub.vim'

var Function = function.Function
var HasCallableNode = function.HasCallableNode


class TanhFunction extends Function implements HasCallableNode
  public this.name: string
  public this.gen: number
  public this.id: number
  public this.inputs: list<tensor.Tensor>
  public this.outputs: list<tensor.Tensor>

  def new()
    super.Init('tanh')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      engine.Elementwise(xs, (hs): float => tanh(hs))
    ]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    var y = this.outputs[0]
    return [
      mul.Mul(gys[0], sub.Sub(1, pow.Pow(y, 2)))
    ]
  enddef
endclass


export def Tanh(x: any): tensor.Tensor
  return callfunc.CallFunction(TanhFunction.new(), x)
enddef