vim9script

import '../core/callfunc.vim'
import '../core/engine.vim'
import '../core/function.vim'
import '../core/tensor.vim'

import './mul.vim'

var Function = function.Function
var HasCallableNode = function.HasCallableNode


export def FloatMax(list_obj: list<float>): float
  var max = 0.0
  for x in list_obj
    if max < x
      max = x
    endif
  endfor
  return max
enddef


class MaxFunction extends Function implements HasCallableNode
  public this.name: string
  public this.gen: number
  public this.id: number
  public this.inputs: list<tensor.Tensor>
  public this.outputs: list<tensor.Tensor>

  def new()
    super.Init('max')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      tensor.Tensor.new([FloatMax(xs[0].data)], [1])
    ]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    var x = this.inputs[0]
    var y = this.outputs[0]
    var gx_mask = engine.Elementwise(
      [x, y], (lhs, rhs): float => lhs == rhs ? 1.0 : 0.0)
    var gx = mul.Mul(gys[0], gx_mask)
    return [gx]
  enddef
endclass


export def Max(x: any): tensor.Tensor
  return callfunc.CallFunction(MaxFunction.new(), x)
enddef
