vim9script

import '../core/callfunc.vim'
import '../core/engine.vim'
import '../core/function.vim'
import '../core/tensor.vim'

import './mul.vim'
import './pow.vim'
import './sum_to.vim'

var Function = function.Function
var HasCallableNode = function.HasCallableNode


class DivFunction extends Function implements HasCallableNode
  public this.name: string
  public this.gen: number
  public this.id: number
  public this.inputs: list<tensor.Tensor>
  public this.outputs: list<tensor.Tensor>

  def new()
    super.Init('div')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      engine.Elementwise(xs, (lhs, rhs): float => lhs / rhs)
    ]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    var x0 = this.inputs[0]
    var x1 = this.inputs[1]

    var gx0 = Div(gys[0], x1)

    # gx1 = gy * -(x0 / x1 ** 2)
    var gx1 = mul.Mul(
      gys[0],
      mul.Mul(Div(x0, pow.Pow(x1, 2)), -1))

    if x0.shape == x1.shape
      return [gx0, gx1]
    endif

    return [
      sum_to.SumTo(gx0, x0.shape),
      sum_to.SumTo(gx1, x1.shape)
    ]
  enddef
endclass


export def Div(x0: any, x1: any): tensor.Tensor
  return callfunc.CallFunction(DivFunction.new(), x0, x1)
enddef
