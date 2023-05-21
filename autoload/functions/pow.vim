vim9script

import '../core/callfunc.vim'
import '../core/engine.vim'
import '../core/function.vim'
import '../core/tensor.vim'

import './log.vim'
import './mul.vim'
import './sub.vim'
import './sum_to.vim'

var Function = function.Function
var HasCallableNode = function.HasCallableNode


class PowFunction extends Function implements HasCallableNode
  public this.name: string
  public this.gen: number
  public this.id: number
  public this.inputs: list<tensor.Tensor>
  public this.outputs: list<tensor.Tensor>

  def new()
    super.Init('pow')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      engine.Elementwise(xs, (lhs, rhs): float => pow(lhs, rhs))
    ]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    var x = this.inputs[0]
    var c = this.inputs[1]
    var y = this.outputs[0]

    # gx = gy * c * x**(c - 1)
    var gx = mul.Mul(
      gys[0], mul.Mul(c, Pow(x, sub.Sub(c, 1))))

    # gc = gy * y * log(x)
    var gc = mul.Mul(
      gys[0], mul.Mul(y, log.Log(x)))

    if x.shape == c.shape
      return [gx, gc]
    endif

    return [
      sum_to.SumTo(gx, x.shape),
      sum_to.SumTo(gc, c.shape)
    ]
  enddef
endclass


export def Pow(x: any, c: any): tensor.Tensor
  var fn = PowFunction.new()
  return callfunc.CallFunction(fn, x, c)
enddef


export def Sqrt(x: any): tensor.Tensor
  return Pow(x, 0.5)
enddef
