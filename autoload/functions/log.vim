vim9script

import '../core/callfunc.vim'
import '../core/engine.vim'
import '../core/function.vim'
import '../core/tensor.vim'

import './div.vim'

const Function = function.Function


class LogFunction extends Function
  def new()
    super.Init('log')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      engine.Elementwise(xs, (hs): float => log(hs))
    ]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    var x = this.inputs[0]
    return [
      div.Div(gys[0], x)
    ]
  enddef
endclass


export def Log(x: any): tensor.Tensor
  return callfunc.CallFunction(LogFunction.new(), x)
enddef
