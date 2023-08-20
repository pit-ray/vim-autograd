vim9script

import '../core/callfunc.vim'
import '../core/function.vim'
import '../core/tensor.vim'
import '../utils/matrix.vim'
import '../utils/system.vim'

import './sum_to.vim'

const Function = function.Function


class ReshapeFunction extends Function
  this._shape: list<number>

  def new()
    super.Init('reshape')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    var x = xs[0]

    if matrix.ShapeToSize(this._shape) != len(x.data)
      system.Error(
        'Cannot reshape array of size ' ..
        len(x.data) .. ' into ' .. string(this._shape))
      return []
    endif

    return [tensor.Tensor.new(x.data, this._shape)]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    return [Reshape(gys[0], this.inputs[0].shape)]
  enddef

  def SetShape(shape: list<number>)
    this._shape = shape
  enddef
endclass


export def Reshape(x: tensor.Tensor, shape: list<number>): tensor.Tensor
  if x.shape == shape
    return x
  endif

  var fn = ReshapeFunction.new()
  fn.SetShape(shape)
  return callfunc.CallFunction(fn, x)
enddef


export def Flatten(x: any): tensor.Tensor
  return Reshape(x, [x.Numel()])
enddef
