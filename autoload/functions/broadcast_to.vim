vim9script

import '../core/callfunc.vim'
import '../core/function.vim'
import '../core/tensor.vim'
import '../utils/matrix.vim'
import '../utils/system.vim'

import './sum_to.vim'

var Function = function.Function


class BroadcastToFunction extends Function
  this._shape: list<number>

  def new()
    super.Init('broadcast_to')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    var x = xs[0]
    var x_dim = x.Dim()

    if x_dim > len(this._shape)
      system.Error(
        'cannot broadcast the array of ' ..
        string(x.shape) .. ' to ' .. string(this._shape))
      return []
    endif

    var size: number = matrix.ShapeToSize(this._shape)

    # left side broadcast
    var right_subshape = matrix.SqueezeLeftShape(x.shape)
    if right_subshape == [1]
      return [tensor.Tensor.new(repeat(x.data, size), this._shape)]
    endif

    if this._shape[-len(right_subshape) :] == right_subshape
      var ss: number = x.Numel()
      var repeat = float2nr(size / ss)
      return [tensor.Tensor.new(repeat(x.data, repeat), this._shape)]
    endif

    # right side broadcast
    var left_subshape = matrix.SqueezeRightShape(x.shape)
    if this._shape[: len(left_subshape) - 1] == left_subshape
      var repeat = float2nr(size / len(x.data))
      return [
        tensor.Tensor.new(
          flattennew(mapnew(x.data, (_, v) => repeat([v], repeat))),
          this._shape)
      ]
    endif

    system.Error(
      'cannot broadcast array of shape ' ..
      string(x.shape) .. ' into ' .. string(this._shape))
    return []
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      sum_to.SumTo(gys[0], this.inputs[0].shape)
    ]
  enddef

  def SetShape(shape: list<number>)
    this._shape = shape
  enddef
endclass


export def BroadcastTo(x: any, shape: list<number>): tensor.Tensor
  var xt: tensor.Tensor = tensor.AsTensor(x)
  if xt.shape == shape
    return xt
  endif

  var fn = BroadcastToFunction.new()
  fn.SetShape(shape)
  return callfunc.CallFunction(fn, x)
enddef
