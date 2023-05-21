vim9script

import '../core/callfunc.vim'
import '../core/function.vim'
import '../core/tensor.vim'
import '../utils/matrix.vim'
import '../utils/system.vim'

import './broadcast_to.vim'

var Function = function.Function
var HasCallableNode = function.HasCallableNode


export def LeftSideSumTo(
    x: tensor.Tensor,
    shape: list<number>): tensor.Tensor
  var y: tensor.Tensor = tensor.Zeros(shape)

  var xd = x.data
  var yd = y.data

  var x_size = len(xd)
  var y_size = len(yd)

  for i in range(x_size / y_size)
    var base = i * y_size
    for j in range(y_size)
      yd[j] += xd[base + j]
    endfor
  endfor
  return y
enddef


export def RightSideSumTo(
    x: tensor.Tensor,
    shape: list<number>): tensor.Tensor
  var y: tensor.Tensor = tensor.Zeros(shape)

  var xd = x.data
  var yd = y.data

  var x_size = len(xd)
  var y_size = len(yd)

  var block_size = x_size / y_size
  for i in range(y_size)
    var base = block_size * i
    for j in range(block_size)
      yd[i] += xd[base + j]
    endfor
  endfor
  return y
enddef


class SumToFunction extends Function implements HasCallableNode
  public this.name: string
  public this.gen: number
  public this.id: number
  public this.inputs: list<tensor.Tensor>
  public this.outputs: list<tensor.Tensor>

  this._shape: list<number>

  def new()
    super.Init('sum_to')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    var x = xs[0]
    var y = tensor.Zeros(this._shape)

    var y_dim = len(this._shape)

    # left side sum
    var right_subshape = matrix.SqueezeLeftShape(this._shape)
    if right_subshape == [1] 
        || x.shape[-len(right_subshape) :] == right_subshape
      return [LeftSideSumTo(x, this._shape)]
    endif

    # right side sum
    var left_subshape = matrix.SqueezeRightShape(this._shape)
    if x.shape[: len(left_subshape) - 1] == left_subshape
      return [RightSideSumTo(x, this._shape)]
    endif

    system.Error(
      'cannot sum from ' .. 
      string(x.shape) .. ' into ' .. string(this._shape))
    return []
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    return [broadcast_to.BroadcastTo(gys[0], this.inputs[0].shape)]
  enddef

  def SetShape(shape: list<number>)
    this._shape = shape
  enddef
endclass


export def SumTo(x: any, shape: list<number>): tensor.Tensor
  var xt: tensor.Tensor = tensor.AsTensor(x)
  if xt.shape == shape
    return xt
  endif

  var fn = SumToFunction.new()
  fn.SetShape(shape)
  return callfunc.CallFunction(fn, x)
enddef
