vim9script

import '../core/callfunc.vim'
import '../core/function.vim'
import '../core/tensor.vim'
import '../utils/matrix.vim'
import '../utils/system.vim'

import './broadcast_to.vim'
import './sum_to.vim'

var Function = function.Function
var HasCallableNode = function.HasCallableNode


class SumFunction extends Function implements HasCallableNode
  public this.name: string
  public this.gen: number
  public this.id: number
  public this.inputs: list<tensor.Tensor>
  public this.outputs: list<tensor.Tensor>

  this._axis: list<number>
  this._keepdims: bool = false

  def new()
    super.Init('sum')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    var x = xs[0]
    var x_dim: number = x.Dim()

    # all sum (e.g. (2, 3, 4) -> (1))
    if empty(this._axis) || x.Dim() == len(this._axis)
      var total: float = 0.0
      for val in x.data
        total += val
      endfor

      var data: list<float> = [total]
      var shape: list<number> = [1]

      if this._keepdims
        shape = shape->repeat(x_dim)
      endif

      var y = tensor.Tensor.new(data, shape)
      return [y]
    endif

    # left side sum (e.g. (2, 3, 4) -> (3, 4))
    if this._axis[0] == 0
      var reduced_shape = x.shape[len(this._axis) :]
      var sx = sum_to.LeftSideSumTo(x, reduced_shape)

      if this._keepdims
        sx.shape = repeat([1], len(this._axis)) + reduced_shape
      endif

      return [sx]
    endif

    # right side sum (e.g. (2, 3, 4) -> (2, 3)
    if (x_dim - 1) == this._axis[-1]
      var reduced_shape = x.shape[: -len(this._axis) - 1]
      var sx = sum_to.RightSideSumTo(x, reduced_shape)

      if this._keepdims
        sx.shape = reduced_shape + repeat([1], len(this._axis))
      endif
      return [sx]
    endif

    system.Error('intermediate or sparse axis sums are not supported.')
    return []
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      broadcast_to.BroadcastTo(gys[0], this.inputs[0].shape)
    ]
  enddef

  def SetAxis(axis: list<number>)
    this._axis = axis
  enddef

  def SetKeepdims(keepdims: bool)
    this._keepdims = keepdims
  enddef
endclass


export def Sum(
    x: tensor.Tensor,
    axis: any = [],
    keepdims: bool = false): tensor.Tensor
  var x_dim: number = x.Dim()
  var axis_list: list<number> = matrix.AsList(axis)
  map(axis_list, (_, v): number => v < 0 ? v + x_dim : v)
  map(axis_list, (_, v): number => v >= x_dim ? x_dim - 1 : v)

  var fn = SumFunction.new()
  fn.SetAxis(uniq(sort(axis_list)))
  fn.SetKeepdims(keepdims)
  return callfunc.CallFunction(fn, x)
enddef
