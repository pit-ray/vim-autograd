vim9script

import '../core/callfunc.vim'
import '../core/function.vim'
import '../core/tensor.vim'
import '../utils/system.vim'

import './transpose.vim'

var Function = function.Function
var HasCallableNode = function.HasCallableNode


class MatmulFunction extends Function implements HasCallableNode
  public this.name: string
  public this.gen: number
  public this.id: number
  public this.inputs: list<tensor.Tensor>
  public this.outputs: list<tensor.Tensor>

  this._x0_shape_fix: list<number>
  this._x1_shape_fix: list<number>

  def new()
    super.Init('matmul')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    var x0 = xs[0]
    var x1 = xs[1]

    var x0_dim = x0.Dim()
    var x1_dim = x1.Dim()

    if x0_dim > 2 || x1_dim > 2
      system.Error('inputs must be 2D-2D or 1D-1D.')
      return []
    endif

    var x0_shape = copy(x0.shape)
    var x1_shape = copy(x1.shape)
    this._x0_shape_fix = x0_shape
    this._x1_shape_fix = x1_shape

    # 1D-tensor is converted to 2D-tensor
    if x0_dim == 1
      x0_shape->insert(1, 0)
    endif
    if x1_dim == 1
      x1_shape->add(1)
    endif

    if x0_shape[1] != x1_shape[0]
      system.Error('axis 1 of left operand mismatchs axis 0 of right.')
      return []
    endif

    var n_i = x0_shape[0]
    var n_k = x0_shape[1]
    var n_j = x1_shape[1]

    var out: tensor.Tensor = tensor.Zeros([n_i, n_j])

    var od = out.data
    var d0 = x0.data
    var d1 = x1.data

    # 2D matrix product (ikj-algorithm)
    var n_k_range = range(n_k)
    var n_j_range = range(n_j)
    for i in range(n_i)
      for k in n_k_range
        var buf = d0[i * n_k + k]
        for j in n_j_range
          od[i * n_j + j] += buf * d1[k * n_j + j]
        endfor
      endfor
    endfor

    # If one is 1D, output in 1D
    if x0_dim == 1
      out.shape->remove(0)
    elseif x1_dim == 1
      out.shape->remove(1)
    endif

    return [out]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    var x0 = this.inputs[0]
    var x1 = this.inputs[1]
    var gy = gys[0]

    var x0_shape_raw = x0.shape
    var x1_shape_raw = x1.shape

    # temporarily restores the shape of x when y is calculated.
    x0.shape = this._x0_shape_fix
    x1.shape = this._x1_shape_fix

    var gx0 = Matmul(gy, transpose.Transpose(x1))
    var gx1 = Matmul(transpose.Transpose(x0), gy)

    # return to the original shape
    x0.shape = x0_shape_raw
    x1.shape = x1_shape_raw

    return [gx0, gx1]
  enddef
endclass


export def Matmul(a: any, b: any): tensor.Tensor
  return callfunc.CallFunction(MatmulFunction.new(), a, b)
enddef
