vim9script

import '../core/callfunc.vim'
import '../core/function.vim'
import '../core/tensor.vim'
import '../utils/matrix.vim'
import '../utils/system.vim'

import './sum_to.vim'

const Function = function.Function


class TransposeFunction extends Function
  def new()
    super.Init('transpose')
  enddef

  def Forward(xs: list<tensor.Tensor>): list<tensor.Tensor>
    var x = xs[0]

    var dim = x.Dim()
    if dim > 2
      system.Error('transpose() is supported only for 1D-tensor and 2D-tensor.')
      return []
    endif

    if dim == 1
      return [x]
    endif

    var xd = x.data
    var out_data = matrix.CreateVector(len(xd))

    var n_i = x.shape[0]
    var n_j = x.shape[1]

    var n_j_range = range(n_j)
    for i in range(n_i)
      var buf = i * n_j
      for j in n_j_range
        out_data[j * n_i + i] = xd[buf + j]
      endfor
    endfor

    return [tensor.Tensor.new(out_data, [n_j, n_i])]
  enddef

  def Backward(gys: list<tensor.Tensor>): list<tensor.Tensor>
    return [
      Transpose(gys[0])
    ]
  enddef
endclass


export def Transpose(x: any): tensor.Tensor
  return callfunc.CallFunction(TransposeFunction.new(), x)
enddef
