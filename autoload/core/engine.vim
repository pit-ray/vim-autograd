vim9script

import './context.vim'
import './tensor.vim'
import '../functions/broadcast_to.vim'


export def Elementwise(
    inputs: list<any>,
    ElementalFunc: func: float,
    out: any = null): tensor.Tensor

  if len(inputs) == 1
    var x: tensor.Tensor = inputs[0]
    var dst: tensor.Tensor = out == null ? tensor.ZerosLike(x) : out
    for i in range(x.Numel())
      dst.data[i] = ElementalFunc(x.data[i])
    endfor
    return dst
  endif

  var x0: tensor.Tensor = inputs[0]
  var x1: tensor.Tensor = inputs[1]

  context.NoGrad(() => {
    var x0_dim = x0.Dim()
    var x1_dim = x1.Dim()

    if x0_dim > x1_dim
      x1 = broadcast_to.BroadcastTo(x1, x0.shape)
    elseif x0_dim < x1_dim
      x0 = broadcast_to.BroadcastTo(x0, x1.shape)
    else
      if len(x0.data) > len(x1.data)
        x1 = broadcast_to.BroadcastTo(x1, x0.shape)
      else
        x0 = broadcast_to.BroadcastTo(x0, x1.shape)
      endif
    endif
  })

  var dst: tensor.Tensor = out == null ? tensor.ZerosLike(x0) : out
  for i in range(dst.Numel())
    dst.data[i] = ElementalFunc(x0.data[i], x1.data[i])
  endfor
  return dst
enddef
