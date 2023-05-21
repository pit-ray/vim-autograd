vim9script

import './tensor.vim'
import './backward.vim'


export def Grad(
    output: tensor.Tensor,
    inputs: any,
    create_graph: bool = false,
    retain_outgrad: bool = false): any
  var xs: list<tensor.Tensor> = tensor.IsTensor(inputs) ? [inputs] : inputs

  var old_grads: list<any>
  for x in xs
    old_grads->add(x.grad)
    x.ClearGrad()
  endfor

  backward.Backward(output, create_graph, retain_outgrad)

  var grads = []
  for i in range(len(xs))
    grads->add(xs[i].grad)
    xs[i].grad = old_grads[i]
  endfor

  return len(grads) > 1 ? grads : grads[0]
enddef
