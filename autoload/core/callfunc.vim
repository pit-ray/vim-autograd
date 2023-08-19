vim9script

import './context.vim'
import './tensor.vim'
import './function.vim'


def FindMaxGen(xs: list<tensor.Tensor>): number
  var gens: list<number>
  for x in xs
    gens->add(x.gen)
  endfor
  return max(gens)
enddef


export def CallFunction(
    fn: function.Function,
    ...any_inputs: list<any>): any
  var inputs: list<tensor.Tensor>
  for input in any_inputs
    inputs->add(tensor.AsTensor(input))
  endfor

  var outputs: list<tensor.Tensor> = fn.Forward(inputs)

  if context.IsBackpropEnabled()
    fn.gen = FindMaxGen(inputs)

    for output in outputs
      output.parent_fn = fn
      output.gen = fn.gen + 1
    endfor

    fn.inputs = inputs
    fn.outputs = outputs
  endif

  return len(outputs) > 1 ? outputs : outputs[0]
enddef
