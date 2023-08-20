vim9script

import './context.vim'
import './tensor.vim'
import './function.vim'
import '../functions/add.vim' as add_node


export def Backward(
    x: tensor.Tensor,
    create_graph: bool = false,
    retain_outgrad: bool = false)

  if x.grad == null
    x.grad = tensor.OnesLike(x)
  endif

  if x.parent_fn == null
    return
  endif

  # If create_graph is false, does not create graph.
  const Ctx = create_graph ? (F) => F() : context.NoGrad

  var funcs = [x.parent_fn]
  var scanned_fnids = []
  while len(funcs) > 0
    var fn: function.Function = funcs->remove(-1)

    var gys: list<tensor.Tensor> = []
    for output in fn.outputs
      gys->add(output.grad)
    endfor

    Ctx(() => {
      var gxs = fn.Backward(gys)

      var input_gradids: list<number> = []
      var input_num = len(gxs)
      for i in range(input_num)
        var input = fn.inputs[i]
        if input.grad == null
          input.grad = gxs[i]
        else
          input.grad = add_node.Add(input.grad, gxs[i])
        endif

        var input_grad: tensor.Tensor = input.grad
        input_gradids->add(input_grad.id)

        # It prevents multiple calling backward() of the same function.
        if input.parent_fn != null
          var parent_fn: function.Function = input.parent_fn

          if scanned_fnids->index(parent_fn.id) == -1
            scanned_fnids->add(parent_fn.id)
            funcs->add(parent_fn)
          endif
        endif
      endfor

      sort(funcs, (lhs: function.Function, rhs: function.Function): number => {
          if lhs.gen == rhs.gen
            return 0
          elseif lhs.gen < rhs.gen
            return -1
          endif
          return 1
        })

      # Usually when we differentiate y=f(x) we are
      # interested in df/dx and do not need df/dy(=1) etc.
      # Therefore, we usually release.
      if !retain_outgrad
        for output in fn.outputs
          var output_grad: tensor.Tensor = output.grad
          if input_gradids->index(output_grad.id) == -1
            output.ClearGrad()
          endif
        endfor
      endif
    })

  endwhile
enddef
