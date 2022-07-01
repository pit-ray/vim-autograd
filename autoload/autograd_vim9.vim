vim9script
scriptencoding utf-8


final ENABLE_BACKPROP = 1

final LAST_TENSOR_ID = 0
final LAST_FUNC_ID = v:numbermax / 2 - 1

const PI = acos(-1.0)

def Error(msg: string): void
  echohl ErrorMsg
  echomsg 'autograd: ' .. msg
  echohl None
enddef

function Tensor_cleargrad() dict abort
  let self.grad = {}
endfunction

function Tensor_set_parent_fn(parent_fn) dict abort
  let self.parent_fn = a:parent_fn
  let self.gen = self.parent_fn.gen + 1
endfunction

def CompareTensorGeneration(lhs: any, rhs: any): number
  if lhs['gen'] == rhs['gen']
    return 0
  elseif lhs['gen'] < rhs['gen']
    return -1
  endif
  return 1
enddef

function Tensor_backward(create_graph=v:false, retain_outgrad=v:false) dict abort
  if empty(self.grad)
    let self.grad = autograd#ones_like(self)
  endif

  if empty(self.parent_fn)
    return
  endif

  let l:funcs = [self.parent_fn]
  let l:scanned_fn_ids = []
  while len(l:funcs) > 0
    let l:func = remove(l:funcs, -1)

    let l:gys = []
    for l:output in l:func.outputs
      call add(l:gys, l:output.grad)
    endfor

    " If create_graph is false, does not create graph in the following range.
    " ---------------------------------------------
    if !a:create_graph
      let l:ng = autograd#no_grad()
    endif

    let l:gxs = l:func.backward(l:gys)

    let l:input_grad_ids = []
    let l:input_num = len(l:gxs)
    for l:i in range(l:input_num)
      let l:input = l:func.inputs[l:i]
      if empty(l:input.grad)
        let l:input.grad = l:gxs[l:i]
      else
        let l:input.grad = s:add(l:input.grad, l:gxs[l:i])
      endif

      call add(l:input_grad_ids, l:input.grad.id)

      " It prevents multiple calling backward() of the same function.
      if !empty(l:input.parent_fn)
         \ && index(l:scanned_fn_ids, l:input.parent_fn.id) == -1
        call add(l:scanned_fn_ids, l:input.parent_fn.id)
        call add(l:funcs, l:input.parent_fn)
      endif
    endfor

    call sort(l:funcs, function('CompareTensorGeneration'))

    " Usually when we differentiate y=f(x) we are
    " interested in df/dx and do not need df/dy(=1) etc.
    " Therefore, we usually release.
    if !a:retain_outgrad
      for l:output in l:func.outputs
        if index(l:input_grad_ids, l:output.grad.id) == -1
          let l:output.grad = {}
        endif
      endfor
    endif

    if !a:create_graph
      call l:ng.end()
    endif
    " ---------------------------------------------
  endwhile
endfunction

function Tensor_a(x) dict abort
  return s:add(self, a:x)
endfunction

function Tensor_m(x) dict abort
  return s:mul(self, a:x)
endfunction

function Tensor_s(x) dict abort
  return s:sub(self, a:x)
endfunction

function Tensor_d(x) dict abort
  return s:div(self, a:x)
endfunction

function Tensor_p(x) dict abort
  return s:pow(self, a:x)
endfunction

function Tensor_n() dict abort
  return s:mul(self, -1)
endfunction

function Tensor_T() dict abort
  return s:transpose(self)
endfunction

function Tensor_reshape(...) dict abort
  let l:shape = (a:0 == 1 && type(a:1) == v:t_list) ? a:1 : a:000
  return s:reshape(self, l:shape)
endfunction

function Tensor_flatten() dict abort
  return s:flatten(self)
endfunction

function Tensor_clone() dict abort
  return Tensor(copy(self.data), copy(self.shape))
endfunction

# It returns a new tensor detached from the current graph.
# However, returned tensor shares the same data and shape attribute.
function Tensor_detach() dict abort
  return Tensor(self.data, self.shape)
endfunction

# Tensor class

def Tensor(data: any, shape: any): any
  var tensor = {
    'name': '',
    'id': 0,
    'data': null,
    'grad': {},
    'parent_fn': {},
    'gen': 0,
    'shape': [],
    'cleargrad': function('Tensor_cleargrad'),
    'set_parent_fn': function('Tensor_set_parent_fn'),
    'backward': function('Tensor_backward'),
    'a': function('Tensor_a'),
    'm': function('Tensor_m'),
    's': function('Tensor_s'),
    'd': function('Tensor_d'),
    'p': function('Tensor_p'),
    'n': function('Tensor_n'),
    'T': function('Tensor_T'),
    'reshape': function('Tensor_reshape'),
    'flatten': function('Tensor_flatten'),
    'clone': function('Tensor_clone'),
    'detach': function('Tensor_detach')
  }

  var tensor.data = data
  var tensor.shape = shape

  var tensor.id = last_tensor_id + 1
  var LAST_TENSOR_ID = tensor.id
  return tensor
enddef
