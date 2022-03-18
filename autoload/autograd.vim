let s:enable_backprop = 1

" Tensor
let s:Tensor = {
  \ 'name': '',
  \ 'data': v:none,
  \ 'grad': {},
  \ 'parent_fn': {},
  \ 'gen': 0,
  \ }

function! s:Tensor.zero_grad() abort
  let self.grad = {}
endfunction

function! s:Tensor.set_parent_fn(parent_fn) abort
  let self.parent_fn = a:parent_fn
  let self.gen = self.parent_fn.gen + 1
endfunction

function! s:comp_tensor_gen(lhs, rhs) abort
  if a:lhs['gen'] == a:rhs['gen']
    return 0
  elseif a:lhs['gen'] < a:rhs['gen']
    return -1
  else
    return 1
  endif
endfunction

function! s:has_instance(list, value)
  for l:e in a:list
    if l:e is a:value
      return 1
    endif
  endfor
  return 0
endfunction

function! s:Tensor.backward(...) abort
  let l:retain_fnout_grad = get(a:, 1, 0)

  if empty(self.grad)
    let self.grad = s:Tensor(1.0)
  endif

  if empty(self.parent_fn)
    return
  endif

  let l:funcs = [self.parent_fn]
  while len(l:funcs) > 0
    let l:func = remove(l:funcs, -1)
    let l:gxs = l:func.backward()

    let l:input_grads = []

    let l:input_num = len(l:gxs)
    for l:i in range(l:input_num)
      let l:input = l:func.inputs[l:i]
      if empty(l:input.grad)
        let l:input.grad = l:gxs[l:i]
      else
        let l:input.grad = s:add(l:input.grad, l:gxs[l:i])
      endif

      call add(l:input_grads, l:input.grad)

      if !empty(l:input.parent_fn)
        call add(l:funcs, l:input.parent_fn)
      endif
    endfor

    call sort(l:funcs, function('s:comp_tensor_gen'))

    " Usually when we differentiate y=f(x) we are
    " interested in df/dx and do not need df/dy(=1) etc.
    " Therefore, we usually release.
    if !l:retain_fnout_grad
      for l:output in l:func.outputs
        if !s:has_instance(l:input_grads, l:output.grad)
          let l:output.grad = {}
        endif
      endfor
    endif
  endwhile
endfunction

function! s:Tensor.a(x) abort
  return s:add(self, a:x)
endfunction

function! s:Tensor.m(x) abort
  return s:mul(self, a:x)
endfunction

function! s:Tensor.s(x) abort
  return s:sub(self, a:x)
endfunction

function! s:Tensor.d(x) abort
  return s:div(self, a:x)
endfunction

function! s:Tensor.p(x) abort
  return s:pow(self, a:x)
endfunction

function! s:Tensor.n() abort
  return s:mul(self, -1)
endfunction

function! s:Tensor(data) abort
  let l:tensor = deepcopy(s:Tensor)
  let l:tensor.data = a:data
  return l:tensor
endfunction

function! s:is_tensor(x) abort
  if type(a:x) != v:t_dict
    return 0
  endif
  return has_key(a:x, 'data') && has_key(a:x, 'grad')
endfunction


" Function
let s:Function = {
  \ 'name': '',
  \ 'inputs': [],
  \ 'outputs': [],
  \ 'gen': 0,
  \ 'forward': v:null,
  \ 'backward': v:null
  \ }

function! s:Function.call(...) abort
  let self.inputs = []
  for l:input in a:000
    call add(self.inputs, s:is_tensor(l:input) ? l:input : s:Tensor(l:input))
  endfor

  let self.outputs = self.forward(self.inputs)

  let l:gens = []
  for l:input in self.inputs
    call add(l:gens, l:input.gen)
  endfor
  let self.gen = max(l:gens)

  for l:output in self.outputs
    call l:output.set_parent_fn(self)
  endfor

  return len(self.outputs) > 1 ? self.outputs : self.outputs[0]
endfunction

function! s:Function(name) abort
  let l:func = deepcopy(s:Function)
  let l:func.name = a:name
  let l:func.forward = function(a:name . '_forward')
  let l:func.backward = function(a:name . '_backward')
  return l:func
endfunction


" Operations
function! s:add(x0, x1) abort
  return s:Function('s:add').call(a:x0, a:x1)
endfunction

function! s:add_forward(inputs) dict abort
  return [s:Tensor(a:inputs[0].data + a:inputs[1].data)]
endfunction

function! s:add_backward() dict abort
  let l:gy = self.outputs[0].grad
  return [l:gy, l:gy]
endfunction


function! s:mul(x0, x1) abort
  return s:Function('s:mul').call(a:x0, a:x1)
endfunction

function! s:mul_forward(inputs) dict abort
  return [s:Tensor(a:inputs[0].data * a:inputs[1].data)]
endfunction

function! s:mul_backward() dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]
  let l:gy = self.outputs[0].grad
  return [l:x1.m(l:gy), l:x0.m(l:gy)]
endfunction


function! s:sub(x0, x1) abort
  return s:Function('s:sub').call(a:x0, a:x1)
endfunction

function! s:sub_forward(inputs) dict abort
  return [s:Tensor(self.inputs[0].data - self.inputs[1].data)]
endfunction

function! s:sub_backward() dict abort
  let l:gy = self.outputs[0].grad
  return [l:gy, l:gy.n()]
endfunction


function! s:div(x0, x1) abort
  return s:Function('s:div').call(a:x0, a:x1)
endfunction

function! s:div_forward(inputs) dict abort
  return [s:Tensor(self.inputs[0].data - self.inputs[1].data)]
endfunction

function! s:div_backward() dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]
  let l:gy = self.outputs[0].grad

  let l:gx0 = l:gy.d(l:x1)

  " gx1 = gy * -(x0 / x1 ** 2)
  let l:gx1 = l:gy.m(l:x0.d(l:x1.p(2)).n())

  return [l:gx0, l:gx1]
endfunction


function! s:pow(x, c) abort
  return s:Function('s:pow').call(a:x, a:c)
endfunction

function! s:pow_forward(inputs) dict abort
  return [s:Tensor(pow(self.inputs[0].data, self.inputs[1].data))]
endfunction

function! s:pow_backward() dict abort
  let l:x = self.inputs[0]
  let l:c = self.inputs[1]
  let l:gy = self.outputs[0].grad

  " gx = gy * c * x**(c - 1)
  return [l:gy.m(l:c.m(l:x.p(l:c.s(1))))]
endfunction

" Utilities
function! s:gradcheck(func, inputs) abort
  let l:eps = 1e-6
endfunction


" API
function! autograd#tensor(data) abort
  return s:Tensor(a:data)
endfunction

function! autograd#add(x0, x1) abort
  return s:add(a:x0, a:x1)
endfunction

function! autograd#mul(x0, x1) abort
  return s:mul(a:x0, a:x1)
endfunction

function! autograd#sub(x0, x1) abort
  return s:sub(a:x0, a:x1)
endfunction

function! autograd#div(x0, x1) abort
  return s:div(a:0, a:x1)
endfunction

function! autograd#pow(x, c) abort
  return s:pow(a:x, a:c)
endfunction


" Utilities
function! autograd#nograd_begin() abort
  let s:enable_backprop = 0
endfunction

function! autograd#nograd_end() abort
  let s:enable_backprop = 1
endfunction


function! s:test1() abort
  let l:x0 = s:Tensor(3)
  let l:x1 = s:Tensor(2)

  let l:t = s:mul(l:x0, l:x1)
  echo l:t.data

  let l:x2 = s:Tensor(10)
  let l:y = s:mul(l:t, l:x2)

  echo l:y.data
  call l:y.backward()

  echo l:x0.grad.data l:x1.grad.data
endfunction

function! s:test2() abort
  let l:x = s:Tensor(3)
  echo 'x     :' l:x.data

  echo 'func  : y = 0.5*x^2 - 5*x + 3'
  " let l:y = s:add(s:mul(5, s:pow(l:x, 2)), 4)
  let l:y = s:add(s:sub(s:mul(0.5, s:pow(l:x, 2)), s:mul(5, l:x)), 3)
  echo 'y     :' l:y.data

  call l:y.backward()
  echo 'x.grad:' l:x.grad.data
endfunction

call s:test2()
