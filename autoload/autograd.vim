let s:enable_backprop = 1

let s:last_tensor_id = 0
let s:last_func_id = v:numbermax / 2 - 1

" Tensor
let s:Tensor = {
  \ 'name': '',
  \ 'id': 0,
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
  let l:scanned_fnids = []
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

      " It prevents multiple calling backward() of the same function.
      if !empty(l:input.parent_fn)
         \ && index(l:scanned_fnids, l:input.parent_fn.id) == -1
        call add(l:scanned_fnids, l:input.parent_fn.id)
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

  let l:tensor.id = s:last_tensor_id + 1
  let s:last_tensor_id = l:tensor.id
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
  \ 'id': 0,
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

  let l:func.id = s:last_func_id + 1
  let s:last_func_id = l:func.id

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

function! s:dump_tensor_as_dotlang(tensor) abort
  if empty(a:tensor.name) && !empty(a:tensor.data)
    let a:tensor.name = a:tensor.data
  endif
  return a:tensor.id . '[label="' . a:tensor.name . '", color=lightblue, style=filled]'
endfunction

function! s:dump_func_as_dotlang(fn) abort
  let l:def = a:fn.id . '[label="' . a:fn.name . '(' . a:fn.id . '", color=gray, style=filled, shape=box]'

  let l:links = []
  for l:x in a:fn.inputs
    call add(l:links, l:x.id . ' -> ' . a:fn.id)
  endfor

  for l:y in a:fn.outputs
    call add(l:links, a:fn.id . ' -> ' . l:y.id)
  endfor

  return [l:def, l:links]
endfunction


function! s:dump_as_dotlang(last_node, filepath) abort
  let l:defs = [s:dump_tensor_as_dotlang(a:last_node)]
  let l:links = []
  let l:funcs = [a:last_node.parent_fn]

  while len(l:funcs) > 0
    let l:func = remove(l:funcs, -1)
    let l:fn_dot = s:dump_func_as_dotlang(l:func)
    call add(l:defs, l:fn_dot[0])
    let l:links += l:fn_dot[1]

    for l:x in l:func.inputs
      call add(l:defs, s:dump_tensor_as_dotlang(l:x))

      if !empty(l:x.parent_fn)
        call add(l:funcs, l:x.parent_fn)
      endif
    endfor
  endwhile

  let l:links = uniq(sort(l:links))

  let l:texts = ['digraph g {'] + l:defs + l:links + ['}']

  let l:paths = split(a:filepath, '/\|\')
  let l:path = l:paths[-1]
  if len(l:paths) > 1
    let l:dir = join(l:paths[:-2], '/')
    if !isdirectory(l:dir)
      call mkdir(l:dir, 'p')
    endif
    let l:path = l:dir . '/' . l:path
  endif

  call writefile(l:texts, l:path . '.dot')

  if executable('dot')
    echo system(
      \ 'dot ' . l:path . '.dot' .
      \ ' -T ' . split(l:path , '\.')[-1] .
      \ ' -o ' . l:path
      \ )
  endif
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

function! autograd#dump_graph(last_node, filepath) abort
  return s:dump_as_dotlang(a:last_node, a:filepath)
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
  let l:x = s:Tensor(3, 'x')
  echo 'x     :' l:x.data
  echo s:dump_tensor_as_dotlang(l:x)

  echo 'func  : y = 0.5*x^2 - 5*x + 3'
  " let l:y = s:add(s:mul(5, s:pow(l:x, 2)), 4)
  let l:y = s:add(s:sub(s:mul(0.5, s:pow(l:x, 2)), s:mul(5, l:x)), 3)
  echo 'y     :' l:y.data

  call s:dump_as_dotlang(l:y, '.autograd/test2.png')

  call l:y.backward()
  echo 'x.grad:' l:x.grad.data
endfunction

" call s:test2()
