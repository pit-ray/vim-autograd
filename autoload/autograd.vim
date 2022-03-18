let s:enable_backprop = 1

let s:last_tensor_id = 0
let s:last_func_id = v:numbermax / 2 - 1

let s:eps = 0.000001

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

    let l:gys = []
    for l:output in l:func.outputs
      call add(l:gys, l:output.grad)
    endfor
    let l:gxs = l:func.backward(l:gys)

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


" Maths
function! s:isclose(a, b, ...) abort
  let l:rtol = get(a:, 1, 0.00001)
  let l:atol = get(a:, 2, 0.00000001)
  return abs(a:a - a:b) <= (l:atol + l:rtol * abs(a:b))
endfunction

" it returns random value from 0.0 to 1.0.
function! s:rand()
  return rand() / 4294967295.0
endfunction


" Operations
function! s:add(x0, x1) abort
  return s:Function('s:add').call(a:x0, a:x1)
endfunction

function! s:add_forward(xs) dict abort
  return [s:Tensor(a:xs[0].data + a:xs[1].data)]
endfunction

function! s:add_backward(gys) dict abort
  return [a:gys[0], a:gys[0]]
endfunction


function! s:mul(x0, x1) abort
  return s:Function('s:mul').call(a:x0, a:x1)
endfunction

function! s:mul_forward(xs) dict abort
  return [s:Tensor(a:xs[0].data * a:xs[1].data)]
endfunction

function! s:mul_backward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]
  return [l:x1.m(a:gys[0]), l:x0.m(a:gys[0])]
endfunction


function! s:sub(x0, x1) abort
  return s:Function('s:sub').call(a:x0, a:x1)
endfunction

function! s:sub_forward(xs) dict abort
  return [s:Tensor(a:xs[0].data - a:xs[1].data)]
endfunction

function! s:sub_backward(gys) dict abort
  return [a:gys[0], a:gys[0].n()]
endfunction


function! s:div(x0, x1) abort
  return s:Function('s:div').call(a:x0, a:x1)
endfunction

function! s:div_forward(xs) dict abort
  return [s:Tensor(a:xs[0].data / a:xs[1].data)]
endfunction

function! s:div_backward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]

  let l:gx0 = a:gys[0].d(l:x1)

  " gx1 = gy * -(x0 / x1 ** 2)
  let l:gx1 = s:mul(a:gys[0], l:x0.d(l:x1.p(2)).n())

  return [l:gx0, l:gx1]
endfunction

function! s:pow(x, c) abort
  return s:Function('s:pow').call(a:x, a:c)
endfunction

function! s:pow_forward(xs) dict abort
  return [s:Tensor(pow(a:xs[0].data, a:xs[1].data))]
endfunction

function! s:pow_backward(gys) dict abort
  let l:x = self.inputs[0]
  let l:c = self.inputs[1]
  let l:y = self.outputs[0]

  " gx = gy * c * x**(c - 1)
  let l:gx = s:mul(a:gys[0], l:c.m(l:x.p(l:c.s(1))))

  " gc = gy * y * log(x)
  let l:gc = s:mul(a:gys[0], l:y.m(s:log(l:x)))
  return [l:gx, l:gc]
endfunction


function! s:log(x) abort
  return s:Function('s:log').call(a:x)
endfunction

function! s:log_forward(xs) dict abort
  return [s:Tensor(log(a:xs[0].data))]
endfunction

function! s:log_backward(gys) dict abort
  let l:x = self.inputs[0]
  return [s:div(a:gys[0], l:x)]
endfunction


" Utilities
function! s:numerical_grad(f, x) abort
  let l:y0 = a:f(s:Tensor(a:x.data - s:eps))
  let l:y1 = a:f(s:Tensor(a:x.data + s:eps))
  return (l:y1.data - l:y0.data) / (2 * s:eps)
endfunction

function! s:gradcheck(f, inputs) abort
  let l:y = a:f(a:inputs)

  for l:x in a:inputs
    call l:x.zero_grad()
  endfor
  call l:y.backward()

  let l:grads = []
  for l:x in a:inputs
    call add(l:grads, l:x.grad.data)
  endfor

  let l:result = 1
  let l:input_num = len(a:inputs)
  for l:i in range(l:input_num)
    let l:before_args = l:i > 0 ? a:inputs[:l:i - 1] : []
    let l:after_args = l:i < l:input_num - 1 ? a:inputs[l:i + 1:] : []

    let l:num_grad = s:numerical_grad(
      \ {x -> a:f(l:before_args + [x] + l:after_args)},
      \ a:inputs[l:i]
      \ )

    call assert_true(s:isclose(l:grads[l:i], l:num_grad))
  endfor
endfunction


function! s:dump_tensor_as_dotlang(tensor) abort
  return a:tensor.id . '[label="' . a:tensor.name . '", color=lightblue, style=filled]'
endfunction

function! s:dump_func_as_dotlang(fn) abort
  let l:def = a:fn.id . '[label="' . a:fn.name . '", color=gray, style=filled, shape=box]'

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
" Tensor
function! autograd#tensor(data) abort
  return s:Tensor(a:data)
endfunction

" Maths
function! autograd#rand() abort
  return s:rand()
endfunction

" Functions
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
  return s:div(a:x0, a:x1)
endfunction

function! autograd#pow(x, c) abort
  return s:pow(a:x, a:c)
endfunction

function! autograd#log(x) abort
  return s:log(a:x)
endfunction

" Utilities
function! autograd#nograd_begin() abort
  let s:enable_backprop = 0
endfunction

function! autograd#nograd_end() abort
  let s:enable_backprop = 1
endfunction

function! autograd#numerical_grad(f, x) abort
  return s:numerical_grad(a:f, a:x)
endfunction

function! autograd#gradcheck(f, inputs) abort
  return s:gradcheck(a:f, a:inputs)
endfunction

function! autograd#dump_graph(last_node, filepath) abort
  return s:dump_as_dotlang(a:last_node, a:filepath)
endfunction
