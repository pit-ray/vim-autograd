let s:enable_backprop = 1

let s:last_tensor_id = 0
let s:last_func_id = v:numbermax / 2 - 1

function! s:error(msg) abort
  echohl ErrorMsg
  echomsg 'autograd: ' . a:msg
  echohl None
endfunction

" Tensor
let s:Tensor = {
  \ 'name': '',
  \ 'id': 0,
  \ 'data': v:none,
  \ 'grad': {},
  \ 'parent_fn': {},
  \ 'gen': 0,
  \ 'shape': [],
  \ 'size': 0
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
    let self.grad = s:ones_like(self)
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

function! s:Tensor.clone() abort
  return s:Tensor(self.data, self.shape, self.size)
endfunction

function! s:Tensor(data, shape, size) abort
  let l:tensor = deepcopy(s:Tensor)

  let l:tensor.data = a:data
  let l:tensor.shape = a:shape
  let l:tensor.size = a:size

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

function! s:tensor(data) abort
  let l:data = type(a:data) != v:t_list ? [a:data] : a:data

  let l:shape = s:get_matrix_shape(l:data)
  let l:size = s:shape_to_size(l:shape)
  let l:data = flatten(l:data)

  call map(l:data, 'v:val * 1.0')  " int to float

  if len(l:data) != l:size
    call s:error('Invalid matrix shape.')
  endif
  return s:Tensor(l:data, l:shape, l:size)
endfunction

function! s:as_tensor(data) abort
  return s:is_tensor(a:data) ? a:data : s:tensor(a:data)
endfunction

function! s:zeros_like(tensor) abort
  return s:Tensor(
    \ s:vector(a:tensor.size, 0.0),
    \ a:tensor.shape,
    \ a:tensor.size
    \ )
endfunction

function! s:ones_like(tensor) abort
  return s:Tensor(
    \ s:vector(a:tensor.size, 1.0),
    \ a:tensor.shape,
    \ a:tensor.size
    \ )
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
  let l:inputs = []
  for l:input in a:000
    call add(l:inputs, s:as_tensor(l:input))
  endfor

  let l:outputs = self.forward(l:inputs)

  if s:enable_backprop
    let l:gens = []
    for l:input in l:inputs
      call add(l:gens, l:input.gen)
    endfor
    let self.gen = max(l:gens)

    for l:output in l:outputs
      call l:output.set_parent_fn(self)
    endfor

    let self.inputs = l:inputs
    let self.outputs = l:outputs
  endif

  return len(l:outputs) > 1 ? l:outputs : l:outputs[0]
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
function! s:_add(x0, x1) abort
  return a:x0 + a:x1
endfunction

function! s:add(x0, x1) abort
  return s:Function('s:add').call(a:x0, a:x1)
endfunction

function! s:add_forward(xs) dict abort
  return [s:elemwise_binary_op(function('s:_add'), a:xs[0], a:xs[1])]
endfunction

function! s:add_backward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]

  let l:gx0 = a:gys[0]
  let l:gx1 = a:gys[0]

  if l:x0.shape == l:x1.shape
    return [l:gx0, l:gx1]
  endif
  return [s:sum_to(l:gx0, l:x0.shape), s:sum_to(l:gx1, l:x1.shape)]
endfunction


function! s:_mul(x0, x1) abort
  return a:x0 * a:x1
endfunction

function! s:mul(x0, x1) abort
  return s:Function('s:mul').call(a:x0, a:x1)
endfunction

function! s:mul_forward(xs) dict abort
  return [s:elemwise_binary_op(function('s:_mul'), a:xs[0], a:xs[1])]
endfunction

function! s:mul_backward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]

  let l:gx0 = l:x1.m(a:gys[0])
  let l:gx1 = l:x0.m(a:gys[0])

  if l:x0.shape == l:x1.shape
    return [l:gx0, l:gx1]
  endif
  return [s:sum_to(l:gx0, l:x0.shape), s:sum_to(l:gx1, l:x1.shape)]
endfunction


function! s:_sub(x0, x1) abort
  return a:x0 - a:x1
endfunction

function! s:sub(x0, x1) abort
  return s:Function('s:sub').call(a:x0, a:x1)
endfunction

function! s:sub_forward(xs) dict abort
  return [s:elemwise_binary_op(function('s:_sub'), a:xs[0], a:xs[1])]
endfunction

function! s:sub_backward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]

  let l:gx0 = a:gys[0]
  let l:gx1 = a:gys[0].n()

  if l:x0.shape == l:x1.shape
    return [l:gx0, l:gx1]
  endif
  return [s:sum_to(l:gx0, l:x0.shape), s:sum_to(l:gx1, l:x1.shape)]
endfunction


function! s:_div(x0, x1) abort
  return a:x0 / a:x1
endfunction

function! s:div(x0, x1) abort
  return s:Function('s:div').call(a:x0, a:x1)
endfunction

function! s:div_forward(xs) dict abort
  return [s:elemwise_binary_op(function('s:_div'), a:xs[0], a:xs[1])]
endfunction

function! s:div_backward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]

  let l:gx0 = a:gys[0].d(l:x1)

  " gx1 = gy * -(x0 / x1 ** 2)
  let l:gx1 = s:mul(a:gys[0], l:x0.d(l:x1.p(2)).n())

  if l:x0.shape == l:x1.shape
    return [l:gx0, l:gx1]
  endif
  return [s:sum_to(l:gx0, l:x0.shape), s:sum_to(l:gx1, l:x1.shape)]
endfunction

function! s:pow(x, c) abort
  return s:Function('s:pow').call(a:x, a:c)
endfunction

function! s:pow_forward(xs) dict abort
  return [s:elemwise_binary_op({a, b ->pow(a, b)}, a:xs[0], a:xs[1])]
endfunction

function! s:pow_backward(gys) dict abort
  let l:x = self.inputs[0]
  let l:c = self.inputs[1]
  let l:y = self.outputs[0]

  " gx = gy * c * x**(c - 1)
  let l:gx = s:mul(a:gys[0], l:c.m(l:x.p(l:c.s(1))))

  " gc = gy * y * log(x)
  let l:gc = s:mul(a:gys[0], l:y.m(s:log(l:x)))

  if l:x.shape == l:c.shape
    return [l:gx, l:gc]
  endif
  return [s:sum_to(l:gx, l:x.shape), s:sum_to(l:gc, l:c.shape)]
endfunction


function! s:log(x) abort
  return s:Function('s:log').call(a:x)
endfunction

function! s:log_forward(xs) dict abort
  return [s:elemwise_unary_op({a -> log(a)}, a:xs[0])]
endfunction

function! s:log_backward(gys) dict abort
  let l:x = self.inputs[0]
  return [s:div(a:gys[0], l:x)]
endfunction


function! s:sum(x) abort
  return s:Function('s:sum').call(a:x)
endfunction

function! s:sum_forward(xs) dict abort
  let self['x_shape'] = a:xs[0].shape

  let l:total = 0
  for l:e in a:xs[0].data
    let l:total += l:e
  endfor

  return [s:Tensor(l:total, [1], 1)]
endfunction

function! s:sum_backward(gys) dict abort
  return [s:broadcast_to(a:gys[0], self.x_shape)]
endfunction


function! s:broadcast_to(x, shape) abort
  let l:xt = s:as_tensor(a:x)
  if l:xt.shape == a:shape
    return l:xt
  endif

  let l:fn = s:Function('s:broadcast_to')
  let l:fn['shape'] = a:shape
  return l:fn.call(a:x)
endfunction

function! s:broadcast_to_forward(xs) dict abort
  let l:x = a:xs[0]

  " TODO: currently only scalar broadcast are supported.
  if l:x.size > 1
    call s:error('matrix broadcast is not supported yet.')
  endif

  let self['x_shape'] = l:x.shape

  let l:size = s:shape_to_size(self.shape)
  return [s:Tensor(s:vector(l:size, l:x.data[0]), self.shape, l:size)]
endfunction

function! s:broadcast_to_backward(gys) dict abort
  " assume the input size is 1
  return [s:sum_to(a:gys[0], self.x_shape)]
endfunction


function! s:sum_to(x, shape) abort
  let l:xt = s:as_tensor(a:x)
  if l:xt.shape == a:shape
    return l:xt
  endif

  if s:shape_to_size(a:shape) > 1
    call s:error('matrix sum_to is not supported yet.')
  endif

  let l:fn = s:Function('s:sum_to')
  let l:fn.forward = function('s:sum_forward')
  let l:fn.backward = function('s:sum_backward')

  " let l:fn['shape'] = a:shape
  return l:fn.call(a:x)
endfunction


" it returns random value from 0.0 to 1.0.
function! s:rand()
  return rand() / 4294967295.0
endfunction

function! s:vector(size, ...) abort
  let l:init_val = get(a:, 1, 0.0)
  let l:v = repeat([0.0], a:size)
  return l:init_val != 0.0 ? map(l:v, l:init_val) : l:v
endfunction

function! s:shape_to_size(shape) abort
  let l:size = 1
  for l:x in a:shape
    let l:size *= l:x
  endfor
  return l:size
endfunction

function! s:get_matrix_shape(array) abort
  let l:shape = []
  let l:sub_array = a:array
  while type(l:sub_array) == v:t_list
    call add(l:shape, len(l:sub_array))
    let l:sub_array = l:sub_array[0]
  endwhile
  return l:shape
endfunction

function! s:elemwise_unary_op(func, x) abort
  let l:tensor = s:zeros_like(a:x)
  for l:i in range(a:x.size)
    let l:tensor.data[l:i] = a:func(a:x.data[l:i])
  endfor
  return l:tensor
endfunction

function! s:elemwise_binary_op(func, x0, x1) abort
  let l:tensor = s:zeros_like(a:x0.size > a:x1.size ? a:x0 : a:x1)

  " If at least one of them is scalar, it broadcast.
  if a:x0.size == 1 || a:x1.size == 1
    if a:x0.size == l:tensor.size
      for l:i in range(l:tensor.size)
        let l:tensor.data[l:i] = a:func(a:x0.data[l:i], a:x1.data[0])
      endfor
    else
      for l:i in range(l:tensor.size)
        let l:tensor.data[l:i] = a:func(a:x0.data[0], a:x1.data[l:i])
      endfor
    endif
    return l:tensor
  endif

  if a:x0.shape != a:x1.shape
    call s:error('matrix broadcast is not supported yet.')
    return l:tensor
  endif

  for l:i in range(l:tensor.size)
    let l:tensor.data[l:i] = a:func(a:x0.data[l:i], a:x1.data[l:i])
  endfor
  return l:tensor
endfunction


" Utilities
function! s:isclose(a, b, ...) abort
  let l:rtol = get(a:, 1, 0.00001)
  let l:atol = get(a:, 2, 0.00000001)
  return abs(a:a - a:b) <= (l:atol + l:rtol * abs(a:b))
endfunction

function! s:allclose(a, b, ...) abort
  let l:rtol = get(a:, 1, 0.00001)
  let l:atol = get(a:, 2, 0.00000001)

  let l:results = s:elemwise_binary_op(function('s:isclose'), a:a, a:b)
  return min(l:results.data) == 1
endfunction

function! s:numerical_grad(f, x) abort
  let l:eps = s:tensor(0.000001)
  let l:dx = s:tensor(l:eps.data[0] * 2)

  let l:x0 = s:elemwise_binary_op(function('s:_sub'), a:x, l:eps)
  let l:x1 = s:elemwise_binary_op(function('s:_add'), a:x, l:eps)

  let l:y0 = s:elemwise_unary_op(a:f, l:x0)
  let l:y1 = s:elemwise_unary_op(a:f, l:x1)

  let l:dy = s:elemwise_binary_op(function('s:_sub'), l:y1, l:y0)
  return s:elemwise_binary_op(function('s:_div'), l:dy, l:dx)
endfunction

function! s:gradcheck(f, inputs) abort
  let l:y = a:f(a:inputs)

  for l:x in a:inputs
    call l:x.zero_grad()
  endfor
  call l:y.backward()

  let l:grads = []
  for l:x in a:inputs
    call add(l:grads, l:x.grad)
  endfor

  let l:input_num = len(a:inputs)
  for l:i in range(l:input_num)
    let l:before_args = l:i > 0 ? a:inputs[:l:i - 1] : []
    let l:after_args = l:i < l:input_num - 1 ? a:inputs[l:i + 1:] : []

    let l:num_grad = s:numerical_grad(
      \ {x -> a:f(l:before_args + [x] + l:after_args).data[0]},
      \ a:inputs[l:i])

    call assert_true(s:allclose(l:grads[l:i], l:num_grad))
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


function! s:dump_graph(last_node, filepath) abort
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
  return s:tensor(a:data)
endfunction

function! autograd#as_tensor(data) abort
  return s:as_tensor(a:data)
endfunction

function! autograd#zeros_like(tensor) abort
  return s:zeros_like(a:tensor)
endfunction

function! autograd#ones_like(tensor) abort
  return s:ones_like(a:tensor)
endfunction

" Maths
function! autograd#rand() abort
  return s:rand()
endfunction

" Functions
function! autograd#Function(name) abort
  return s:Function(a:name)
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
  return s:div(a:x0, a:x1)
endfunction

function! autograd#pow(x, c) abort
  return s:pow(a:x, a:c)
endfunction

function! autograd#log(x) abort
  return s:log(a:x)
endfunction

function! autograd#sum(x) abort
  return s:sum(a:x)
endfunction

function! autograd#broadcast_to(x, shape) abort
  return s:broadcast_to(a:x, a:shape)
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
  return s:dump_graph(a:last_node, a:filepath)
endfunction
