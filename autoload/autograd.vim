let s:enable_backprop = 1

let s:last_tensor_id = 0
let s:last_func_id = v:numbermax / 2 - 1

let s:pi = acos(-1.0)

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

function! s:Tensor.cleargrad() abort
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

function! s:Tensor.backward(...) abort
  let l:create_graph = get(a:, 1, 0)
  let l:retain_outgrad = get(a:, 2, 0)

  if empty(self.grad)
    let self.grad = s:ones_like(self)
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
    if !l:create_graph
      call s:nograd_begin()
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

    call sort(l:funcs, function('s:comp_tensor_gen'))

    " Usually when we differentiate y=f(x) we are
    " interested in df/dx and do not need df/dy(=1) etc.
    " Therefore, we usually release.
    if !l:retain_outgrad
      for l:output in l:func.outputs
        if index(l:input_grad_ids, l:output.grad.id) == -1
          let l:output.grad = {}
        endif
      endfor
    endif

    if !l:create_graph
      call s:nograd_end()
    endif
    " ---------------------------------------------
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

function! s:Tensor.T() abort
  return s:transpose(self)
endfunction

function! s:Tensor.clone() abort
  return s:Tensor(copy(self.data), copy(self.shape), self.size)
endfunction

" It returns a new tensor detached from the current graph.
" However, returned tensor shares the same data and shape attribute.
function! s:Tensor.detach() abort
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

function! s:zeros(shape) abort
  let l:size = s:shape_to_size(a:shape)
  if l:size == 0
    call s:error('axis without element is invalid.')
  endif
  return s:Tensor(s:vector(l:size, 0.0), a:shape, l:size)
endfunction

function! s:zeros_like(tensor) abort
  return s:Tensor(
    \ s:vector(a:tensor.size, 0.0),
    \ a:tensor.shape,
    \ a:tensor.size
    \ )
endfunction

function! s:ones(shape) abort
  let l:size = s:shape_to_size(a:shape)
  if l:size == 0
    call s:error('axis without element is invalid.')
  endif
  return s:Tensor(s:vector(l:size, 1.0), a:shape, l:size)
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

function! s:Function.apply(...) abort
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
  return s:Function('s:add').apply(a:x0, a:x1)
endfunction

function! s:add_forward(xs) dict abort
  return [s:elementwise(function('s:_add'), a:xs)]
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
  return s:Function('s:mul').apply(a:x0, a:x1)
endfunction

function! s:mul_forward(xs) dict abort
  return [s:elementwise(function('s:_mul'), a:xs)]
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
  return s:Function('s:sub').apply(a:x0, a:x1)
endfunction

function! s:sub_forward(xs) dict abort
  return [s:elementwise(function('s:_sub'), a:xs)]
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
  return s:Function('s:div').apply(a:x0, a:x1)
endfunction

function! s:div_forward(xs) dict abort
  return [s:elementwise(function('s:_div'), a:xs)]
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
  return s:Function('s:pow').apply(a:x, a:c)
endfunction

function! s:pow_forward(xs) dict abort
  return [s:elementwise({a, b ->pow(a, b)}, a:xs)]
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
  return s:Function('s:log').apply(a:x)
endfunction

function! s:log_forward(xs) dict abort
  return [s:elementwise({a -> log(a)}, a:xs)]
endfunction

function! s:log_backward(gys) dict abort
  let l:x = self.inputs[0]
  return [s:div(a:gys[0], l:x)]
endfunction


function! s:exp(x) abort
  return s:Function('s:exp').apply(a:x)
endfunction

function! s:exp_forward(xs) dict abort
  return [s:elementwise({a -> exp(a)}, a:xs)]
endfunction

function! s:exp_backward(gys) dict abort
  let l:y = self.outputs[0]
  return [s:mul(a:gys[0], l:y)]
endfunction


function! s:sin(x) abort
  return s:Function('s:sin').apply(a:x)
endfunction

function! s:sin_forward(xs) dict abort
  return [s:elementwise({a -> sin(a)}, a:xs)]
endfunction

function! s:sin_backward(gys) dict abort
  let l:x = self.inputs[0]
  return [s:mul(a:gys[0], s:cos(l:x))]
endfunction


function! s:cos(x) abort
  return s:Function('s:cos').apply(a:x)
endfunction

function! s:cos_forward(xs) dict abort
  return [s:elementwise({a -> cos(a)}, a:xs)]
endfunction

function! s:cos_backward(gys) dict abort
  let l:x = self.inputs[0]
  return [s:mul(a:gys[0], s:sin(l:x).n())]
endfunction


function! s:tanh(x) abort
  return s:Function('s:tanh').apply(a:x)
endfunction

function! s:tanh_forward(xs) dict abort
  return [s:elementwise({a -> tanh(a)}, a:xs)]
endfunction

function! s:tanh_backward(gys) dict abort
  let l:y = self.outputs[0]
  return [s:mul(a:gys[0], s:sub(1, l:y.p(2)))]
endfunction


function! s:abs(x) abort
  return s:Function('s:abs').apply(a:x)
endfunction

function! s:abs_forward(xs) dict abort
  return [s:elementwise({a -> abs(a)}, a:xs)]
endfunction

function! s:abs_backward(gys) dict abort
  let l:x = self.inputs[0]
  return [s:mul(a:gys[0], s:sign(l:x))]
endfunction


function! s:_sign(x) abort
  return a:x > 0.0 ? 1.0 : (a:x < -1.0 ? -1.0 : 0.0)
endfunction

function! s:sign(x) abort
  return s:Function('s:sign').apply(a:x)
endfunction

function! s:sign_forward(xs) dict abort
  return [s:elementwise(function('s:_sign'), a:xs)]
endfunction

function! s:sign_backward(gys) dict abort
  return [s:mul(a:gys[0], 0.0)]
endfunction


function! s:_sum(x) abort
  let l:total = 0
  for l:e in a:x.data
    let l:total += l:e
  endfor
  return l:total
endfunction

function! s:sum(x) abort
  return s:Function('s:sum').apply(a:x)
endfunction

function! s:sum_forward(xs) dict abort
  let self['x_shape'] = a:xs[0].shape
  let l:s = s:_sum(a:xs[0])
  return [s:Tensor(l:s, [1], 1)]
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
  return l:fn.apply(a:x)
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

  return s:Function('s:sum_to').apply(a:x)
endfunction

function! s:sum_to_forward(xs) dict abort
  let self['x_shape'] = a:xs[0].shape
  let l:s = s:_sum(a:xs[0])
  return [s:Tensor(l:s, [1], 1)]
endfunction

function! s:sum_to_backward(gys) dict abort
  return [s:broadcast_to(a:gys[0], self.x_shape)]
endfunction


function! s:_transpose(x) abort
  let l:dim = len(a:x.shape)
  if l:dim > 2
    call s:error('transpose() is supported only for 1D-tensor and 2D-tensor.')
  endif

  if l:dim == 1
    return a:x
  endif

  let l:xd = a:x.data

  let l:out_data = s:vector(a:x.size)

  let l:n_i = a:x.shape[0]
  let l:n_j = a:x.shape[1]

  let l:n_j_range = range(l:n_j)
  for l:i in range(l:n_i)
    let l:buf = l:i * l:n_j
    for l:j in l:n_j_range
      let l:out_data[l:j * l:n_i + l:i] = l:xd[l:buf + l:j]
    endfor
  endfor

  return s:Tensor(l:out_data, [l:n_j, l:n_i], a:x.size)
endfunction


function! s:transpose(x) abort
  return s:Function('s:transpose').apply(a:x)
endfunction

function! s:transpose_forward(xs) dict abort
  return [s:_transpose(a:xs[0])]
endfunction

function! s:transpose_backward(gys) dict abort
  return [s:transpose(a:gys[0])]
endfunction


function! s:_matmul(x0, x1) abort

  return l:out
endfunction

function! s:matmul(a, b) abort
  return s:Function('s:matmul').apply(a:a, a:b)
endfunction

function! s:matmul_forward(xs) dict abort
  let l:x0 = a:xs[0]
  let l:x1 = a:xs[1]

  let l:x0_dim = len(l:x0.shape)
  let l:x1_dim = len(l:x1.shape)

  if l:x0_dim > 2 || l:x1_dim > 2
    call s:error('inputs must be 2D-2D or 1D-1D.')
    return
  endif

  let l:x0_shape = copy(l:x0.shape)
  let l:x1_shape = copy(l:x1.shape)
  let self['x0_shape_fix'] = l:x0_shape
  let self['x1_shape_fix'] = l:x1_shape

  " 1D-tensor is converted to 2D-tensor
  if l:x0_dim == 1
    call insert(l:x0_shape, 1, 0)
  endif
  if l:x1_dim == 1
    call add(l:x1_shape, 1)
  endif

  if l:x0_shape[1] != l:x1_shape[0]
    call s:error('axis 1 of left operand mismatchs axis 0 of right.')
  endif

  let l:n_i = l:x0_shape[0]
  let l:n_k = l:x0_shape[1]
  let l:n_j = l:x1_shape[1]

  let l:out = s:zeros([l:n_i, l:n_j])

  let l:od = l:out.data
  let l:d0 = l:x0.data
  let l:d1 = l:x1.data

  " 2D matrix product (ikj-algorithm)
  let l:n_k_range = range(l:n_k)
  let l:n_j_range = range(l:n_j)
  for l:i in range(l:n_i)
    for l:k in l:n_k_range
      let l:buf = l:d0[l:i * l:n_k + l:k]
      for l:j in l:n_j_range
        let l:od[l:i * l:n_j + l:j] += l:buf * l:d1[l:k * l:n_j + l:j]
      endfor
    endfor
  endfor

  " If one is 1D, output in 1D
  if l:x0_dim == 1
    call remove(l:out.shape, 0)
  elseif l:x1_dim == 1
    call remove(l:out.shape, 1)
  endif

  return [l:out]
endfunction

function! s:matmul_backward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]
  let l:gy = a:gys[0]

  let l:x0_shape_raw = l:x0.shape
  let l:x1_shape_raw = l:x1.shape

  " temporarily restores the shape of x when y is calculated.
  let l:x0.shape = self.x0_shape_fix
  let l:x1.shape = self.x1_shape_fix

  let l:gx0 = s:matmul(l:gy, l:x1.T())
  let l:gx1 = s:matmul(l:x0.T(), l:gy)

  " return to the original shape
  let l:x0.shape = l:x0_shape_raw
  let l:x1.shape = l:x1_shape_raw

  return [l:gx0, l:gx1]
endfunction


" it returns random value from 0.0 to 1.0.
function! s:random_sample()
  return rand() / 4294967295.0
endfunction

function! s:box_muller(u1, u2) abort
  return sqrt(-2 * log(a:u1)) * cos(2 * s:pi * a:u2)
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

function! s:elementwise(func, inputs) abort
  if len(a:inputs) == 1
    let l:x = a:inputs[0]
    let l:tensor = s:zeros_like(l:x)

    let l:td = l:tensor.data
    let l:xd = l:x.data
    for l:i in range(l:x.size)
      let l:td[l:i] = a:func(l:xd[l:i])
    endfor
    return l:tensor
  endif

  let l:x0 = a:inputs[0]
  let l:x1 = a:inputs[1]
  let l:tensor = s:zeros_like(l:x0.size > l:x1.size ? l:x0 : l:x1)

  let l:td = l:tensor.data
  let l:x0d = l:x0.data
  let l:x1d = l:x1.data

  " If at least one of them is scalar, it broadcast.
  if l:x0.size == 1 || l:x1.size == 1
    if l:x0.size == l:tensor.size
      for l:i in range(l:tensor.size)
        let l:td[l:i] = a:func(l:x0d[l:i], l:x1d[0])
      endfor
    else
      for l:i in range(l:tensor.size)
        let l:td[l:i] = a:func(l:x0d[0], l:x1d[l:i])
      endfor
    endif
    return l:tensor
  endif

  if l:x0.shape != l:x1.shape
    call s:error('matrix broadcast is not supported yet.')
    return l:tensor
  endif

  for l:i in range(l:tensor.size)
    let l:td[l:i] = a:func(l:x0d[l:i], l:x1d[l:i])
  endfor
  return l:tensor
endfunction


" Utilities
let s:nograd_state_cache = s:enable_backprop
function! s:nograd_begin() abort
  let s:nograd_state_cache = s:enable_backprop
  let s:enable_backprop = 0
endfunction

function! s:nograd_end() abort
  let s:enable_backprop = s:nograd_state_cache
endfunction

function! s:isclose(a, b, ...) abort
  let l:rtol = get(a:, 1, 0.00001)
  let l:atol = get(a:, 2, 0.00000001)
  return abs(a:a - a:b) <= (l:atol + l:rtol * abs(a:b))
endfunction

function! s:allclose(a, b, ...) abort
  let l:rtol = get(a:, 1, 0.00001)
  let l:atol = get(a:, 2, 0.00000001)

  let l:results = s:elementwise(function('s:isclose'), [a:a, a:b])
  return min(l:results.data) == 1
endfunction

function! s:numerical_grad(f, x) abort
  let l:eps = s:tensor(0.000001)
  let l:dx = s:tensor(l:eps.data[0] * 2)

  let l:x0 = s:sub(a:x, l:eps)
  let l:x1 = s:add(a:x, l:eps)

  let l:y0 = a:f(l:x0)
  let l:y1 = a:f(l:x1)

  let l:dy = s:sub(l:y1, l:y0)
  return s:div(l:dy, l:dx)
endfunction

function! s:gradcheck(f, inputs) abort
  let l:y = a:f(a:inputs)

  for l:x in a:inputs
    call l:x.cleargrad()
  endfor
  call l:y.backward()

  let l:grads = []
  for l:x in a:inputs
    call add(l:grads, l:x.grad)
  endfor

  call s:nograd_begin()
  let l:input_num = len(a:inputs)
  for l:i in range(l:input_num)
    let l:before_args = l:i > 0 ? a:inputs[:l:i - 1] : []
    let l:after_args = l:i < l:input_num - 1 ? a:inputs[l:i + 1:] : []

    let l:num_grad = s:numerical_grad(
      \ {x -> a:f(l:before_args + [x] + l:after_args)},
      \ a:inputs[l:i])

    call assert_true(s:allclose(l:grads[l:i], l:num_grad))
  endfor
  call s:nograd_end()
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

function! autograd#zeros(shape) abort
  return s:zeros(a:shape)
endfunction

function! autograd#zeros_like(tensor) abort
  return s:zeros_like(a:tensor)
endfunction

function! autograd#ones(shape) abort
  return s:ones(a:shape)
endfunction

function! autograd#ones_like(tensor) abort
  return s:ones_like(a:tensor)
endfunction

" Maths
function! autograd#rand(...) abort
  let l:shape = a:0 > 0 ? a:000 : [1]
  let l:size = s:shape_to_size(l:shape)
  let l:data = map(repeat([0.0], l:size), 's:random_sample()')
  return s:Tensor(l:data, l:shape, l:size)
endfunction

function! autograd#uniform(...) abort
  let l:low = get(a:, 1, 0.0) * 1.0
  let l:high = get(a:, 2, 1.0) * 1.0
  let l:shape = get(a:, 3, [1])

  let l:size = s:shape_to_size(l:shape)
  let l:data = map(
    \ repeat([0.0], l:size),
    \ 'l:low + (l:high - l:low) * s:random_sample()')
  return s:Tensor(l:data, l:shape, l:size)
endfunction

function! autograd#randn(...) abort
  let l:shape = a:0 > 0 ? a:000 : [1]
  let l:size = s:shape_to_size(l:shape)
  let l:data = map(
    \ repeat([0.0], l:size),
    \ 's:box_muller(s:random_sample(), s:random_sample())')
  return s:Tensor(l:data, l:shape, l:size)
endfunction

function! autograd#normal(mean, std, ...) abort
  let l:shape = get(a:, 1, [1])

  let l:size = s:shape_to_size(l:shape)
  let l:data = map(
    \ repeat([0.0], l:size),
    \ 'a:std * s:box_muller(s:random_sample(), s:random_sample()) + a:mean')
  return s:Tensor(l:data, l:shape, l:size)
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

function! autograd#exp(x) abort
  return s:exp(a:x)
endfunction

function! autograd#sin(x) abort
  return s:sin(a:x)
endfunction

function! autograd#cos(x) abort
  return s:cos(a:x)
endfunction

function! autograd#tanh(x) abort
  return s:tanh(a:x)
endfunction

function! autograd#abs(x) abort
  return s:abs(a:x)
endfunction

function! autograd#sign(x) abort
  return s:sign(a:x)
endfunction

function! autograd#sum(x) abort
  return s:sum(a:x)
endfunction

function! autograd#broadcast_to(x, shape) abort
  return s:broadcast_to(a:x, a:shape)
endfunction

function! autograd#transpose(x) abort
  return s:transpose(a:x)
endfunction

function! autograd#matmul(a, b) abort
  return s:matmul(a:a, a:b)
endfunction

function! autograd#pi() abort
  return acos(-1.0)
endfunction

" Utilities
function! autograd#grad(output, inputs, ...) abort
  let l:create_graph = get(a:, 1, 0)
  let l:retain_outgrad = get(a:, 2, 0)

  let l:xs = s:is_tensor(a:inputs) ? [a:inputs] : a:inputs

  let l:old_grads = []
  for l:x in l:xs
    call add(l:old_grads, l:x.grad)
    call l:x.cleargrad()
  endfor

  call a:output.backward(l:create_graph, l:retain_outgrad)

  let l:grads = []
  for l:i in range(len(l:xs))
    call add(l:grads, l:xs[l:i].grad)
    let l:xs[l:i].grad = l:old_grads[l:i]
  endfor

  return len(l:grads) > 1 ? l:grads : l:grads[0]
endfunction


function! autograd#nograd_begin() abort
  return s:nograd_begin()
endfunction

function! autograd#nograd_end() abort
  return s:nograd_end()
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
