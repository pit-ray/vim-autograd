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
    if !l:create_graph
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
      call l:ng.end()
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

function! s:Tensor.reshape(...) abort
  let l:shape = (a:0 == 1 && type(a:1) == v:t_list) ? a:1 : a:000
  return s:reshape(self, l:shape)
endfunction

function! s:Tensor.clone() abort
  return s:Tensor(copy(self.data), copy(self.shape))
endfunction

" It returns a new tensor detached from the current graph.
" However, returned tensor shares the same data and shape attribute.
function! s:Tensor.detach() abort
  return s:Tensor(self.data, self.shape)
endfunction

function! s:Tensor(data, shape) abort
  let l:tensor = deepcopy(s:Tensor)

  let l:tensor.data = a:data
  let l:tensor.shape = a:shape

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

function! autograd#tensor(data) abort
  let l:data = s:as_list(deepcopy(a:data))

  let l:shape = s:get_matrix_shape(l:data)
  let l:data = flatten(l:data)

  call map(l:data, 'v:val * 1.0')  " int to float

  if len(l:data) != s:shape_to_size(l:shape)
    call s:error('Invalid matrix shape.')
  endif
  return s:Tensor(l:data, l:shape)
endfunction

function! s:as_list(data) abort
  return type(a:data) != v:t_list ? [a:data] : a:data
endfunction

function! autograd#as_tensor(data) abort
  return s:is_tensor(a:data) ? a:data : autograd#tensor(a:data)
endfunction

function! autograd#zeros(shape) abort
  let l:size = s:shape_to_size(a:shape)
  if l:size == 0
    call s:error('axis without element is invalid.')
  endif
  return s:Tensor(s:vector(l:size, 0.0), a:shape)
endfunction

function! autograd#zeros_like(tensor) abort
  return s:Tensor(s:vector(len(a:tensor.data), 0.0), a:tensor.shape)
endfunction

function! autograd#ones(shape) abort
  let l:size = s:shape_to_size(a:shape)
  if l:size == 0
    call s:error('axis without element is invalid.')
  endif
  return s:Tensor(s:vector(l:size, 1.0), a:shape)
endfunction

function! autograd#ones_like(tensor) abort
  return s:Tensor(s:vector(len(a:tensor.data), 1.0), a:tensor.shape)
endfunction


let s:random_seed = srand()
function! autograd#manual_seed(seed) abort
  let s:random_seed = srand(a:seed)
endfunction

function! autograd#random() abort
  " it returns random value from 0.0 to 1.0.
  return rand(s:random_seed) / 4294967295.0
endfunction

function! s:box_muller(u1, u2) abort
  return sqrt(-2 * log(a:u1)) * cos(2 * s:pi * a:u2)
endfunction

function! autograd#rand(...) abort
  let l:shape = a:0 > 0 ? a:000 : [1]
  let l:size = s:shape_to_size(l:shape)
  let l:data = map(repeat([0.0], l:size), 'autograd#random()')
  return s:Tensor(l:data, l:shape)
endfunction

function! autograd#uniform(...) abort
  let l:low = get(a:, 1, 0.0) * 1.0
  let l:high = get(a:, 2, 1.0) * 1.0
  let l:shape = get(a:, 3, [1])

  let l:size = s:shape_to_size(l:shape)
  let l:data = map(
    \ repeat([0.0], l:size),
    \ 'l:low + (l:high - l:low) * autograd#random()')
  return s:Tensor(l:data, l:shape)
endfunction

function! autograd#randn(...) abort
  let l:shape = a:0 > 0 ? a:000 : [1]
  let l:size = s:shape_to_size(l:shape)
  let l:data = map(
    \ repeat([0.0], l:size),
    \ 's:box_muller(autograd#random(), autograd#random())')
  return s:Tensor(l:data, l:shape)
endfunction

function! autograd#normal(mean, std, ...) abort
  let l:shape = get(a:, 1, [1])

  let l:size = s:shape_to_size(l:shape)
  let l:data = map(
    \ repeat([0.0], l:size),
    \ 'a:std * s:box_muller(autograd#random(), autograd#random()) + a:mean')
  return s:Tensor(l:data, l:shape)
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
    call add(l:inputs, autograd#as_tensor(l:input))
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

function! autograd#Function(name) abort
  let l:func = deepcopy(s:Function)
  let l:func.name = a:name

  let l:func.forward = function(a:name . '_forward')
  let l:func.backward = function(a:name . '_backward')

  let l:func.id = s:last_func_id + 1
  let s:last_func_id = l:func.id

  return l:func
endfunction


" Operations
function! autograd#add(x0, x1) abort
  return s:add(a:x0, a:x1)
endfunction

function! s:_add(x0, x1) abort
  return a:x0 + a:x1
endfunction

function! s:add(x0, x1) abort
  return autograd#Function('s:add').apply(a:x0, a:x1)
endfunction

function! s:add_forward(xs) dict abort
  return [autograd#elementwise(a:xs, function('s:_add'))]
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


function! autograd#mul(x0, x1) abort
  return s:mul(a:x0, a:x1)
endfunction

function! s:_mul(x0, x1) abort
  return a:x0 * a:x1
endfunction

function! s:mul(x0, x1) abort
  return autograd#Function('s:mul').apply(a:x0, a:x1)
endfunction

function! s:mul_forward(xs) dict abort
  return [autograd#elementwise(a:xs, function('s:_mul'))]
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


function! autograd#sub(x0, x1) abort
  return s:sub(a:x0, a:x1)
endfunction

function! s:_sub(x0, x1) abort
  return a:x0 - a:x1
endfunction

function! s:sub(x0, x1) abort
  return autograd#Function('s:sub').apply(a:x0, a:x1)
endfunction

function! s:sub_forward(xs) dict abort
  return [autograd#elementwise(a:xs, function('s:_sub'))]
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


function! autograd#div(x0, x1) abort
  return s:div(a:x0, a:x1)
endfunction

function! s:_div(x0, x1) abort
  return a:x0 / a:x1
endfunction

function! s:div(x0, x1) abort
  return autograd#Function('s:div').apply(a:x0, a:x1)
endfunction

function! s:div_forward(xs) dict abort
  return [autograd#elementwise(a:xs, function('s:_div'))]
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


function! autograd#pow(x, c) abort
  return s:pow(a:x, a:c)
endfunction

function! s:pow(x, c) abort
  return autograd#Function('s:pow').apply(a:x, a:c)
endfunction

function! s:pow_forward(xs) dict abort
  return [autograd#elementwise(a:xs, {a, b ->pow(a, b)})]
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


function! autograd#sqrt(x) abort
  return s:pow(a:x, 0.5)
endfunction


function! autograd#log(x) abort
  return s:log(a:x)
endfunction

function! s:log(x) abort
  return autograd#Function('s:log').apply(a:x)
endfunction

function! s:log_forward(xs) dict abort
  return [autograd#elementwise(a:xs, {a -> log(a)})]
endfunction

function! s:log_backward(gys) dict abort
  let l:x = self.inputs[0]
  return [s:div(a:gys[0], l:x)]
endfunction


function! autograd#exp(x) abort
  return s:exp(a:x)
endfunction

function! s:exp(x) abort
  return autograd#Function('s:exp').apply(a:x)
endfunction

function! s:exp_forward(xs) dict abort
  return [autograd#elementwise(a:xs, {a -> exp(a)})]
endfunction

function! s:exp_backward(gys) dict abort
  let l:y = self.outputs[0]
  return [s:mul(a:gys[0], l:y)]
endfunction


function! autograd#sin(x) abort
  return s:sin(a:x)
endfunction

function! s:sin(x) abort
  return autograd#Function('s:sin').apply(a:x)
endfunction

function! s:sin_forward(xs) dict abort
  return [autograd#elementwise(a:xs, {a -> sin(a)})]
endfunction

function! s:sin_backward(gys) dict abort
  let l:x = self.inputs[0]
  return [s:mul(a:gys[0], s:cos(l:x))]
endfunction


function! autograd#cos(x) abort
  return s:cos(a:x)
endfunction

function! s:cos(x) abort
  return autograd#Function('s:cos').apply(a:x)
endfunction

function! s:cos_forward(xs) dict abort
  return [autograd#elementwise(a:xs, {a -> cos(a)})]
endfunction

function! s:cos_backward(gys) dict abort
  let l:x = self.inputs[0]
  return [s:mul(a:gys[0], s:sin(l:x).n())]
endfunction


function! autograd#tanh(x) abort
  return s:tanh(a:x)
endfunction

function! s:tanh(x) abort
  return autograd#Function('s:tanh').apply(a:x)
endfunction

function! s:tanh_forward(xs) dict abort
  return [autograd#elementwise(a:xs, {a -> tanh(a)})]
endfunction

function! s:tanh_backward(gys) dict abort
  let l:y = self.outputs[0]
  return [s:mul(a:gys[0], s:sub(1, l:y.p(2)))]
endfunction


function! autograd#abs(x) abort
  return s:abs(a:x)
endfunction

function! s:abs(x) abort
  return autograd#Function('s:abs').apply(a:x)
endfunction

function! s:abs_forward(xs) dict abort
  return [autograd#elementwise(a:xs, {a -> abs(a)})]
endfunction

function! s:abs_backward(gys) dict abort
  let l:x = self.inputs[0]
  return [s:mul(a:gys[0], s:sign(l:x))]
endfunction


function! autograd#sign(x) abort
  return s:sign(a:x)
endfunction

function! s:_sign(x) abort
  return a:x > 0.0 ? 1.0 : (a:x < -1.0 ? -1.0 : 0.0)
endfunction

function! s:sign(x) abort
  return autograd#Function('s:sign').apply(a:x)
endfunction

function! s:sign_forward(xs) dict abort
  return [autograd#elementwise(a:xs, function('s:_sign'))]
endfunction

function! s:sign_backward(gys) dict abort
  return [s:mul(a:gys[0], 0.0)]
endfunction


function! s:left_side_sum_to(x, shape) abort
  let l:y = autograd#zeros(a:shape)

  let l:xd = a:x.data
  let l:yd = l:y.data

  let l:x_size = len(l:xd)
  let l:y_size = len(l:yd)

  for l:i in range(l:x_size / l:y_size)
    let l:start = l:i * l:y_size
    for l:j in range(l:y_size)
      let l:yd[l:j] += l:xd[l:start + l:j]
    endfor
  endfor
  return l:y
endfunction

function! s:right_side_sum_to(x, shape) abort
  let l:y = autograd#zeros(a:shape)

  let l:xd = a:x.data
  let l:yd = l:y.data

  let l:x_size = len(l:xd)
  let l:y_size = len(l:yd)

  let l:block_size = l:x_size / l:y_size
  for l:i in range(l:y_size)
    let l:start = l:block_size * l:i
    for l:j in range(l:block_size)
      let l:yd[l:i] += l:xd[l:start + l:j]
    endfor
  endfor
  return l:y
endfunction


function! autograd#sum(x, ...) abort
  let l:axis = get(a:, 1, [])
  let l:keepdims = get(a:, 2, 0)
  return s:sum(a:x, l:axis, l:keepdims)
endfunction

function! s:sum(x, axis, keepdims)abort
  let l:axis = s:as_list(a:axis)

  let l:x_dim = len(a:x.shape)
  call map(l:axis, 'v:val < 0 ? v:val + l:x_dim : v:val')
  call map(l:axis, 'v:val >= l:x_dim ? l:x_dim - 1 : v:val')

  let l:fn = autograd#Function('s:sum')
  let l:fn['axis'] = uniq(sort(l:axis))
  let l:fn['keepdims'] = a:keepdims
  return l:fn.apply(a:x)
endfunction

function! s:sum_forward(xs) dict abort
  let l:x = a:xs[0]

  " all sum (e.g. (2, 3, 4) -> (1))
  if empty(self.axis) || len(self.axis) == len(l:x.shape)
    let l:total = 0
    for l:e in l:x.data
      let l:total += l:e
    endfor

    if !self.keepdims
      return [s:Tensor([l:total], [1])]
    endif
    return [s:Tensor([l:total], repeat([1], len(l:x.shape)))]
  endif

  " left side sum (e.g. (2, 3, 4) -> (3, 4))
  if self.axis[0] == 0
    let l:reduced_shape = l:x.shape[len(self.axis):]
    let l:s = s:left_side_sum_to(l:x, l:reduced_shape)

    if self.keepdims
      let l:s.shape = repeat([1], len(self.axis)) + l:reduced_shape
    endif
    return [l:s]
  endif

  " right side sum (e.g. (2, 3, 4) -> (2, 3)
  if self.axis[-1] == (len(l:x.shape) - 1)
    let l:reduced_shape = l:x.shape[:-len(self.axis) - 1]
    let l:s = s:right_side_sum_to(l:x, l:reduced_shape)

    if self.keepdims
      let l:s.shape = l:reduced_shape + repeat([1], len(self.axis))
    endif
    return [l:s]
  endif

  call s:error('intermediate or sparse axis sums are not supported.')
  return
endfunction

function! s:sum_backward(gys) dict abort
  return [s:broadcast_to(a:gys[0], self.inputs[0].shape)]
endfunction


" ex) [1, 5, 6, 1, 1, 1] -> [1, 5, 6]
function! s:left_valid_shape(shape) abort
  let l:dim = len(a:shape)
  let l:valid_size = l:dim
  for l:i in range(-1, -l:dim, -1)
    if a:shape[l:i] != 1
      break
    endif
    let l:valid_size -= 1
  endfor

  if l:valid_size == 0
    return [1]
  endif

  return a:shape[:l:valid_size - 1]
endfunction

" ex [1, 1, 1, 6, 7, 1] -> [6, 7, 1]
function! s:right_valid_shape(shape) abort
  let l:dim = len(a:shape)
  let l:valid_size = l:dim
  for l:i in range(l:dim)
    if a:shape[l:i] != 1
      break
    endif
    let l:valid_size -= 1
  endfor

  if l:valid_size == 0
    return [1]
  endif

  return a:shape[-l:valid_size:]
endfunction


function! autograd#broadcast_to(x, shape) abort
  return s:broadcast_to(a:x, a:shape)
endfunction

function! s:broadcast_to(x, shape) abort
  let l:xt = autograd#as_tensor(a:x)
  if l:xt.shape == a:shape
    return l:xt
  endif

  let l:fn = autograd#Function('s:broadcast_to')
  let l:fn['shape'] = a:shape
  return l:fn.apply(a:x)
endfunction

function! s:broadcast_to_forward(xs) dict abort
  let l:x = a:xs[0]
  let l:x_dim = len(l:x.shape)

  if l:x_dim > len(self.shape)
    call s:error(
      \ 'cannot broadcast the array of ' .
      \ string(l:x.shape) . ' to ' . string(self.shape))
  endif

  let l:size = s:shape_to_size(self.shape)

  " left side broadcast
  let l:right_subshape = s:right_valid_shape(l:x.shape)
  if l:right_subshape == [1]
    return [s:Tensor(repeat(l:x.data, l:size), self.shape)]
  endif
  if self.shape[-len(l:right_subshape):] == l:right_subshape
    let l:repeat = float2nr(l:size / len(l:x.data))
    return [s:Tensor(repeat(l:x.data, l:repeat), self.shape)]
  endif

  " right side broadcast
  let l:left_subshape = s:left_valid_shape(l:x.shape)
  if self.shape[:len(l:left_subshape) - 1] == l:left_subshape
    let l:repeat = float2nr(l:size / len(l:x.data))
    return [s:Tensor(flatten(
      \ mapnew(l:x.data, 'repeat([v:val], l:repeat)')), self.shape)]
  endif

  call s:error(
    \ 'cannot broadcast array of shape ' .
    \ string(l:x.shape) . ' into ' . string(self.shape))
endfunction

function! s:broadcast_to_backward(gys) dict abort
  return [s:sum_to(a:gys[0], self.inputs[0].shape)]
endfunction


function! autograd#sum_to(x, shape) abort
  return s:sum_to(a:x, a:shape)
endfunction

function! s:sum_to(x, shape) abort
  let l:xt = autograd#as_tensor(a:x)
  if l:xt.shape == a:shape
    return l:xt
  endif
  let l:fn = autograd#Function('s:sum_to')
  let l:fn['shape'] = a:shape
  return l:fn.apply(a:x)
endfunction

function! s:sum_to_forward(xs) dict abort
  let l:x = a:xs[0]
  let l:y = autograd#zeros(self.shape)

  let l:y_dim = len(self.shape)

  " left side sum
  let l:right_subshape = s:right_valid_shape(self.shape)
  if l:right_subshape == [1] || l:x.shape[-len(right_subshape):] == l:right_subshape
    return [s:left_side_sum_to(l:x, self.shape)]
  endif

  " right side sum
  let l:left_subshape = s:left_valid_shape(self.shape)
  if l:x.shape[:len(l:left_subshape) - 1] == l:left_subshape
    return [s:right_side_sum_to(l:x, self.shape)]
  endif

  call s:error('cannot sum from ' . string(l:x.shape) . ' into ' . string(self.shape))
endfunction

function! s:sum_to_backward(gys) dict abort
  return [s:broadcast_to(a:gys[0], self.inputs[0].shape)]
endfunction


function! autograd#fmax(list_obj) abort
  let l:max = 0.0
  for l:x in a:list_obj
    if l:max < l:x
      let l:max = l:x
    endif
  endfor
  return l:max
endfunction

function! autograd#max(x) abort
  return s:max(a:x)
endfunction

function! s:max(x) abort
  return autograd#Function('s:max').apply(a:x)
endfunction

function! s:max_forward(xs) dict abort
  return [s:Tensor([autograd#fmax(a:xs[0].data)], [1])]
endfunction

function! s:max_backward(gys) dict abort
  let l:x = self.inputs[0]
  let l:y = self.outputs[0]
  let l:gx_mask = autograd#elementwise([l:x, l:y], {a, b -> 1.0 * (a == b)})
  let l:gx = autograd#mul(a:gys[0], l:gx_mask)
  return [l:gx]
endfunction


function! autograd#maximum(a, b) abort
  return s:maximum(a:a, a:b)
endfunction

function! s:maximum(a, b) abort
  return autograd#Function('s:maximum').apply(a:a, a:b)
endfunction

function! s:maximum_forward(xs) dict abort
  return [autograd#elementwise(a:xs, {a, b -> a >= b ? a : b})]
endfunction

function! s:maximum_backward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]

  let l:gx0_mask = autograd#elementwise([l:x0, l:x1], {a, b -> a >= b})
  let l:gx1_mask = autograd#elementwise([l:x0, l:x1], {a, b -> a < b})

  let l:gx0 = autograd#mul(a:gys[0], l:gx0_mask)
  let l:gx1 = autograd#mul(a:gys[0], l:gx1_mask)

  return [s:sum_to(l:gx0, l:x0.shape), s:sum_to(l:gx1, l:x1.shape)]
endfunction


function! autograd#transpose(x) abort
  return s:transpose(a:x)
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

  let l:out_data = s:vector(len(l:xd))

  let l:n_i = a:x.shape[0]
  let l:n_j = a:x.shape[1]

  let l:n_j_range = range(l:n_j)
  for l:i in range(l:n_i)
    let l:buf = l:i * l:n_j
    for l:j in l:n_j_range
      let l:out_data[l:j * l:n_i + l:i] = l:xd[l:buf + l:j]
    endfor
  endfor

  return s:Tensor(l:out_data, [l:n_j, l:n_i])
endfunction

function! s:transpose(x) abort
  return autograd#Function('s:transpose').apply(a:x)
endfunction

function! s:transpose_forward(xs) dict abort
  return [s:_transpose(a:xs[0])]
endfunction

function! s:transpose_backward(gys) dict abort
  return [s:transpose(a:gys[0])]
endfunction


function! autograd#matmul(a, b) abort
  return s:matmul(a:a, a:b)
endfunction

function! s:matmul(a, b) abort
  return autograd#Function('s:matmul').apply(a:a, a:b)
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

  let l:out = autograd#zeros([l:n_i, l:n_j])

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


function! autograd#reshape(x, shape) abort
  return s:reshape(a:x, a:shape)
endfunction

function! s:reshape(x, shape) abort
  if a:x.shape == a:shape
    return a:x
  endif

  let l:fn = autograd#Function('s:reshape')
  let l:fn['shape'] = a:shape
  return l:fn.apply(a:x)
endfunction

function! s:reshape_forward(xs) dict abort
  let l:x = a:xs[0]
  if s:shape_to_size(self.shape) != len(l:x.data)
    call s:error('Cannot reshape array of size ' . len(l:x.data). ' into ' . string(self.shape))
    return
  endif
  return [s:Tensor(l:x.data, self.shape)]
endfunction

function! s:reshape_backward(gys) dict abort
  return [s:reshape(a:gys[0], self.inputs[0].shape)]
endfunction


function! autograd#flatten(x) abort
  return s:reshape(a:x, [len(a:x.data)])
endfunction


function! autograd#pi() abort
  return s:pi
endfunction

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


let s:NoGrad = {'state': 0}

function! s:NoGrad.end() abort
  let s:enable_backprop = self.state
endfunction

function! autograd#no_grad() abort
  let l:h = deepcopy(s:NoGrad)
  let l:h.state = s:enable_backprop
  let s:enable_backprop = 0
  return l:h
endfunction


function! autograd#elementwise(inputs, func, ...) abort
  let l:out = get(a:, 1, {})

  if len(a:inputs) == 1
    let l:x = a:inputs[0]
    let l:tensor = empty(l:out) ? autograd#zeros_like(l:x) : l:out

    let l:td = l:tensor.data
    let l:xd = l:x.data
    for l:i in range(len(l:xd))
      let l:td[l:i] = a:func(l:xd[l:i])
    endfor
    return l:tensor
  endif

  let l:x0 = a:inputs[0]
  let l:x1 = a:inputs[1]

  let l:ng = autograd#no_grad()

  let l:x0_dim = len(l:x0.shape)
  let l:x1_dim = len(l:x1.shape)

  if l:x0_dim > l:x1_dim
    let l:x1 = s:broadcast_to(l:x1, l:x0.shape)
  elseif x0_dim < l:x1_dim
    let l:x0 = s:broadcast_to(l:x0, l:x1.shape)
  else
    if len(l:x0.data) > len(l:x1.data)
      let l:x1 = s:broadcast_to(l:x1, l:x0.shape)
    else
      let l:x0 = s:broadcast_to(l:x0, l:x1.shape)
    endif
  endif
  call l:ng.end()

  let l:tensor = empty(l:out) ? autograd#zeros_like(l:x0) : l:out

  let l:td = l:tensor.data
  let l:x0d = l:x0.data
  let l:x1d = l:x1.data

  for l:i in range(len(l:tensor.data))
    let l:td[l:i] = a:func(l:x0d[l:i], l:x1d[l:i])
  endfor
  return l:tensor
endfunction
