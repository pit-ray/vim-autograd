" Tensor
let s:Tensor = {
  \ 'name': '',
  \ 'data': v:none,
  \ 'grad': {},
  \ 'parent': {},
  \ 'gen': 0,
  \ }

function! s:Tensor.zero_grad() abort
  let self.grad = {}
endfunction

function! s:Tensor.set_parent(parent_link) abort
  let self.parent = a:parent_link
  let self.gen = self.parent.gen + 1
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

function! s:Tensor.backward() abort
  if empty(self.grad)
    let self.grad = s:Tensor(1.0)
  endif

  if empty(self.parent)
    return
  endif

  let l:links = [self.parent]

  while len(l:links) > 0
    let l:link = remove(l:links, -1)

    let l:x_grads = l:link.backward()

    let l:x_len = len(l:x_grads)
    for l:i in range(l:x_len)
      let l:input = l:link.inputs[l:i]
      if empty(l:input.grad)
        let l:input.grad = l:x_grads[l:i]
      else
        let l:input.grad = s:add(l:input.grad, l:x_grads[l:i])
      endif

      if !empty(l:input.parent)
        call add(l:links, l:input.parent)
      endif
    endfor

    call sort(l:links, function('s:comp_tensor_gen'))
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


" Link
let s:Link = {
  \ 'name': '',
  \ 'inputs': [],
  \ 'outputs': [],
  \ 'gen': 0,
  \ 'forward': v:null,
  \ 'backward': v:null
  \ }

function! s:Link.call(...) abort
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
    call l:output.set_parent(self)
  endfor

  return len(self.outputs) > 1 ? self.outputs : self.outputs[0]
endfunction

function! s:Link(name) abort
  let l:link = deepcopy(s:Link)
  let l:link.name = a:name
  let l:link.forward = function(a:name . '_forward')
  let l:link.backward = function(a:name . '_backward')
  return l:link
endfunction


" Operations
function! s:add(x0, x1) abort
  return s:Link('s:add').call(a:x0, a:x1)
endfunction

function! s:add_forward(inputs) dict abort
  return [s:Tensor(a:inputs[0].data + a:inputs[1].data)]
endfunction

function! s:add_backward() dict abort
  let l:gy = self.outputs[0].grad
  return [l:gy, l:gy]
endfunction


function! s:mul(x0, x1) abort
  return s:Link('s:mul').call(a:x0, a:x1)
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
  return s:Link('s:sub').call(a:x0, a:x1)
endfunction

function! s:sub_forward(inputs) dict abort
  return [s:Tensor(self.inputs[0].data - self.inputs[1].data)]
endfunction

function! s:sub_backward() dict abort
  let l:gy = self.outputs[0].grad
  return [l:gy, l:gy.n()]
endfunction


function! s:div(x0, x1) abort
  return s:Link('s:div').call(a:x0, a:x1)
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
  return s:Link('s:pow').call(a:x, a:c)
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
