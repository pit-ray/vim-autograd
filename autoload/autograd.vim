" Tensor
let s:Tensor = {
  \ 'name': '',
  \ 'data': v:none,
  \ 'grad': {},
  \ 'parent': {},
  \ 'gen': 0,
  \ }

function! s:Tensor.ZeroGrad() abort
  let self.grad = {}
endfunction

function! s:Tensor.SetParent(parent_link) abort
  let self.parent = a:parent_link
  let self.gen = self.parent.gen + 1
endfunction

function! s:TensorGenComp(lhs, rhs) abort
  if lhs['gen'] == rhs['gen']
    return 0
  elseif lhs['gen'] < rhs['gen']
    return -1
  else
    return 1
  endif
endfunction

function! s:Tensor.Backward() abort
  if empty(self.grad)
    let self.grad = s:Tensor(1.0)
  endif

  if empty(self.parent)
    return
  endif

  let l:links = [self.parent]

  while len(l:links) > 0
    let l:link = remove(l:links, -1)

    let l:x_grads = l:link.Backward()

    let l:x_len = len(l:x_grads)
    for l:i in range(l:x_len)
      let l:input = l:link.inputs[l:i]
      if empty(l:input.grad)
        let l:input.grad = l:x_grads[l:i]
      else
        let l:input.grad = s:Add(l:input.grad, l:x_grads[l:i])
      endif

      if !empty(l:input.parent)
        call add(l:links, l:input.parent)
      endif
    endfor

    call sort(l:links, function('s:TensorGenComp'))
  endwhile
endfunction

function! s:Tensor(data) abort
  let l:tensor = deepcopy(s:Tensor)
  let l:tensor.data = a:data
  return l:tensor
endfunction

function! s:IsTensor(x) abort
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
  \ 'Forward': v:null,
  \ 'Backward': v:null
  \ }

function! s:Link.Call(...) abort
  let self.inputs = []
  for l:input in a:000
    call add(self.inputs, s:IsTensor(l:input) ? l:input : s:Tensor(l:input))
  endfor

  let self.outputs = self.Forward(self.inputs)

  let l:gens = []
  for l:input in self.inputs
    call add(l:gens, l:input.gen)
  endfor
  let self.gen = max(l:gens)

  for l:output in self.outputs
    call l:output.SetParent(self)
  endfor

  return len(self.outputs) > 1 ? self.outputs : self.outputs[0]
endfunction

function! s:Link() abort
  return deepcopy(s:Link)
endfunction


" Operations

function! s:Add_Forward(inputs) dict abort
  return [s:Tensor(a:inputs[0].data + a:inputs[1].data)]
endfunction

function! s:Add_Backward() dict abort
  let l:gy = self.outputs[0].grad
  return [l:gy, l:gy]
endfunction

function! s:Add(x0, x1) abort
  let l:link = s:Link()

  let l:link.name = 'Add'
  let l:link.Forward = function('s:Add_Forward')
  let l:link.Backward = function('s:Add_Backward')

  return l:link.Call(a:x0, a:x1)
endfunction


function! s:Mul_Forward(inputs) dict abort
  return [s:Tensor(a:inputs[0].data * a:inputs[1].data)]
endfunction

function! s:Mul_Backward() dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]
  let l:gy = self.outputs[0].grad
  return [s:Mul(l:x1, l:gy), s:Mul(l:x0, l:gy)]
endfunction

function! s:Mul(x0, x1) abort
  let l:link = s:Link()

  let l:link.name = 'Mul'
  let l:link.Forward = function('s:Mul_Forward')
  let l:link.Backward = function('s:Mul_Backward')

  return l:link.Call(a:x0, a:x1)
endfunction


function! s:test1() abort
  let l:x0 = s:Tensor(3)
  let l:x1 = s:Tensor(2)

  let l:t = s:Mul(l:x0, l:x1)
  echo l:t.data

  let l:x2 = s:Tensor(10)
  let l:y = s:Mul(l:t, l:x2)

  echo l:y.data
  call l:y.Backward()

  echo l:x0.grad.data l:x1.grad.data
endfunction

function! s:test2() abort
  let l:x = s:Tensor(3)
  echo 'x     :' l:x.data

  echo 'func  : y=5*x^2+4'
  let l:y = s:Add(s:Mul(5, s:Mul(l:x, l:x)), 4)
  echo 'y     :' l:y.data

  call l:y.Backward()
  echo 'x.grad:' l:x.grad.data
endfunction

call s:test2()
