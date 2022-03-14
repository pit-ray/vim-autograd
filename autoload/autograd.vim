let s:Tensor = {
  \ 'name': v:none,
  \ 'data': v:none,
  \ 'grad': v:none,
  \ 'parent': v:none,
  \ 'gen': 0,
  \ }

function! s:Tensor.ClearGrad() abort
  let self.grad = v:none
endfunction

function! s:Tensor.SetParent(parent_link) abort
  let self.parent = a:parent_link
endfunction

function! s:Tensor.Backward() abort
endfunction


function! s:Tensor(data) abort
  let l:tensor = deepcopy(s:Tensor)
  let l:tensor.data = a:data
  return l:tensor
endfunction


let s:Link = {
  \ 'inputs': [],
  \ 'outputs': [],
  \ 'Forward': v:null,
  \ 'Backward': v:null
  \ }

function! s:Link.Call(...) abort
  let self.inputs = a:000

  let l:xs = []
  for l:x in self.inputs
    call add(l:xs, l:x.data)
  endfor

  let l:outputs = self.Forward(l:xs)
  if type(l:outputs) != v:t_list
    let l:outputs = [l:outputs]
  endif

  for l:i in range(len(l:outputs))
    let l:outputs[l:i] = s:Tensor(l:outputs[l:i])
  endfor

  for l:output in l:outputs
    let l:output
  endfor

  let self.outputs = l:outputs

  if len(l:outputs) > 1
    return l:outputs
  endif
  return l:outputs[0]
endfunction

function! s:Link() abort
  return deepcopy(s:Link)
endfunction


function! s:Mul_Forward(inputs) dict abort
  return a:inputs[0] * a:inputs[1]
endfunction

function! s:Mul_Backward(gy) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]
  return [s:Mul(l:x0, gy), s:Mul(l:x1, gy)]
endfunction


function! s:Mul(x0, x1) abort
  let l:link = s:Link()

  let l:link.Forward = function('s:Mul_Forward')
  let l:link.Backward = function('s:Mul_Backward')

  call l:link.Call(a:x0, a:x1)
endfunction



function! s:test() abort
  let l:x0 = s:Tensor(10)
  let l:x1 = s:Tensor(2)

  call l:x0.ClearGrad()
  call l:x1.ClearGrad()

  let l:y = s:Mul(l:x0, l:x1)
  " l:y.Backward()

  " echo lx1.grad lx2.grad
endfunction

call s:test()
