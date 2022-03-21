function! s:isclose(a, b, ...) abort
  let l:rtol = get(a:, 1, 0.00001)
  let l:atol = get(a:, 2, 0.00000001)
  return abs(a:a - a:b) <= (l:atol + l:rtol * abs(a:b))
endfunction

function! s:allclose(a, b, ...) abort
  let l:rtol = get(a:, 1, 0.00001)
  let l:atol = get(a:, 2, 0.00000001)

  let l:results = autograd#elementwise(function('s:isclose'), [a:a, a:b])
  return min(l:results.data) == 1
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

function! autograd#utils#dump_graph(last_node, filepath) abort
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


function! autograd#utils#numerical_grad(f, x) abort
  let l:eps = autograd#tensor(0.000001)
  let l:dx = autograd#tensor(l:eps.data[0] * 2)

  let l:x0 = autograd#sub(a:x, l:eps)
  let l:x1 = autograd#add(a:x, l:eps)

  let l:y0 = a:f(l:x0)
  let l:y1 = a:f(l:x1)

  let l:dy = autograd#sub(l:y1, l:y0)
  return autograd#sum_to(autograd#div(l:dy, l:dx), a:x.shape)
endfunction

function! autograd#utils#gradcheck(f, inputs) abort
  let l:y = a:f(a:inputs)

  for l:x in a:inputs
    call l:x.cleargrad()
  endfor
  call l:y.backward()

  let l:grads = []
  for l:x in a:inputs
    call add(l:grads, l:x.grad)
  endfor

  let l:ng = autograd#no_grad()

  let l:input_num = len(a:inputs)
  for l:i in range(l:input_num)
    let l:before_args = l:i > 0 ? a:inputs[:l:i - 1] : []
    let l:after_args = l:i < l:input_num - 1 ? a:inputs[l:i + 1:] : []

    let l:num_grad = autograd#utils#numerical_grad(
      \ {x -> a:f(l:before_args + [x] + l:after_args)},
      \ a:inputs[l:i])

    call assert_true(s:allclose(l:grads[l:i], l:num_grad))
  endfor

  call l:ng.end()
endfunction
