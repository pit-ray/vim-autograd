function! s:test_add() abort
  let F = {xs -> autograd#add(xs[0], xs[1])}

  let x0 = autograd#uniform(0, 100, [2, 3])
  let x1 = autograd#uniform(0, 100, [2, 3])

  call autograd#gradcheck(F, [x0, x1])
endfunction

function! s:test_mul() abort
  let F = {xs -> autograd#mul(xs[0], xs[1])}

  let x0 = autograd#uniform(0, 100, [2, 3])
  let x1 = autograd#uniform(0, 100, [2, 3])

  call autograd#gradcheck(F, [x0, x1])

  let x2 = autograd#uniform(0, 100, [2, 3])
  let x3 = autograd#tensor([10])
  call autograd#gradcheck(F, [x2, x3])
endfunction

function! s:test_sub() abort
  let F = {xs -> autograd#sub(xs[0], xs[1])}

  let x0 = autograd#uniform(0, 100, [2, 3])
  let x1 = autograd#uniform(0, 100, [2, 3])

  call autograd#gradcheck(F, [x0, x1])
endfunction

function! s:test_div() abort
  let F = {xs -> autograd#div(xs[0], xs[1])}

  let x0 = autograd#uniform(0, 100, [2, 3])
  let x1 = autograd#uniform(1, 100, [2, 3])

  call autograd#gradcheck(F, [x0, x1])
endfunction

function! s:test_pow() abort
  let F = {xs -> autograd#pow(xs[0], xs[1])}

  let x0 = autograd#uniform(0, 10, [2, 3])
  let x1 = autograd#uniform(0, 10, [2, 3])

  call autograd#gradcheck(F, [x0, x1])
endfunction


function! test_basic#run_test_suite() abort
  call s:test_add()
  call s:test_mul()
  call s:test_sub()
  call s:test_div()
  call s:test_pow()
endfunction
