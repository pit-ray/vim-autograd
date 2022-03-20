function! s:test_add() abort
  let F = {xs -> autograd#add(xs[0], xs[1])}

  let x0 = autograd#rand(2, 3).m(100).detach()
  let x1 = autograd#rand(2, 3).m(100).detach()

  call autograd#gradcheck(F, [x0, x1])
endfunction

function! s:test_mul() abort
  let F = {xs -> autograd#mul(xs[0], xs[1])}

  let x0 = autograd#rand(2, 3).m(100).detach()
  let x1 = autograd#rand(2, 3).m(100).detach()

  call autograd#gradcheck(F, [x0, x1])
endfunction

function! s:test_sub() abort
  let F = {xs -> autograd#sub(xs[0], xs[1])}

  let x0 = autograd#rand(2, 3).m(100).detach()
  let x1 = autograd#rand(2, 3).m(100).detach()

  call autograd#gradcheck(F, [x0, x1])
endfunction

function! s:test_div() abort
  let F = {xs -> autograd#div(xs[0], xs[1])}

  let x0 = autograd#rand(2, 3).m(100).detach()
  let x1 = autograd#rand(2, 3).m(100).a(1).detach()

  call autograd#gradcheck(F, [x0, x1])
endfunction

function! s:test_pow() abort
  let F = {xs -> autograd#pow(xs[0], xs[1])}

  let x0 = autograd#rand(2, 3).m(10).detach()
  let x1 = autograd#rand(2, 3).m(10).detach()

  call autograd#gradcheck(F, [x0, x1])
endfunction


function! test_basic#run_test_suite() abort
  call s:test_add()
  call s:test_mul()
  call s:test_sub()
  call s:test_div()
  call s:test_pow()
endfunction
