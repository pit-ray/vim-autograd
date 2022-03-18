function! s:test_add() abort
  let F = {xs -> autograd#add(xs[0], xs[1])}

  let x0 = autograd#tensor(autograd#rand() * 100)
  let x1 = autograd#tensor(autograd#rand() * 100)

  call autograd#gradcheck(F, [x0, x1])
endfunction

function! s:test_mul() abort
  let F = {xs -> autograd#mul(xs[0], xs[1])}

  let x0 = autograd#tensor(autograd#rand() * 100)
  let x1 = autograd#tensor(autograd#rand() * 100)

  call autograd#gradcheck(F, [x0, x1])
endfunction

function! s:test_sub() abort
  let F = {xs -> autograd#sub(xs[0], xs[1])}

  let x0 = autograd#tensor(autograd#rand() * 100)
  let x1 = autograd#tensor(autograd#rand() * 100)

  call autograd#gradcheck(F, [x0, x1])
endfunction

function! s:test_div() abort
  let F = {xs -> autograd#div(xs[0], xs[1])}

  let x0 = autograd#tensor(autograd#rand() * 100)
  let x1 = autograd#tensor(autograd#rand() * 100 + 1)

  call autograd#gradcheck(F, [x0, x1])
endfunction

function! s:test_pow() abort
  let F = {xs -> autograd#pow(xs[0], xs[1])}

  let x0 = autograd#tensor(autograd#rand() * 10)
  let x1 = autograd#tensor(autograd#rand() * 10)

  call autograd#gradcheck(F, [x0, x1])
endfunction


function! s:test_generation() abort
  let x = autograd#tensor(2.0)

  let y = autograd#add(x.p(2).p(2), x.p(2).p(2))
  call y.backward()

  call assert_equal(32.0, y.data)
  call assert_equal(64.0, x.grad.data)
endfunction


function! test_basic#run_test_suite() abort
  call s:test_add()
  call s:test_mul()
  call s:test_sub()
  call s:test_div()
  call s:test_pow()
  call s:test_generation()
endfunction
