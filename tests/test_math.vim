function! s:test_log() abort
  let F = {xs -> autograd#log(xs[0])}
  let x = autograd#tensor(autograd#rand() * 100)
  call autograd#gradcheck(F, [x])
endfunction

function! s:test_exp() abort
  let F = {xs -> autograd#exp(xs[0])}
  let x = autograd#tensor(autograd#rand() * 10)
  call autograd#gradcheck(F, [x])
endfunction

function! s:test_sin() abort
  let F = {xs -> autograd#sin(xs[0])}
  let x = autograd#tensor(autograd#rand() * autograd#pi() * 2)
  call autograd#gradcheck(F, [x])
endfunction

function! s:test_cos() abort
  let F = {xs -> autograd#cos(xs[0])}
  let x = autograd#tensor(autograd#rand() * autograd#pi() * 2)
  call autograd#gradcheck(F, [x])
endfunction

function! s:test_tanh() abort
  let F = {xs -> autograd#tanh(xs[0])}
  let x = autograd#tensor(autograd#rand())
  call autograd#gradcheck(F, [x])
endfunction

function! s:test_abs() abort
  let x1 = autograd#tensor([0.7, -1.2, 0.0, 2.3])

  let y1 = autograd#abs(x1)
  call assert_equal([0.7, 1.2, 0.0, 2.3], y1.data)

  call y1.backward()
  call assert_equal([1.0, -1.0, 0.0, 1.0], x1.grad.data)
endfunction

function! s:test_sign() abort
  let x1 = autograd#tensor([0.2, 1.5, -5.6, 0.0])

  let y1 = autograd#sign(x1)
  call assert_equal([1.0, 1.0, -1.0, 0.0], y1.data)

  call y1.backward()
  call assert_equal([0.0, 0.0, 0.0, 0.0], x1.grad.data)
endfunction

function! test_math#run_test_suite() abort
  call s:test_log()
  call s:test_exp()
  call s:test_sin()
  call s:test_cos()
  call s:test_tanh()
  call s:test_abs()
  call s:test_sign()
endfunction
