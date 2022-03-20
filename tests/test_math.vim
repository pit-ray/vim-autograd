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

function! test_math#run_test_suite() abort
  call s:test_log()
  call s:test_exp()
  call s:test_sin()
  call s:test_cos()
  call s:test_tanh()
endfunction
