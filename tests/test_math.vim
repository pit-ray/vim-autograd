function! s:test_log() abort
  let F = {xs -> autograd#log(xs[0])}
  let x = autograd#tensor(autograd#rand() * 100)
  call autograd#gradcheck(F, [x])
endfunction


function! test_math#run_test_suite() abort
  call s:test_log()
endfunction
