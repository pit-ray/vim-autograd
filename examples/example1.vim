function! s:f(x) abort
  " y = x^5 - 2x^3
  let y = autograd#sub(a:x.p(5), a:x.p(3).m(2))
  return y
endfunction

function! s:example1() abort
  let x = autograd#tensor(2.0)

  let y = s:f(x)
  call y.backward()

  " output: 56
  echo x.grad.data
endfunction

call s:example1()
