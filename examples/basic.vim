function! s:f(x) abort
  " y = x^5 - 2x^3
  let y = autograd#sub(a:x.p(5), a:x.p(3).m(2))
  return y
endfunction

function! s:main() abort
  let x = autograd#tensor(2.0)

  let y = s:f(x)
  call y.backward()

  echo x.grad.data

  let x.name = 'x'
  let y.name = 'y'
  call autograd#utils#dump_graph(y, '.autograd/example1.png')
endfunction

call s:main()
