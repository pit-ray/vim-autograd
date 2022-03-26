function! s:f(x) abort
  " y = x^5 - 2x^3 + 4x^2 + 6x + 5
  let t1 = a:x.p(5)
  let t2 = a:x.p(3).m(2).n()
  let t3 = a:x.p(2).m(4)
  let t4 = a:x.m(6)
  let t5 = 5
  let y = t1.a(t2).a(t3).a(t4).a(t5)
  return y
endfunction

function! s:main() abort
  let x = autograd#tensor(2.0)
  let y = s:f(x)
  echo 'y  :' y.data

  " gx1 = 5x^4 - 6x^2 + 8x + 6
  let gx1 = autograd#grad(y, x, 1)
  echo 'gx1:' gx1.data

  " gx2 = 20x^3 - 12x + 8
  let gx2 = autograd#grad(gx1, x, 1)
  echo 'gx2:' gx2.data

  " gx3 = 60x^2 - 12
  call gx2.backward(1)
  echo 'gx3:' x.grad.data
endfunction

call s:main()
