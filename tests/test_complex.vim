function! s:goldstein_price(inputs) abort
  let x = autograd#as_tensor(a:inputs[0])
  let y = autograd#as_tensor(a:inputs[1])

  let t1 = x.a(y.a(1)).p(2)
  let t2 = x.m(-14).a(19).a(x.p(2).m(3)).s(y.m(14)).a(x.m(y.m(6))).a(y.p(2).m(3))
  let t3 = x.m(2).s(y.m(3)).p(2)
  let t4 = x.m(-32).a(18).a(x.p(2).m(12)).a(y.m(48)).s(x.m(y.m(36))).a(y.p(2).m(27))

  return autograd#mul(t1.m(t2).a(1), t3.m(t4).a(30))
endfunction

function! s:test_goldstein_price() abort
  let x = autograd#tensor(1.0)
  call assert_equal([1.0], x.data)

  let y = autograd#tensor(1.0)
  call assert_equal([1.0], y.data)

  let z = s:goldstein_price([x, y])
  call assert_equal([1876.0], z.data)

  call z.backward()
  call assert_equal([-5376.0], x.grad.data)
  call assert_equal([8064.0], y.grad.data)
endfunction

function! s:test_goldstein_price_gradcheck() abort
  let x = autograd#uniform(0, 10, [2, 3])
  let y = autograd#uniform(0, 10, [2, 3])

  call autograd#gradcheck(function('s:goldstein_price'), [x, y])
endfunction


function! test_complex#run_test_suite() abort
  call s:test_goldstein_price()
  call s:test_goldstein_price_gradcheck()
endfunction
