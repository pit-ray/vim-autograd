function! s:test_higer_order_differential() abort
  let x = autograd#tensor(2.0)
  call assert_equal([2.0], x.data)

  " y = x^5 - 3*x^3 + 1
  let y = x.p(5).s(x.p(3).m(3)).a(1)
  call assert_equal([9.0], y.data)

  call y.backward()
  call assert_equal([44.0], x.grad.data)

  let gx = x.grad
  call x.zero_grad()
  call gx.backward()
  call assert_equal([124.0], x.grad.data)

  let gx = x.grad
  call x.zero_grad()
  call gx.backward()
  call assert_equal([222.0], x.grad.data)

  let gx = x.grad
  call x.zero_grad()
  call gx.backward()
  call assert_equal([240.0], x.grad.data)

  let gx = x.grad
  call x.zero_grad()
  call gx.backward()
  call assert_equal([120.0], x.grad.data)
endfunction


function! test_higher_order#run_test_suite() abort
  call s:test_higer_order_differential()
endfunction
