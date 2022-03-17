function! s:test_add() abort
  call assert_equal(1, 1)
endfunction

function! s:test_mul() abort
  call assert_equal(1, 1)
endfunction


function! s:test_generation() abort
  let x = autograd#tensor(2.0)

  let y = autograd#add(x.p(2).p(2), x.p(2).p(2))
  call y.backward()

  call assert_equal(32.0, y.data)
  call assert_equal(64.0, x.grad.data)
endfunction


function! s:test_higer_order_differential() abort
  let x = autograd#tensor(2.0)
  call assert_equal(2.0, x.data)

  " y = x^5 - 3*x^3 + 1
  let y = x.p(5).s(x.p(3).m(3)).a(1)
  call assert_equal(9.0, y.data)

  call y.backward()
  call assert_equal(44.0, x.grad.data)

  let gx = x.grad
  call x.zero_grad()
  call gx.backward()
  call assert_equal(124.0, x.grad.data)

  let gx = x.grad
  call x.zero_grad()
  call gx.backward()
  call assert_equal(222.0, x.grad.data)

  let gx = x.grad
  call x.zero_grad()
  call gx.backward()
  call assert_equal(240.0, x.grad.data)

  let gx = x.grad
  call x.zero_grad()
  call gx.backward()
  call assert_equal(120.0, x.grad.data)
endfunction


function! s:test_goldstein_price() abort
  let x = autograd#tensor(1.0)
  call assert_equal(1.0, x.data)

  let y = autograd#tensor(1.0)
  call assert_equal(1.0, y.data)

  let t1 = x.a(y.a(1)).p(2)
  let t2 = x.m(-14).a(19).a(x.p(2).m(3)).s(y.m(14)).a(x.m(y.m(6))).a(y.p(2).m(3))
  let t3 = x.m(2).s(y.m(3)).p(2)
  let t4 = x.m(-32).a(18).a(x.p(2).m(12)).a(y.m(48)).s(x.m(y.m(36))).a(y.p(2).m(27))

  let z = autograd#mul(t1.m(t2).a(1), t3.m(t4).a(30))
  call assert_equal(1876.0, z.data)

  call z.backward()
  call assert_equal(-5376.0, x.grad.data)
  call assert_equal(8064.0, y.grad.data)
endfunction


function! test_basic#run_test_suite() abort
  call s:test_generation()
  call s:test_higer_order_differential()
  call s:test_goldstein_price()
endfunction
