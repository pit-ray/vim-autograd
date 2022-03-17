function! s:test_add() abort
  call assert_equal(1, 1)
endfunction

function! s:test_mul() abort
  call assert_equal(1, 1)
endfunction


function! s:test_generation() abort
  let l:x = autograd#tensor(2.0)

  let l:y = autograd#add(l:x.p(2).p(2), l:x.p(2).p(2))
  call l:y.backward()

  call assert_equal(32.0, l:y.data)
  call assert_equal(64.0, l:x.grad.data)
endfunction


function! s:test_higer_order_differential() abort
  let l:x = autograd#tensor(2.0)
  call assert_equal(2.0, l:x.data)

  " y = x^5 - 3*x^3 + 1
  let l:y = l:x.p(5).s(l:x.p(3).m(3)).a(1)
  call assert_equal(9.0, l:y.data)

  call l:y.backward()
  call assert_equal(44.0, l:x.grad.data)

  let l:gx = l:x.grad
  call l:x.zero_grad()
  call l:gx.backward()
  call assert_equal(124.0, l:x.grad.data)

  let l:gx = l:x.grad
  call l:x.zero_grad()
  call l:gx.backward()
  call assert_equal(222.0, l:x.grad.data)

  let l:gx = l:x.grad
  call l:x.zero_grad()
  call l:gx.backward()
  call assert_equal(240.0, l:x.grad.data)

  let l:gx = l:x.grad
  call l:x.zero_grad()
  call l:gx.backward()
  call assert_equal(120.0, l:x.grad.data)
endfunction


function! s:test_goldstein_price() abort
  let l:x = autograd#tensor(1.0)
  call assert_equal(1.0, l:x.data)

  let l:y = autograd#tensor(1.0)
  call assert_equal(1.0, l:y.data)

  let l:t1 = l:x.a(l:y.a(1)).p(2)
  let l:t2 = l:x.m(-14).a(19).a(l:x.p(2).m(3)).s(l:y.m(14)).a(l:x.m(l:y.m(6))).a(l:y.p(2).m(3))
  let l:t3 = l:x.m(2).s(l:y.m(3)).p(2)
  let l:t4 = l:x.m(-32).a(18).a(l:x.p(2).m(12)).a(l:y.m(48)).s(l:x.m(l:y.m(36))).a(l:y.p(2).m(27))

  let l:z = autograd#mul(l:t1.m(l:t2).a(1), l:t3.m(l:t4).a(30))
  call assert_equal(1876.0, l:z.data)

  call l:z.backward()
  call assert_equal(-5376.0, l:x.grad.data)
  call assert_equal(8064.0, l:y.grad.data)
endfunction


function! test_basic#run_test_suite() abort
  call s:test_generation()
  call s:test_higer_order_differential()
  call s:test_goldstein_price()
endfunction
