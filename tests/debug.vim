function! s:debug() abort
  let l:x = autograd#tensor(3)
  echo 'x     :' l:x.data

  echo 'func  : y = 0.5*x^2 - 5*x + 3'
  " let l:y = autograd#add(autograd#mul(5, autograd#pow(l:x, 2)), 4)
  let l:y = autograd#add(autograd#sub(autograd#mul(0.5, autograd#pow(l:x, 2)), autograd#mul(5, l:x)), 3)
  echo 'y     :' l:y.data

  call l:y.backward(0)
  echo 'x.grad:' l:x.grad.data


  let l:gh = {'key': {'second': 'aaa'}}
  let l:v = l:gh
  echo l:gh l:v

  let l:v2 = l:v.key
  let l:v2.second = {}
  echo l:gh l:v

  let l:l1 = {'aa': 5}
  let l:l2 = {'aa': 5}
  let l:l3 = l:l1
  echo l:l1 is l:l2
  echo l:l1 is l:l3

  let x = autograd#tensor(2.0)
  call assert_equal(2.0, x.data)

  echo '0 diff'
  " y = x^5 - 3*x^3 + 1
  let y = x.p(5).s(x.p(3).m(3)).a(1)
  call assert_equal(9.0, y.data)

  echo '1 diff'
  call y.backward()
  call assert_equal(44.0, x.grad.data)

  echo '2 diff'
  let gx = x.grad
  call x.zero_grad()
  call gx.backward()
  call assert_equal(124.0, x.grad.data)

  echo '3 diff'
  let gx = x.grad
  call x.zero_grad()
  call gx.backward()
  call assert_equal(222.0, x.grad.data)

endfunction

call s:debug()
