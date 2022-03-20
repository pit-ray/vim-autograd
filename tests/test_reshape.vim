function! s:test_sum()
  let x1 = autograd#tensor([[20, 40, 60], [1, 4, 5]])
  let y1 = autograd#sum(x1)
  call assert_equal([130.0], x1.data)
  call assert_equal([1], x1.shape)
  call assert_equal(1, x1.size)

  call y1.backward()
  call assert_equal([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], x1.grad.data)
  call assert_equal([2, 3], x1.grad.shape)
  call assert_equal(6, x1.grad.size)
endfunction

function! s:test_broadcast_to()
  let x1 = autograd#tensor([20])
  let y1 = autograd#broadcast_to(x1, [2, 2, 3])

  let y1_expect = [
    \ [
      \ [20.0, 20.0, 20.0],
      \ [20.0, 20.0, 20.0]
    \ ],
    \ [
      \ [20.0, 20.0, 20.0],
      \ [20.0, 20.0, 20.0]
    \ ]]
  call assert_equal(y1_expect, y1.data)

  call assert_equal([2, 2, 3], y1.shape)
  call assert_equal(12, y1.size)

  call y1.backward()
  call assert_equal([12.0], x1.grad.data)
  call assert_equal([1], x1.shape)
  call assert_equal(1, x1.size)
endfunction


function! test_reshape#run_test_suite() abort
  call s:test_sum()
  call s:test_broadcast_to()
endfunction

