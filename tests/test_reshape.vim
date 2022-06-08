function! s:test_sum() abort
  " case 1
  let x1 = autograd#tensor([[20, 40, 60], [1, 4, 5]])
  let y1 = autograd#sum(x1)
  call assert_equal([130.0], y1.data)
  call assert_equal([1], y1.shape)

  call y1.backward()
  call assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], x1.grad.data)
  call assert_equal([2, 3], x1.grad.shape)

  " case 2
  let x2 = autograd#tensor([[1, 2, 3], [4, 5, 6]])
  let y2 = autograd#sum(x2, 1)
  call assert_equal([2], y2.shape)
  call assert_equal([6.0, 15.0], y2.data)

  let y2_2 = autograd#sum(x2, 0)
  call assert_equal([3], y2_2.shape)
  call assert_equal([5.0, 7.0, 9.0], y2_2.data)

  let y2_3 = autograd#sum(x2, 1, 1)
  call assert_equal([2, 1], y2_3.shape)
  call assert_equal([6.0, 15.0], y2_3.data)

  call x2.cleargrad()
  call y2.backward()
  call assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], x1.grad.data)
  call assert_equal([2, 3], x1.grad.shape)

  " case 3
  let x3 = autograd#tensor([[[2, 3], [2, 4]], [[7, 8], [9, 6]]])
  call assert_equal([2, 2, 2], x3.shape)

  let y3_0 = autograd#sum(x3, 0)
  call assert_equal([2, 2], y3_0.shape)
  call assert_equal([9.0, 11.0, 11.0, 10.0], y3_0.data)

  call x3.cleargrad()
  call y3_0.backward()

  call assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], x3.grad.data)
  call assert_equal([2, 2, 2], x3.grad.shape)

  " case 4
  let y3_1 = autograd#sum(x3, 0, 1)
  call assert_equal([1, 2, 2], y3_1.shape)
  call assert_equal([9.0, 11.0, 11.0, 10.0], y3_1.data)

  call x3.cleargrad()
  call y3_1.backward()

  call assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], x3.grad.data)
  call assert_equal([2, 2, 2], x3.grad.shape)

  " case 5
  let y3_4 = autograd#sum(x3, [0, 1])
  call assert_equal([2], y3_4.shape)
  call assert_equal([20.0, 21.0], y3_4.data)

  call x3.cleargrad()
  call y3_4.backward()

  call assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], x3.grad.data)
  call assert_equal([2, 2, 2], x3.grad.shape)

  " case 6
  let y3_5 = autograd#sum(x3, [0, 1], 1)
  call assert_equal([1, 1, 2], y3_5.shape)
  call assert_equal([20.0, 21.0], y3_5.data)

  call x3.cleargrad()
  call y3_5.backward()

  call assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], x3.grad.data)
  call assert_equal([2, 2, 2], x3.grad.shape)

  " case 7
  let y3_6 = autograd#sum(x3, [1, 2])
  call assert_equal([2], y3_6.shape)
  call assert_equal([11.0, 30.0], y3_6.data)

  call x3.cleargrad()
  call y3_6.backward()

  call assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], x3.grad.data)
  call assert_equal([2, 2, 2], x3.grad.shape)

  " case 8
  let y3_7 = autograd#sum(x3, [1, 2], 1)
  call assert_equal([2, 1, 1], y3_7.shape)
  call assert_equal([11.0, 30.0], y3_7.data)

  call x3.cleargrad()
  call y3_7.backward()

  call assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], x3.grad.data)
  call assert_equal([2, 2, 2], x3.grad.shape)
endfunction

function! s:test_broadcast_to() abort
  let x1 = autograd#tensor([20])
  let y1 = autograd#broadcast_to(x1, [2, 2, 3])

  let y1_expect = [
    \ 20.0, 20.0, 20.0,
    \ 20.0, 20.0, 20.0,
    \
    \ 20.0, 20.0, 20.0,
    \ 20.0, 20.0, 20.0
    \ ]
  call assert_equal(y1_expect, y1.data)

  call assert_equal([2, 2, 3], y1.shape)

  call y1.backward()
  call assert_equal([12.0], x1.grad.data)
  call assert_equal([1], x1.shape)


  let x2 = autograd#uniform(10, 100, [2, 3])
  let y2 = autograd#broadcast_to(x2, [2, 4, 2, 3])
  call assert_equal([2, 4, 2, 3], y2.shape)
  call assert_equal(flatten(repeat(x2.data, 8)), y2.data)

  call y2.backward()
  call assert_equal([2, 3], x2.grad.shape)
  call assert_equal([8.0, 8.0, 8.0, 8.0, 8.0, 8.0], x2.grad.data)


  let x3 = autograd#uniform(10, 100, [1, 2, 3])
  let y3 = autograd#broadcast_to(x3, [3, 2, 3])
  call assert_equal([3, 2, 3], y3.shape)
  call assert_equal(flatten(repeat(x3.data, 3)), y3.data)

  call y3.backward()
  call assert_equal([1, 2, 3], x3.grad.shape)
  call assert_equal([3.0, 3.0, 3.0, 3.0, 3.0, 3.0], x3.grad.data)


  let x4 = autograd#uniform(10, 100, [1, 1, 2])
  let y4 = autograd#broadcast_to(x4, [3, 4, 2])
  call assert_equal([3, 4, 2], y4.shape)
  call assert_equal(flatten(repeat(x4.data, 12)), y4.data)

  call y4.backward()
  call assert_equal([1, 1, 2], x4.grad.shape)
  call assert_equal([12.0, 12.0], x4.grad.data)


  let x5 = autograd#randn(1, 3, 2, 1)
  let y5 = autograd#broadcast_to(x5, [1, 3, 2, 2])
  call assert_equal([1, 3, 2, 2], y5.shape)

  call y5.backward()
  call assert_equal([1, 3, 2, 1], x5.grad.shape)
  call assert_equal([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], x5.grad.data)


  let x6 = autograd#randn(3)
  let y6 = autograd#broadcast_to(x6, [1, 4, 3])
  call assert_equal([1, 4, 3], y6.shape)

  call y6.backward()
  call assert_equal([3], x6.grad.shape)
  call assert_equal([4.0, 4.0, 4.0], x6.grad.data)


  let x7 = autograd#randn(2, 3, 1, 1)
  let y7 = autograd#broadcast_to(x7, [2, 3, 5, 6])
  call assert_equal([2, 3, 5, 6], y7.shape)

  call y7.backward()
  call assert_equal([2, 3, 1, 1], x7.grad.shape)
  call assert_equal([30.0, 30.0, 30.0, 30.0, 30.0, 30.0], x7.grad.data)
endfunction

function! s:test_transpose() abort
  let x1 = autograd#tensor([[1, 2, 3], [4, 5, 6]])
  call assert_equal([2, 3], x1.shape)

  let y1 = autograd#transpose(x1)
  call assert_equal([3, 2], y1.shape)
  call assert_equal([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], y1.data)

  call y1.backward()
  call assert_equal([2, 3], x1.grad.shape)
  call assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], x1.grad.data)
endfunction

function! s:test_reshape() abort
  let x1 = autograd#randn(2, 3)
  call assert_equal([2, 3], x1.shape)

  let y1 = autograd#reshape(x1, [1, 6])
  call assert_equal([1, 6], y1.shape)
  call assert_equal(x1.data, y1.data)

  call y1.backward()
  call assert_equal([2, 3], x1.grad.shape)
  call assert_equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], x1.grad.data)

  let y2 = x1.reshape([3, 2])
  call assert_equal([3, 2], y2.shape)
  call assert_equal(x1.data, y2.data)

  let y3 = x1.reshape(6, 1)
  call assert_equal([6, 1], y3.shape)
  call assert_equal(x1.data, y3.data)
endfunction


function! test_reshape#run_test_suite() abort
  call s:test_sum()
  call s:test_broadcast_to()
  call s:test_transpose()
  call s:test_reshape()
endfunction
