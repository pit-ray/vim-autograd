function! s:test_log() abort
  let F = {xs -> autograd#log(xs[0])}
  let x = autograd#uniform(0, 100, [2, 3])
  call autograd#gradcheck(F, [x])
endfunction

function! s:test_exp() abort
  let F = {xs -> autograd#exp(xs[0])}
  let x = autograd#uniform(0, 10, [2, 3])
  call autograd#gradcheck(F, [x])
endfunction

function! s:test_sin() abort
  let F = {xs -> autograd#sin(xs[0])}
  let x = autograd#uniform(0.0, autograd#pi() * 2, [2, 3])
  call autograd#gradcheck(F, [x])
endfunction

function! s:test_cos() abort
  let F = {xs -> autograd#cos(xs[0])}
  let x = autograd#uniform(0.0, autograd#pi() * 2, [2, 3])
  call autograd#gradcheck(F, [x])
endfunction

function! s:test_tanh() abort
  let F = {xs -> autograd#tanh(xs[0])}
  let x = autograd#rand(2, 3)
  call autograd#gradcheck(F, [x])
endfunction

function! s:test_abs() abort
  let x1 = autograd#tensor([0.7, -1.2, 0.0, 2.3])

  let y1 = autograd#abs(x1)
  call assert_equal([0.7, 1.2, 0.0, 2.3], y1.data)

  call y1.backward()
  call assert_equal([1.0, -1.0, 0.0, 1.0], x1.grad.data)
endfunction

function! s:test_sign() abort
  let x1 = autograd#tensor([0.2, 1.5, -5.6, 0.0])

  let y1 = autograd#sign(x1)
  call assert_equal([1.0, 1.0, -1.0, 0.0], y1.data)

  call y1.backward()
  call assert_equal([0.0, 0.0, 0.0, 0.0], x1.grad.data)
endfunction

function! s:test_matmul() abort
  let x0 = autograd#tensor([[2, 4, 5], [4, 5, 6]])
  call assert_equal([2, 3], x0.shape)

  let x1 = autograd#tensor([[5, 7], [7, 8], [5, 7]])
  call assert_equal([3, 2], x1.shape)

  let y0 = autograd#matmul(x0, x1)
  call assert_equal([2, 2], y0.shape)
  call assert_equal([63.0, 81.0, 85.0, 110.0], y0.data)

  let y1 = autograd#matmul(x1, x0)
  call assert_equal([3, 3], y1.shape)
  call assert_equal([
    \ 38.0, 55.0, 67.0,
    \ 46.0, 68.0, 83.0,
    \ 38.0, 55.0, 67.0], y1.data)

  call y0.backward()
  let gx0 = x0.grad
  let gx1 = x1.grad

  call assert_equal([
    \ 12.0, 15.0, 12.0,
    \ 12.0, 15.0, 12.0], gx0.data)
  call assert_equal([2, 3], gx0.shape)

  call assert_equal([
    \ 6.0, 6.0, 9.0,
    \ 9.0, 11.0, 11.0], gx1.data)
  call assert_equal([3, 2], gx1.shape)

  let x2 = autograd#tensor([2, 4, 5])
  let x3 = autograd#tensor([5, 7, 9])
  let y2 = autograd#matmul(x2, x3)
  call assert_equal([1], y2.shape)
  call assert_equal([83.0], y2.data)

  call y2.backward()
  let gx2 = x2.grad
  let gx3 = x3.grad

  call assert_equal([5.0, 7.0, 9.0], gx2.data)
  call assert_equal([3], gx2.shape)

  call assert_equal([2.0, 4.0, 5.0], gx3.data)
  call assert_equal([3], gx3.shape)
endfunction


function! s:test_max() abort
  " case 1
  let x1 = autograd#tensor([[2.0, 2.1, 1.9], [3.1, 2.0, 3.0]])
  let y1 = autograd#max(x1)
  call assert_equal([3.1], y1.data)
  call assert_equal([1], y1.shape)

  call y1.backward()
  call assert_equal([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], x1.grad.data)
  call assert_equal([2, 3], x1.grad.shape)

  " case 2
  let x2 = autograd#tensor([5, 7, 9])
  let y2 = autograd#pow(autograd#max(x2), 2)
  call assert_equal([81.0], y2.data)
  call assert_equal([1], y2.shape)

  call y2.backward()
  call assert_equal([0.0, 0.0, 18.0], x2.grad.data)
  call assert_equal([3], x2.grad.shape)
endfunction


function! s:test_maximum() abort
  " case 1
  let x0 = autograd#tensor([1, 2, -1])
  let x1 = autograd#tensor([3, 0, 4])
  let y0 = autograd#maximum(x0, x1)
  call assert_equal([3.0, 2.0, 4.0], y0.data)
  call assert_equal([3], y0.shape)

  call y0.backward()
  call assert_equal([0.0, 1.0, 0.0], x0.grad.data)
  call assert_equal([3], x0.grad.shape)
  call assert_equal([1.0, 0.0, 1.0], x1.grad.data)
  call assert_equal([3], x1.grad.shape)

  " case 2
  let x2 = autograd#tensor([4, -1, 9])
  let x3 = autograd#tensor([5])
  let y2 = autograd#maximum(x2, x3)
  call assert_equal([5.0, 5.0, 9.0], y2.data)
  call assert_equal([3], y2.shape)

  call y2.backward()
  call assert_equal([0.0, 0.0, 1.0], x2.grad.data)
  call assert_equal([3], x2.grad.shape)
  call assert_equal([2.0], x3.grad.data)
  call assert_equal([1], x3.grad.shape)

  " case 3
  let x4 = autograd#tensor([[2, 4, 5], [6, -1, 5]])
  let x5 = autograd#tensor([3, 2, 5])
  let y4  = autograd#maximum(x4, x5)
  call assert_equal([3.0, 4.0, 5.0, 6.0, 2.0, 5.0], y4.data)
  call assert_equal([2, 3], y4.shape)

  call y4.backward()
  call assert_equal([0.0, 1.0, 1.0, 1.0, 0.0, 1.0], x4.grad.data)
  call assert_equal([2, 3], x4.grad.shape)
  call assert_equal([1.0, 1.0, 0.0], x5.grad.data)
  call assert_equal([3], x5.grad.shape)
endfunction


function! test_math#run_test_suite() abort
  call s:test_log()
  call s:test_exp()
  call s:test_sin()
  call s:test_cos()
  call s:test_tanh()
  call s:test_abs()
  call s:test_sign()
  call s:test_max()
  call s:test_matmul()
  call s:test_maximum()
endfunction
