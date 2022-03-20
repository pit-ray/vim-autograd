function! s:test_zerograd() abort
  let x = autograd#tensor(2.0)
  let y = x.m(60)
  call y.backward()
  call assert_equal([60.0], x.grad.data)

  call x.zero_grad()
  call assert_true(empty(x.grad))
endfunction

function! s:test_clone() abort
  let a = autograd#tensor(2.0)
  let b = a.clone()
  call assert_notequal(a.id, b.id)
endfunction

function! s:test_tensor() abort
  let a = autograd#tensor([[1, 2, 3], [4, 5, 6]])
  call assert_equal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], a.data)
  call assert_equal([2, 3], a.shape)
  call assert_equal(6, a.size)

  let b = autograd#tensor(5)
  call assert_equal([5.0], b.data)
  call assert_equal([1], b.shape)
  call assert_equal(1, b.size)
endfunction

function! s:test_as_tensor() abort
  let a = autograd#tensor([2, 4, 5])
  let ar = autograd#as_tensor(a)
  call assert_equal(a.id, ar.id)
  call assert_true(a is ar)

  let b = [5, 6, 7]
  let br = autograd#as_tensor(b)
  call assert_equal([5.0, 6.0, 7.0], br.data)
  call assert_equal([3], br.shape)
  call assert_equal(3, br.size)

  let c = 5
  let cr = autograd#as_tensor(c)
  call assert_equal([5.0], cr.data)
  call assert_equal([1], cr.shape)
  call assert_equal(1, cr.size)
endfunction

function! s:test_zeros() abort
  let d1 = float2nr(autograd#rand() * 10 + 1)
  let d2 = float2nr(autograd#rand() * 10 + 1)
  let d3 = float2nr(autograd#rand() * 10 + 1)

  let x = autograd#zeros([d1, d2, d3])
  call assert_equal([d1, d2, d3], x.shape)
  call assert_equal(d1 * d2 * d3, x.size)
  for l:i in range(x.size)
    call assert_equal(0.0, x.data[l:i])
  endfor
endfunction

function! s:test_zeros_like() abort
  let a = autograd#tensor(
    \ [
        \ [1, 5, 6],
        \ [2, 6, 4],
        \ [2, 6, 7]
    \ ])

  let b = autograd#zeros_like(a)
  call assert_notequal(a.id, b.id)
  call assert_equal(a.shape, b.shape)
  call assert_equal(a.size, b.size)
  for l:i in range(b.size)
    call assert_equal(0.0, b.data[l:i])
  endfor
endfunction

function! s:test_ones() abort
  let d1 = float2nr(autograd#rand() * 10 + 1)
  let d2 = float2nr(autograd#rand() * 10 + 1)
  let d3 = float2nr(autograd#rand() * 10 + 1)

  let x = autograd#ones([d1, d2, d3])
  call assert_equal([d1, d2, d3], x.shape)
  call assert_equal(d1 * d2 * d3, x.size)
  for l:i in range(x.size)
    call assert_equal(1.0, x.data[l:i])
  endfor
endfunction

function! s:test_ones_like() abort
  let a = autograd#tensor(
    \ [
        \ [2, 2, 6],
        \ [0, 3, 4],
        \ [7, 0, 5]
    \ ])

  let b = autograd#ones_like(a)
  call assert_notequal(a.id, b.id)
  call assert_equal(a.shape, b.shape)
  call assert_equal(a.size, b.size)
  for l:i in range(b.size)
    call assert_equal(1.0, b.data[l:i])
  endfor
endfunction

function! s:test_generation() abort
  let x = autograd#tensor(2.0)

  let y = autograd#add(x.p(2).p(2), x.p(2).p(2))
  call y.backward()

  call assert_equal([32.0], y.data)
  call assert_equal([64.0], x.grad.data)
endfunction


function! test_tensor#run_test_suite() abort
  call s:test_zerograd()
  call s:test_clone()
  call s:test_tensor()
  call s:test_as_tensor()
  call s:test_zeros()
  call s:test_zeros_like()
  call s:test_ones()
  call s:test_ones_like()
  call s:test_generation()
endfunction
