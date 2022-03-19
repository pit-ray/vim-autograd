function! s:test_nograd() abort
  let x = autograd#tensor(4.0)

  call autograd#nograd_begin()
  let y = x.m(2).p(3).s(10)
  call autograd#nograd_end()

  call assert_equal(502.0, y.data)
endfunction

function! test_util#run_test_suite() abort
  call s:test_nograd()
endfunction
