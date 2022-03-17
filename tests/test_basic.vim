function! s:test_add() abort
  call assert_equal(1, 1)
endfunction

function! s:test_mul() abort
  call assert_equal(1, 1)
endfunction


function! test_basic#run_test_suite() abort
  call s:test_add()
  call s:test_mul()
endfunction
