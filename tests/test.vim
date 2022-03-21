set nocompatible

if has('win32')
  set shellslash
endif

let s:tests_root = expand("<sfile>:p:h")
let s:project_root = s:tests_root . '/..'

let s:test_suite_runner = '#run_test_suite'

let s:test_result_file = 'test_result.log'

let s:test_suites = [
  \ 'test_basic',
  \ 'test_complex',
  \ 'test_higher_order',
  \ 'test_math',
  \ 'test_reshape',
  \ 'test_tensor',
  \ 'test_util'
  \ ]

function! s:test() abort
  let v:errors = []

  execute 'source' (s:project_root . '/autoload/autograd.vim')
  execute 'source' (s:project_root . '/autoload/autograd/utils.vim')

  for l:suite in s:test_suites
    execute 'source' s:tests_root . '/' . l:suite . '.vim'
    execute 'call' l:suite . s:test_suite_runner . '()'
  endfor

  let l:e_len = len(v:errors)

  if l:e_len > 0
    let l:error_messages = []

    for l:i in range(l:e_len)
      let l:e = v:errors[l:i]
      let l:e_msg = s:test_suite_runner . split(l:e, s:test_suite_runner)[1]

      for l:suite_name in s:test_suites
        if stridx(l:e, l:suite_name) != -1
          let l:e_msg = l:suite_name . l:e_msg
        endif
      endfor

      call add(l:error_messages, '[' . (l:i + 1) . '/' . l:e_len . '] ' . l:e_msg)
    endfor

    call writefile(l:error_messages, s:test_result_file)
    cquit!
  endif

  call writefile(['test passed!'], s:test_result_file)
  qall!
endfunction

call s:test()
