vim9script

if has('win32')
  set shellslash
endif

var tests_root = expand("<sfile>:p:h")
var project_root = tests_root .. '/..'

var test_suite_runner = 'RunTestSuite'
var test_result_file = 'test.log'


import './test_basic.vim'
import './test_complex.vim'
import './test_higher_order.vim'
import './test_math.vim'
import './test_reshape.vim'
import './test_tensor.vim'
import './test_util.vim'

const test_suites: list<func> = [
  test_basic.RunTestSuite,
  test_complex.RunTestSuite,
  test_higher_order.RunTestSuite,
  test_math.RunTestSuite,
  test_reshape.RunTestSuite,
  test_tensor.RunTestSuite,
  test_util.RunTestSuite
]


def Test()
  v:errors = []

  for RunSuite in test_suites
    RunSuite()
  endfor

  var e_len = len(v:errors)

  if e_len > 0
    var error_messages = []

    for i in range(e_len)
      var e = v:errors[i]
      var e_msg = test_suite_runner .. e->split(test_suite_runner)[1]

      for suite_name in test_suites
        if stridx(e, string(suite_name)) != -1
          e_msg = string(suite_name) .. e_msg
        endif
      endfor

      error_messages->add( '[' .. (i + 1) .. '/' .. e_len .. '] ' .. e_msg)
    endfor

    writefile(error_messages, test_result_file)
    cquit!
  endif

  writefile(['test passed!'], test_result_file)
  qall!
enddef

Test()
