vim9script

import '../core/function.vim'
import '../core/tensor.vim'


def DumpTensorAsDotlang(x: tensor.Tensor): string
  var data: string = '[' ..  string(x.data[0])
  if len(x.data) > 1
    data = data .. ', ...'
  endif
  data = data .. ']'

  var label = 
    x.name .. '\n' ..
    data .. '\n' ..
    'shape: ' .. string(x.shape)

  return x.id ..  '[label="' .. label ..
    '", color=lightblue, style=filled]'
enddef


def DumpFuncAsDotlang(fn: function.Function): list<string>
  var label = fn.name
  var fndef =
    fn.id .. '[label="' .. label ..
    '", color=gray, style=filled, shape=box]'

  var links = [fndef]
  for x in fn.inputs
    links->add(x.id .. ' -> ' .. fn.id)
  endfor

  for y in fn.outputs
    links->add(fn.id .. ' -> ' .. y.id)
  endfor

  return links
enddef


export def DumpGraph(last_node: tensor.Tensor, filepath: string)
  var defs: list<string> = [DumpTensorAsDotlang(last_node)]
  var links = []
  var funcs: list<function.Function> = []

  if last_node.parent_fn != null
    funcs->add(last_node.parent_fn)
  endif

  while len(funcs) > 0
    var fn = funcs->remove(-1)
    var fn_dot = DumpFuncAsDotlang(fn)
    defs->add(fn_dot[0])
    links += fn_dot[1 : ]

    for x in fn.inputs
      defs->add(DumpTensorAsDotlang(x))

      if x.parent_fn != null
        funcs->add(x.parent_fn)
      endif
    endfor
  endwhile

  links = uniq(sort(links))

  var texts = ['digraph g {'] + defs + links + ['}']

  var paths = split(filepath, '/\|\')
  var path = paths[-1]
  if len(paths) > 1
    var dir = join(paths[ : -2], '/')
    if !isdirectory(dir)
      mkdir(dir, 'p')
    endif
    path = dir .. '/' .. path
  endif

  writefile(texts, path .. '.dot')

  if executable('dot')
    echo system(
      'dot ' .. path .. '.dot' ..
      ' -T ' .. split(path, '\.')[-1] ..
      ' -o ' .. path
    )
  endif
enddef
