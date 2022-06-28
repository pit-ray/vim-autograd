vim9script
scriptencoding utf-8


final ENABLE_BACKPROP = 1

final LAST_TENSOR_ID = 0
final LAST_FUNC_ID = v:numbermax / 2 - 1

const PI = acos(-1.0)

def Error(msg: string): void
  echohl ErrorMsg
  echomsg 'autograd: ' .. msg
  echohl None
enddef
