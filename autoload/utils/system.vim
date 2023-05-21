vim9script


export def Error(msg: string)
  echohl ErrorMsg
  echomsg 'autograd: ' .. msg
  echohl None
enddef
