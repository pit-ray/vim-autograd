vim9script

var is_backprop_enabled = true


export def NoGrad(Fn: func)
  var state = is_backprop_enabled
  try
    is_backprop_enabled = false
    Fn()
  finally
    is_backprop_enabled = state
  endtry
enddef


export def IsBackpropEnabled(): bool
  return is_backprop_enabled
enddef
