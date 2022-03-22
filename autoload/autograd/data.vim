" Fisher-Yates shuffle
function! autograd#data#shuffle(data) abort
  for l:i in range(len(a:data) - 1, 1, -1)
    let l:j = float2nr((l:i + 0.99999) * autograd#random())
    let l:tmp = a:data[l:i]
    let a:data[l:i] = a:data[l:j]
    let a:data[l:j] = l:tmp
  endfor
  return a:data
endfunction
