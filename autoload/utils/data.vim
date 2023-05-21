vim9script

import '../core/random.vim'


export def Shuffle(data: list<any>): list<any>
  # Fisher-Yates shuffle algorithm
  for i in range(len(data) - 1, 1, -1)
    var j = float2nr((i + 0.99999) * random.Random())
    var tmp = data[i]
    data[i] = data[j]
    data[j] = tmp
  endfor
  return data
enddef
