function! autograd#image#open_b(filename) abort
  return blob2list(readblob(a:filename))
endfunction

function! autograd#image#save_b(data, filename) abort
  call writefile(list2blob(a:data, a:filename))
endfunction
