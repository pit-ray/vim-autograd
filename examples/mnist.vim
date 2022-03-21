function!  s:linear(x, W, ...) abort
  let b = get(a:, 1, {})
  let t = autograd#matmul(a:x, a:W)
  return empty(b) ? t : autograd#add(t, b)
endfunction

function! s:sigmoid(x):
  return autograd#div(1, autograd#exp(x.n()).a(1))
endfunction

function! s:softmax(x) abort
  let y = autograd#exp(a:x)
  let s = autograd#sum(y)
  return autograd#div(y, s)
endfunction

let s:MLP = {'params': []}

function! s:MLP(in_size, h1_size, h2_size, out_size) abort
  let l:mlp = deepcopy(s:MLP)
  let W1 = autograd#randn(a:in_size, a:h1_size).m(0.01).detach()
  let b1 = autograd#zeros(a:h1_size)

  let W2 = autograd#randn(a:h1_size, a:h2_size).m(0.01).detach()
  let b2 = autograd#zeros(a:h2_size)

  let W3 = autograd#randn(a:h2_size, a:out_size).m(0.01).detach()
  let b3 = autograd#zeros(a:out_size)

  let self.params = [W1, b1, W2, b2, W3, b3]
endfunction

function! s:MLP.forward(x) abort
  let y = s:linear(a:x, self.params[0], self.params[1])
  let y = s:sigmoid(y)
  let y = s:linear(y, self.params[2], self.params[3])
  let y = s:sigmoid(y)
  let y = s:linear(y, self.params[3], self.params[4])
  let y = s:softmax(y)
  return y
endfunction


function! s:train() abort
  call s:srand(0)

  let lr = 0.1
  let iteration = 1000

  let s = s:softmax(autograd#as_tensor(

  for l:i in range(iteration)

  endfor





endfunction

call s:train()
