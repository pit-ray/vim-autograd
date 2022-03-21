function!  s:linear(x, W, ...) abort
  let b = get(a:, 1, {})
  let t = autograd#matmul(a:x, a:W)
  return empty(b) ? t : autograd#add(t, b)
endfunction

function! s:sigmoid(x) abort
  return autograd#div(1, autograd#exp(a:x.n()).a(1))
endfunction

function! s:softmax(x) abort
  let y = autograd#exp(a:x)
  let s = autograd#sum(y, -1, 1)
  return autograd#div(y, s)
endfunction

function! s:cross_entropy(y, t)
  let loss = autograd#mul(a:t, autograd#log(a:y))
  let size = len(loss.data)
  return autograd#div(autograd#sum(loss), size).n()
endfunction

let s:MLP = {'params': []}
function! s:MLP(in_size, h1_size, out_size) abort
  let l:mlp = deepcopy(s:MLP)

  let W1 = autograd#randn(a:in_size, a:h1_size).m(0.01).detach()
  let W1.name = 'W1'

  let b1 = autograd#zeros([a:h1_size])
  let b1.name = 'b1'

  let W2 = autograd#randn(a:h1_size, a:out_size).m(0.01).detach()
  let W2.name = 'W2'

  let b2 = autograd#zeros([a:out_size])
  let b2.name = 'b2'

  let l:mlp.params = [W1, b1, W2, b2]

  return l:mlp
endfunction

function! s:MLP.forward(x) abort
  let y = s:linear(a:x, self.params[0], self.params[1])
  let y = s:sigmoid(y)
  let y = s:linear(y, self.params[2], self.params[3])
  let y = s:softmax(y)
  return y
endfunction


function! s:toy_cluster_dataset(sample_num) abort
  let out_x = []
  let out_t = []

  let x = autograd#uniform(-10.0, 10.0, [a:sample_num, 2]).data
  for l:i in range(a:sample_num)
    let xd = x[l:i * 2:l:i * 2 + 1]
    call add(out_x, xd)

    let c0 = xd[1] > tanh(xd[0])
    let c1 = xd[1] > sinh(xd[0])
    if c0 && !c1
      call add(out_t, [1.0, 0.0, 0.0, 0.0])
    elseif c0 && c1
      call add(out_t, [0.0, 1.0, 0.0, 0.0])
    elseif !c0 && c1
      call add(out_t, [0.0, 0.0, 1.0, 0.0])
    else
      call add(out_t, [0.0, 0.0, 0.0, 1.0])
    endif
  endfor
  return [out_x, out_t]
endfunction

function! s:shuffle(data) abort
  let size = len(a:data)
  for l:i in range(size - 1, 0, -1)
    let l:j = float2nr(autograd#uniform(0.0, l:i - 1, [1]).data[0])
    let l:tmp = copy(a:data[l:i])
    let a:data[l:i] = a:data[l:j]
    let a:data[l:j] = l:tmp
  endfor
  return a:data
endfunction

function! s:train() abort
  call srand(0)

  let lr = 0.1
  let max_epoch = 100
  let batch_size = 32

  let input_size = 2
  let hidden_size = 100
  let class_num = 4

  let data_num = 1000

  let data = s:toy_cluster_dataset(data_num)
  let model = s:MLP(input_size, hidden_size, class_num)

  let each_iteration = float2nr(ceil(data_num / batch_size))

  let logs = []
  for epoch in range(max_epoch)
    let indexes = s:shuffle(range(data_num))

    let epoch_loss = 0

    for l:i in range(each_iteration)
      let x = []
      let t = []
      for index in indexes[l:i * batch_size:(l:i + 1) * batch_size]
        call add(x, data[0][index])
        call add(t, data[1][index])
      endfor

      let y = model.forward(x)
      let loss = s:cross_entropy(y, t)

      " call autograd#utils#dump_graph(loss, '.autograd/loss.png')

      for param in model.params
        call param.cleargrad()
      endfor
      call loss.backward()

      let l:ng = autograd#no_grad()
      call map(model.params, 'autograd#sub(v:val, v:val.grad.m(lr))')
      call l:ng.end()

      let l:epoch_loss += loss.data[0]
    endfor

    let l:epoch_loss /= each_iteration
    call add(logs, epoch . '/' . max_epoch . '|' . l:epoch_loss)
    call writefile(logs, '.autograd/train.log')
  endfor

  return model
endfunction


function! s:inference(model) abort
  let results = []

  let ng = autograd#no_grad()
  for x_base in range(-100, 100)
    for y_base in range(-100, 100)
      let l:x = x_base / 10
      let l:y = y_base / 10

      let pred = a:model.forward([[l:x, l:y]])
      call map(pred.data, 'float2nr(v:val * 1000)')
      let class = index(pred.data, max(pred.data))
      call add(results, l:x . ', ' . l:y . ', ' . class)
      call writefile(results, '.autograd/result.log')
    endfor
  endfor
  call ng.end()
endfunction

function! s:main() abort
  let model = s:train()
  call s:inference(model)
endfunction

call s:main()
