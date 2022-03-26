function!  s:linear(x, W, ...) abort
  let b = get(a:, 1, {})
  let t = autograd#matmul(a:x, a:W)
  return empty(b) ? t : autograd#add(t, b)
endfunction

function! s:relu(x) abort
  return autograd#maximum(a:x, 0.0)
endfunction

function! s:softmax(x) abort
  let y = autograd#exp(a:x.s(autograd#max(a:x)))
  let s = autograd#sum(y, 1, 1)
  return autograd#div(y, s)
endfunction

function! s:cross_entropy_loss(y, t)
  let loss = autograd#mul(a:t, autograd#log(a:y))
  let batch_size = loss.shape[0]
  return autograd#div(autograd#sum(loss), batch_size).n()
endfunction

let s:MLP = {'params': []}
function! s:MLP(in_size, ...) abort
  let l:mlp = deepcopy(s:MLP)

  " let std = sqrt(1 / a:in_size)
  " let std = 0.01
  let std = sqrt(2.0 / a:in_size)
  let l:W = autograd#normal(0, std, [a:in_size, a:1])
  let l:b = autograd#zeros([a:1])
  let l:W.name = 'W0'
  let l:b.name = 'b0'
  let l:mlp.params += [l:W, l:b]

  for l:i in range(a:0 - 1)
    let std = sqrt(2.0 / a:000[l:i])
    let l:W = autograd#normal(0, std, [a:000[l:i], a:000[l:i + 1]])
    let l:W.name = 'W' . string(l:i + 1)
    let l:b = autograd#zeros([a:000[l:i + 1]])
    let l:b.name = 'b' . string(l:i + 1)
    let l:mlp.params += [l:W, l:b]
  endfor
  return l:mlp
endfunction

function! s:MLP.forward(x) abort
  let y = s:linear(a:x, self.params[0], self.params[1])
  for l:i in range(2, len(self.params) - 1, 2)
    let y = s:relu(y)
    let y = s:linear(y, self.params[l:i], self.params[l:i + 1])
  endfor
  let y = s:softmax(y)
  return y
endfunction

let s:SGD = {
  \ 'vs': {},
  \ 'momentum': 0.9,
  \ 'lr': 0.01,
  \ 'weight_decay': 0.0,
  \ 'grad_clip': -1
  \ }
function! s:SGD.each_update(param) abort
  if self.weight_decay != 0
    call autograd#elementwise(
      \ [a:param.grad, a:param],
      \ {g, p -> g + self.weight_decay * p}, a:param.grad)
  endif

  if self.momentum == 0
    return autograd#elementwise(
      \ [a:param, a:param.grad], {p, g -> p - g * self.lr}, a:param)
  endif

  if !self.vs->has_key(a:param.id)
    let self.vs[a:param.id] = autograd#zeros_like(a:param)
  endif

  let v = self.vs[a:param.id]

  let v = autograd#sub(v.m(self.momentum), a:param.grad.m(self.lr))
  let self.vs[a:param.id] = v
  return autograd#elementwise([a:param, v], {a, b -> a + b}, a:param)
endfunction

function! s:SGD.step(params) abort
  " gradients clipping
  if self.grad_clip > 0
    let grads_norm = 0.0
    for param in a:params
      let grads_norm = autograd#sum(param.grad.p(2))
    endfor
    let grads_norm = autograd#sqrt(grads_norm).data[0]
    let clip_rate = self.grad_clip / (grads_norm + 0.000001)
    if clip_rate < 1.0
      for param in a:params
        let param.grad = param.grad.m(clip_rate)
      endfor
    endif
  endif

  call map(a:params, 'self.each_update(v:val)')
endfunction

function! s:SGD(...) abort
  let l:optim = deepcopy(s:SGD)
  let l:optim.lr = get(a:, 1, 0.01)
  let l:optim.momentum = get(a:, 2, 0.9)
  let l:optim.weight_decay = get(a:, 3, 0.0)
  let l:optim.grad_clip = get(a:, 4, -1)
  return l:optim
endfunction

function! s:get_wine_dataset() abort
  " This refers to the following public toy dataset.
  " https://archive.ics.uci.edu/ml/datasets/Wine
  let dataset = map(readfile('.autograd/wine.data'),
    \ "map(split(v:val, ','), 'str2float(v:val)')")

  let N = len(dataset)

  " average
  let means = repeat([0.0], 14)
  for data in dataset
    for l:i in range(1, 13)
      let means[l:i] += data[l:i]
    endfor
  endfor
  call map(means, 'v:val / N')

  " standard deviation
  let stds = repeat([0.0], 14)
  for data in dataset
    for l:i in range(1, 13)
      let stds[l:i] += pow(data[l:i] - means[l:i], 2)
    endfor
  endfor
  call map(stds, 'sqrt(v:val / N)')

  " standardization
  for data in dataset
    for l:i in range(1, 13)
      let data[l:i] = (data[l:i] - means[l:i]) / stds[l:i]
    endfor
  endfor

  " split the dataset into train and test.
  let train_x = []
  let train_t = []
  let test_x = []
  let test_t = []
  let test_num_per_class = 10
  for l:i in range(3)
    let class_split = autograd#data#shuffle(
      \ filter(deepcopy(dataset), 'v:val[0] == l:i + 1'))

    let train_split = class_split[:-test_num_per_class - 1]
    let test_split = class_split[-test_num_per_class:]

    let train_x += mapnew(train_split, 'v:val[1:]')
    let train_t += mapnew(train_split, "map(v:val[:0], 'v:val - 1')")
    let test_x += mapnew(test_split, 'v:val[1:]')
    let test_t += mapnew(test_split, "map(v:val[:0], 'v:val - 1')")
  endfor
  return {
    \ 'train': [train_x, train_t],
    \ 'test': [test_x, test_t],
    \ 'insize': len(train_x[0]),
    \ 'nclass': 3,
    \ 'mean': means[1:],
    \ 'std': stds[1:]
    \ }
endfunction

function! s:main() abort
  call autograd#manual_seed(42)

  let data = s:get_wine_dataset()
  let model = s:MLP(data['insize'], 100, data['nclass'])
  let optimizer = s:SGD(0.1, 0.9, 0.0001, 10.0)

  " train
  let max_epoch = 50
  let batch_size = 16
  let train_data_num = len(data['train'][0])
  let each_iteration = float2nr(ceil(1.0 * train_data_num / batch_size))

  let logs = []
  for epoch in range(max_epoch)
    let indexes = autograd#data#shuffle(range(train_data_num))
    let epoch_loss = 0
    for l:i in range(each_iteration)
      let x = []
      let t = []
      for index in indexes[l:i * batch_size:(l:i + 1) * batch_size - 1]
        call add(x, data['train'][0][index])

        let onehot = repeat([0.0], data['nclass'])
        let onehot[float2nr(data['train'][1][index][0])] = 1.0
        call add(t, onehot)
      endfor

      let y = model.forward(x)
      let loss = s:cross_entropy_loss(y, t)
      " call autograd#utils#dump_graph(loss, '.autograd/loss.png')

      for param in model.params
        call param.cleargrad()
      endfor
      call loss.backward()

      call optimizer.step(model.params)
      let l:epoch_loss += loss.data[0]
    endfor

    let l:epoch_loss /= each_iteration

    " logging
    call add(logs, epoch . ', ' . l:epoch_loss)
    call writefile(logs, '.autograd/train.log')
  endfor

  " evaluate
  let ng = autograd#no_grad()
  let accuracy = 0.0
  for l:i in range(len(data['test'][0]))
    let pred = model.forward([data['test'][0][l:i]])

    " argmax
    let class_idx = index(
      \ pred.data,
      \ max(map(pred.data, 'float2nr(v:val * 1000)')))
    let accuracy += class_idx == data['test'][1][l:i][0]
  endfor
  call ng.end()

  echomsg 'accuracy: ' . accuracy / len(data['test'][1])
endfunction

call s:main()
