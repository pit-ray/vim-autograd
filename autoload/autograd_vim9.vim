vim9script
scriptencoding utf-8

const PI = acos(-1.0)
final ENABLE_BACKPROP = 1
final LAST_FUNC_ID = v:numbermax / 2 - 1
final LAST_TENSOR_ID = 0

def Error(msg: string): void
  echohl ErrorMsg
  echomsg 'autograd: ' .. msg
  echohl None
enddef

function Tensor_cleargrad() dict abort
  let self.grad = {}
endfunction

function Tensor_set_parent_fn(parent_fn) dict abort
  let self.parent_fn = a:parent_fn
  let self.gen = self.parent_fn.gen + 1
endfunction

def CompareTensorGeneration(lhs: dict<any>, rhs: dict<any>): number
  if lhs['gen'] == rhs['gen']
    return 0
  elseif lhs['gen'] < rhs['gen']
    return -1
  endif
  return 1
enddef

function Tensor_backward(create_graph=v:false, retain_outgrad=v:false) dict abort
  if empty(self.grad)
    let self.grad = OnesLike(self)
  endif

  if empty(self.parent_fn)
    return
  endif

  let l:funcs = [self.parent_fn]
  let l:scanned_fn_ids = []
  while len(l:funcs) > 0
    let l:func = remove(l:funcs, -1)

    let l:gys = []
    for l:output in l:func.outputs
      call add(l:gys, l:output.grad)
    endfor

    " If create_graph is false, does not create graph in the following range.
    " ---------------------------------------------
    if !a:create_graph
      let l:ng = NoGrad()
    endif

    let l:gxs = l:func.backward(l:gys)

    let l:input_grad_ids = []
    let l:input_num = len(l:gxs)
    for l:i in range(l:input_num)
      let l:input = l:func.inputs[l:i]
      if empty(l:input.grad)
        let l:input.grad = l:gxs[l:i]
      else
        let l:input.grad = Add(l:input.grad, l:gxs[l:i])
      endif

      call add(l:input_grad_ids, l:input.grad.id)

      " It prevents multiple calling backward() of the same function.
      if !empty(l:input.parent_fn)
         \ && index(l:scanned_fn_ids, l:input.parent_fn.id) == -1
        call add(l:scanned_fn_ids, l:input.parent_fn.id)
        call add(l:funcs, l:input.parent_fn)
      endif
    endfor

    call sort(l:funcs, function('CompareTensorGeneration'))

    " Usually when we differentiate y=f(x) we are
    " interested in df/dx and do not need df/dy(=1) etc.
    " Therefore, we usually release.
    if !a:retain_outgrad
      for l:output in l:func.outputs
        if index(l:input_grad_ids, l:output.grad.id) == -1
          let l:output.grad = {}
        endif
      endfor
    endif

    if !a:create_graph
      call l:ng.end()
    endif
    " ---------------------------------------------
  endwhile
endfunction

function Tensor_a(x) dict abort
  return Add(self, a:x)
endfunction

function Tensor_m(x) dict abort
  return Mul(self, a:x)
endfunction

function Tensor_s(x) dict abort
  return Sub(self, a:x)
endfunction

function Tensor_d(x) dict abort
  return Div(self, a:x)
endfunction

function Tensor_p(x) dict abort
  return Pow(self, a:x)
endfunction

function Tensor_n() dict abort
  return Mul(self, -1)
endfunction

function Tensor_T() dict abort
  return Transpose(self)
endfunction

function Tensor_reshape(...) dict abort
  let l:shape = (a:0 == 1 && type(a:1) == v:t_list) ? a:1 : a:000
  return Reshape(self, l:shape)
endfunction

function Tensor_flatten() dict abort
  return Flatten(self)
endfunction

function Tensor_clone() dict abort
  return CreateTensor(copy(self.data), copy(self.shape))
endfunction

# It returns a new tensor detached from the current graph.
# However, returned tensor shares the same data and shape attribute.
function Tensor_detach() dict abort
  return CreateTensor(self.data, self.shape)
endfunction

def CreateTensor(data: list<float>, shape: list<number>): dict<any>
  var tensor = {
    'name': '',
    'id': 0,
    'data': null,
    'grad': {},
    'parent_fn': {},
    'gen': 0,
    'shape': [],
    'cleargrad': function('Tensor_cleargrad'),
    'set_parent_fn': function('Tensor_set_parent_fn'),
    'backward': function('Tensor_backward'),
    'a': function('Tensor_a'),
    'm': function('Tensor_m'),
    's': function('Tensor_s'),
    'd': function('Tensor_d'),
    'p': function('Tensor_p'),
    'n': function('Tensor_n'),
    'T': function('Tensor_T'),
    'reshape': function('Tensor_reshape'),
    'flatten': function('Tensor_flatten'),
    'clone': function('Tensor_clone'),
    'detach': function('Tensor_detach')
  }

  var tensor.data = data
  var tensor.shape = shape

  var tensor.id = last_tensor_id + 1
  var LAST_TENSOR_ID = tensor.id
  return tensor
enddef

def IsTensor(x: any): bool
  if type(x) != v:t_dict
    return false
  endif
  return has_key(x, 'data') && has_key(a:x, 'grad')
enddef

def CreateVector(size: number, init_val: float = 0.0): list<float>
  var vec = repeat([0.0], size)
  return init_val != 0.0 ? map(vec, init_val) : vec
enddef

def ShapeToSize(shape: list<number>): number
  var size = 1
  for x in shape
    var size *= x
  endfor
  return size
enddef

def GetMatrixShape(array: any): list<number>
  var shape = []
  var sub_array = copy(array)
  while type(sub_array) == v:t_list
    add(shape, len(sub_array))
    var sub_array = sub_array[0]
  endwhile
  return shape
enddef

def AsList(data: any): list<float>
  return type(data) != v:t_list ? [data] : data
enddef

export def Tensor(rawdata: any): dict<any>
  var data = AsList(deepcopy(rawdata))

  var shape = GetMatrixShape(data)
  var data = flatten(data)

  map(data, 'v:val * 1.0')  " int to float

  if len(data) != ShapeToSize(shape)
    Error('Invalid matrix shape.')
  endif
  return CreateTensor(data, shape)
enddef

export def AsTensor(data: any): dict<any>
  return IsTensor(data) ? data : CreateTensor(data)
enddef

export def Zeros(shape: list<number>): dict<any>
  var size = ShapeToSize(a:shape)
  if size == 0
    Error('axis without element is invalid.')
  endif
  return CreateTensor(CreateVector(size, 0.0), shape)
enddef

export def ZerosLike(tensor: dict<any>): dict<any>
  return CreateTensor(CreateVector(len(tensor.data), 0.0), tensor.shape)
enddef

export def Ones(shape: list<number>): dict<any>
  var size = ShapeToSize(shape)
  if size == 0
    Error('axis without element is invalid.')
  endif
  return CreateTensor(CreateVector(size, 1.0), shape)
enddef

export def OnesLike(tensor: dict<any>): dict<any>
  return CreateTensor(CreateVector(len(tensor.data), 1.0), tensor.shape)
enddef

final random_seed = srand()
export def ManualSeed(seed: number): void
  var random_seed = srand(seed)
enddef

export def Random(): float
  " it returns random value from 0.0 to 1.0.
  return rand(random_seed) / 4294967295.0
enddef

def BoxMuller(u1: float, u2: float): float
  return sqrt(-2 * log(u1)) * cos(2 * PI * u2)
enddef

export def Rand(...shape: list<number>): dict<any>
  var shape = len(shape) > 0 ? shape : [1]
  var size = ShapeToSize(shape)
  var data = map(repeat([0.0], size), 'Random()')
  return CreateTensor(data, shape)
enddef

export def Uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: list<number> = [1]): dict<any>
  var size = ShapeToSize(shape)
  var data = map(
    repeat([0.0], l:size),
    'low + (high - low) * Random()')
  return CreateTensor(data, shape)
enddef

export def Randn(...shape: list<number>): dict<any>
  var shape = len(shape) > 0 ? shape : [1]
  var size = ShapeToSize(shape)
  var data = map(
    repeat([0.0], size),
    'BoxMuller(Random(), Random())')
  return CreateTensor(l:data, l:shape)
enddef

export def Normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: list<number> = [1]): dict<any>
  var size = ShapeToSize(shape)
  var data = map(
    repeat([0.0], size),
    'std * BoxMuller(Random(), Random()) + mean')
  return CreateTensor(data, shape)
enddef


function Function_apply(...) dict abort
  let l:inputs = []
  for l:input in a:000
    call add(l:inputs, AsTensor(l:input))
  endfor

  let l:outputs = self.forward(l:inputs)

  if ENABLE_BACKPROP
    let l:gens = []
    for l:input in l:inputs
      call add(l:gens, l:input.gen)
    endfor
    let self.gen = max(l:gens)

    for l:output in l:outputs
      call l:output.set_parent_fn(self)
    endfor

    let self.inputs = l:inputs
    let self.outputs = l:outputs
  endif

  return len(l:outputs) > 1 ? l:outputs : l:outputs[0]
endfunction

export def CreateFunction(name: string): dict<any>
  var fn = {
    'name': name,
    'id': LAST_FUNC_ID + 1,
    'inputs': [],
    'outputs': [],
    'gen': 0,
    'forward': function(name .. 'Forward'),
    'backward': function(name .. 'Backward')
  }
  var LAST_FUNC_ID = fn.id
  return fn
enddef


# Operations
export def Add(x0: any, x1: any): dict<any>
  return CreateFunction('Add').apply(x0, x1)
enddef

def AddCore(x0: float, x1: float): float
  return x0 + x1
enddef

function AddForward(xs) dict abort
  return [Elementwise(a:xs, function('Add_'))]
endfunction

function AddBackward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]

  let l:gx0 = a:gys[0]
  let l:gx1 = a:gys[0]

  if l:x0.shape == l:x1.shape
    return [l:gx0, l:gx1]
  endif
  return [SumTo(l:gx0, l:x0.shape), SumTo(l:gx1, l:x1.shape)]
endfunction


export def Mul(x0: any, x1: any): dict<any>
  return CreateFunction('Mul').apply(x0, x1)
enddef

def MulCore(x0: float, x1: float): float
  return x0 * x1
enddef

function MulForward(xs) dict abort
  return [Elementwise(a:xs, function('Mul_'))]
endfunction

function MulBackward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]

  let l:gx0 = l:x1.m(a:gys[0])
  let l:gx1 = l:x0.m(a:gys[0])

  if l:x0.shape == l:x1.shape
    return [l:gx0, l:gx1]
  endif
  return [SumTo(l:gx0, l:x0.shape), SumTo(l:gx1, l:x1.shape)]
endfunction


export def Sub(x0: any, x1: any): dict<any>
  return CreateFunction('Sub').apply(x0, x1)
enddef

def SubCore(x0: float, x1: float): float
  return x0 - x1
enddef

function SubForward(xs) dict abort
  return [Elementwise(a:xs, function('Sub_'))]
endfunction

function SubBackward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]

  let l:gx0 = a:gys[0]
  let l:gx1 = a:gys[0].n()

  if l:x0.shape == l:x1.shape
    return [l:gx0, l:gx1]
  endif
  return [SumTo(l:gx0, l:x0.shape), SumTo(l:gx1, l:x1.shape)]
endfunction


export def Div(x0: any, x1: any): dict<any>
  return CreateFunction('Div').apply(x0, x1)
enddef

function DivCore(x0, x1) abort
  return a:x0 / a:x1
endfunction

function DivForward(xs) dict abort
  return [Elementwise(a:xs, function('Div_'))]
endfunction

function DivBackward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]

  let l:gx0 = a:gys[0].d(l:x1)

  " gx1 = gy * -(x0 / x1 ** 2)
  let l:gx1 = Mul(a:gys[0], l:x0.d(l:x1.p(2)).n())

  if l:x0.shape == l:x1.shape
    return [l:gx0, l:gx1]
  endif
  return [SumTo(l:gx0, l:x0.shape), SumTo(l:gx1, l:x1.shape)]
endfunction


export def Pow(x: any, c: any): dict<any>
  return CreateFunction('Pow').apply(x, c)
enddef

function PowForward(xs) dict abort
  return [Elementwise(a:xs, {a, b -> pow(a, b)})]
endfunction

function PowBackward(gys) dict abort
  let l:x = self.inputs[0]
  let l:c = self.inputs[1]
  let l:y = self.outputs[0]

  " gx = gy * c * x**(c - 1)
  let l:gx = Mul(a:gys[0], l:c.m(l:x.p(l:c.s(1))))

  " gc = gy * y * log(x)
  let l:gc = Mul(a:gys[0], l:y.m(Log(l:x)))

  if l:x.shape == l:c.shape
    return [l:gx, l:gc]
  endif
  return [SumTo(l:gx, l:x.shape), SumTo(l:gc, l:c.shape)]
endfunction


def Sqrt(x: any): dict<any>
  return Pow(a:x, 0.5)
enddef


export def Log(x: any): dict<any>
  return CreateFunction('Log').apply(x)
enddef

function LogForward(xs) dict abort
  return [Elementwise(a:xs, {a -> log(a)})]
endfunction

function LogBackward(gys) dict abort
  let l:x = self.inputs[0]
  return [Div(a:gys[0], l:x)]
endfunction


export def Exp(x: any): dict<any>
  return CreateFunction('Exp').apply(x)
enddef

function ExpForward(xs) dict abort
  return [Elementwise(a:xs, {a -> exp(a)})]
endfunction

function ExpBackward(gys) dict abort
  let l:y = self.outputs[0]
  return [Mul(a:gys[0], l:y)]
endfunction


export def Sin(x: any): dict<any>
  return CreateFunction('Sin').apply(x)
enddef

function SinForward(xs) dict abort
  return [Elementwise(a:xs, {a -> sin(a)})]
endfunction

function SinBackward(gys) dict abort
  let l:x = self.inputs[0]
  return [Mul(a:gys[0], Cos(l:x))]
endfunction


export def Cos(x: any): dict<any>
  return CreateFunction('Cos').apply(x)
enddef

function CosForward(xs) dict abort
  return [Elementwise(a:xs, {a -> cos(a)})]
endfunction

function CosBackward(gys) dict abort
  let l:x = self.inputs[0]
  return [Mul(a:gys[0], Sin(l:x).n())]
endfunction


export def Tanh(x: any): dict<any>
  return CreateFunction('Tanh').apply(x)
enddef

function TanhForward(xs) dict abort
  return [Elementwise(a:xs, {a -> tanh(a)})]
endfunction

function TanhBackward(gys) dict abort
  let l:y = self.outputs[0]
  return [Mul(a:gys[0], Sub(1, l:y.p(2)))]
endfunction


export def Abs(x: any): dict<any>
  return CreateFunction('Abs').apply(x)
enddef

function AbsForward(xs) dict abort
  return [Elementwise(a:xs, {a -> abs(a)})]
endfunction

function AbsBackward(gys) dict abort
  let l:x = self.inputs[0]
  return [Mul(a:gys[0], Sign(l:x))]
endfunction


export def Sign(x: any): dict<any>
  return CreateFunction('Sign').apply(x)
enddef

def SignCore(x: float): float
  return x > 0.0 ? 1.0 : (x < -1.0 ? -1.0 : 0.0)
enddef

function SignForward(xs) dict abort
  return [Elementwise(a:xs, function('Sign_'))]
endfunction

function SignBackward(gys) dict abort
  return [Mul(a:gys[0], 0.0)]
endfunction


def LeftSideSumTo(x: dict<any>, shape: list<number>): dict<any>
  var y = Zeros(shape)

  var xd = x.data
  var yd = y.data

  var x_size = len(xd)
  var y_size = len(yd)

  for i in range(x_size / y_size)
    var base = i * y_size
    for j in range(y_size)
      let yd[j] += xd[base + j]
    endfor
  endfor
  return y
enddef

def RightSideSumTo(x: dict<any>, shape: list<number>): dict<any>
  var y = Zeros(shape)

  var xd = x.data
  var yd = y.data

  var x_size = len(xd)
  var y_size = len(yd)

  var block_size = x_size / y_size
  for i in range(y_size)
    var base = block_size * i
    for j in range(block_size)
      var yd[i] += xd[base + j]
    endfor
  endfor
  return y
enddef


export def Sum(
    x: any,
    axis: list<number> = [],
    keepdims: bool = false): dict<any>
  var axis = AsList(axis)

  var x_dim = len(x.shape)
  map(axis, 'v:val < 0 ? v:val + x_dim : v:val')
  map(axis, 'v:val >= x_dim ? x_dim - 1 : v:val')

  var fn = CreateFunction('Sum')
  var fn['axis'] = uniq(sort(axis))
  var fn['keepdims'] = keepdims
  return fn.apply(x)
enddef

function SumForward(xs) dict abort
  let l:x = a:xs[0]

  " all sum (e.g. (2, 3, 4) -> (1))
  if empty(self.axis) || len(self.axis) == len(l:x.shape)
    let l:total = 0
    for l:e in l:x.data
      let l:total += l:e
    endfor

    if !self.keepdims
      return [CreateTensor([l:total], [1])]
    endif
    return [CreateTensor([l:total], repeat([1], len(l:x.shape)))]
  endif

  " left side sum (e.g. (2, 3, 4) -> (3, 4))
  if self.axis[0] == 0
    let l:reduced_shape = l:x.shape[len(self.axis):]
    let l:s = LeftSideSumTo(l:x, l:reduced_shape)

    if self.keepdims
      let l:s.shape = repeat([1], len(self.axis)) + l:reduced_shape
    endif
    return [l:s]
  endif

  " right side sum (e.g. (2, 3, 4) -> (2, 3)
  if self.axis[-1] == (len(l:x.shape) - 1)
    let l:reduced_shape = l:x.shape[:-len(self.axis) - 1]
    let l:s = RightSideSumTo(l:x, l:reduced_shape)

    if self.keepdims
      let l:s.shape = l:reduced_shape + repeat([1], len(self.axis))
    endif
    return [l:s]
  endif

  call Error('intermediate or sparse axis sums are not supported.')
  return
endfunction

function SumBackward(gys) dict abort
  return [BroadcastTo(a:gys[0], self.inputs[0].shape)]
endfunction


# e.g. [1, 5, 6, 1, 1, 1] -> [1, 5, 6]
def LeftValidShape(shape: list<number>): list<number>
  var dim = len(shape)
  var valid_size = dim
  for i in range(-1, -dim, -1)
    if shape[i] != 1
      break
    endif
    var valid_size -= 1
  endfor

  if valid_size == 0
    return [1]
  endif

  return shape[:valid_size - 1]
enddef

# e.g. [1, 1, 1, 6, 7, 1] -> [6, 7, 1]
def RightValidShape(shape: list<number>): list<number>
  var dim = len(shape)
  var valid_size = dim
  for i in range(dim)
    if shape[i] != 1
      break
    endif
    var valid_size -= 1
  endfor

  if valid_size == 0
    return [1]
  endif

  return shape[-valid_size:]
enddef


export def BroadcastTo(x: any, shape: list<number>): dict<any>
  var xt = AsTensor(x)
  if xt.shape == shape
    return xt
  endif

  var fn = CreateFunction('BroadcastTo')
  var fn['shape'] = shape
  return fn.apply(x)
enddef

function BroadcastToForward(xs) dict abort
  let l:x = a:xs[0]
  let l:x_dim = len(l:x.shape)

  if l:x_dim > len(self.shape)
    Error(
      \ 'cannot broadcast the array of ' .
      \ string(l:x.shape) . ' to ' . string(self.shape))
  endif

  let l:size = ShapeToSize(self.shape)

  " left side broadcast
  let l:right_subshape = RightValidShape(l:x.shape)
  if l:right_subshape == [1]
    return [CreateTensor(repeat(l:x.data, l:size), self.shape)]
  endif
  if self.shape[-len(l:right_subshape):] == l:right_subshape
    let l:repeat = float2nr(l:size / len(l:x.data))
    return [CreateTensor(repeat(l:x.data, l:repeat), self.shape)]
  endif

  " right side broadcast
  let l:left_subshape = LeftValidShape(l:x.shape)
  if self.shape[:len(l:left_subshape) - 1] == l:left_subshape
    let l:repeat = float2nr(l:size / len(l:x.data))
    return [CreateTensor(flatten(
      \ mapnew(l:x.data, 'repeat([v:val], l:repeat)')), self.shape)]
  endif

  call Error(
    \ 'cannot broadcast array of shape ' .
    \ string(l:x.shape) . ' into ' . string(self.shape))
endfunction

function BroadcastToBackward(gys) dict abort
  return [SumTo(a:gys[0], self.inputs[0].shape)]
endfunction


export def SumTo(x: any, shape: list<number>): dict<any>
  var xt = AsTensor(x)
  if xt.shape == shape
    return xt
  endif
  var fn = CreateFunction('SumTo')
  var fn['shape'] = shape
  return l:fn.apply(x)
enddef

function SumToForward(xs) dict abort
  let l:x = a:xs[0]
  let l:y = Zeros(self.shape)

  let l:y_dim = len(self.shape)

  " left side sum
  let l:right_subshape = RightValidShape(self.shape)
  if l:right_subshape == [1] || l:x.shape[-len(right_subshape):] == l:right_subshape
    return [LeftSideSumTo(l:x, self.shape)]
  endif

  " right side sum
  let l:left_subshape = LeftValidShape(self.shape)
  if l:x.shape[:len(l:left_subshape) - 1] == l:left_subshape
    return [RightSideSumTo(l:x, self.shape)]
  endif

  call Error('cannot sum from ' . string(l:x.shape) . ' into ' . string(self.shape))
endfunction

function SumToBackward(gys) dict abort
  return [BroadcastTo(a:gys[0], self.inputs[0].shape)]
endfunction


export def FloatMax(list_obj: list<float>): float
  let max = 0.0
  for x in list_obj
    if max < x
      let max = x
    endif
  endfor
  return max
enddef

export def Max(x: any): dict<any>
  return CreateFunction('Max').apply(x)
enddef

function MaxForward(xs) dict abort
  return [CreateTensor([FloatMax(a:xs[0].data)], [1])]
endfunction

function MaxBackward(gys) dict abort
  let l:x = self.inputs[0]
  let l:y = self.outputs[0]
  let l:gx_mask = Elementwise([l:x, l:y], {a, b -> 1.0 * (a == b)})
  let l:gx = Mul(a:gys[0], l:gx_mask)
  return [l:gx]
endfunction


export def Maximum(a: any, b: any): dict<any>
  return CreateFunction('Maximum').apply(a, b)
enddef

function MaximumForward(xs) dict abort
  return [Elementwise(a:xs, {a, b -> a >= b ? a : b})]
endfunction

function MaximumBackward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]

  let l:gx0_mask = Elementwise([l:x0, l:x1], {a, b -> a >= b})
  let l:gx1_mask = Elementwise([l:x0, l:x1], {a, b -> a < b})

  let l:gx0 = Mul(a:gys[0], l:gx0_mask)
  let l:gx1 = Mul(a:gys[0], l:gx1_mask)

  return [SumTo(l:gx0, l:x0.shape), SumTo(l:gx1, l:x1.shape)]
endfunction


export def Transpose(x: any): dict<any>
  return CreateFunction('Transpose').apply(x)
enddef

def TransposeCore(x: dict<any>): dict<any>
  var dim = len(x.shape)
  if dim > 2
    Error('transpose() is supported only for 1D-tensor and 2D-tensor.')
  endif

  if dim == 1
    return x
  endif

  var xd = x.data

  var out_data = CreateVector(len(xd))

  var n_i = x.shape[0]
  var n_j = x.shape[1]

  var n_j_range = range(n_j)
  for i in range(n_i)
    var buf = i * n_j
    for j in n_j_range
      var out_data[j * n_i + i] = xd[buf + j]
    endfor
  endfor

  return CreateTensor(out_data, [n_j, n_i])
enddef

function TransposeForward(xs) dict abort
  return [TransposeCore(a:xs[0])]
endfunction

function TransposeBackward(gys) dict abort
  return [Transpose(a:gys[0])]
endfunction


export def Matmul(a: any, b: any): dict<any>
  return CreateFunction('Matmul').apply(a, b)
enddef

function MatmulForward(xs) dict abort
  let l:x0 = a:xs[0]
  let l:x1 = a:xs[1]

  let l:x0_dim = len(l:x0.shape)
  let l:x1_dim = len(l:x1.shape)

  if l:x0_dim > 2 || l:x1_dim > 2
    call Error('inputs must be 2D-2D or 1D-1D.')
    return
  endif

  let l:x0_shape = copy(l:x0.shape)
  let l:x1_shape = copy(l:x1.shape)
  let self['x0_shape_fix'] = l:x0_shape
  let self['x1_shape_fix'] = l:x1_shape

  " 1D-tensor is converted to 2D-tensor
  if l:x0_dim == 1
    call insert(l:x0_shape, 1, 0)
  endif
  if l:x1_dim == 1
    call add(l:x1_shape, 1)
  endif

  if l:x0_shape[1] != l:x1_shape[0]
    call Error('axis 1 of left operand mismatchs axis 0 of right.')
  endif

  let l:n_i = l:x0_shape[0]
  let l:n_k = l:x0_shape[1]
  let l:n_j = l:x1_shape[1]

  let l:out = Zeros([l:n_i, l:n_j])

  let l:od = l:out.data
  let l:d0 = l:x0.data
  let l:d1 = l:x1.data

  " 2D matrix product (ikj-algorithm)
  let l:n_k_range = range(l:n_k)
  let l:n_j_range = range(l:n_j)
  for l:i in range(l:n_i)
    for l:k in l:n_k_range
      let l:buf = l:d0[l:i * l:n_k + l:k]
      for l:j in l:n_j_range
        let l:od[l:i * l:n_j + l:j] += l:buf * l:d1[l:k * l:n_j + l:j]
      endfor
    endfor
  endfor

  " If one is 1D, output in 1D
  if l:x0_dim == 1
    call remove(l:out.shape, 0)
  elseif l:x1_dim == 1
    call remove(l:out.shape, 1)
  endif

  return [l:out]
endfunction

function MatmulBackward(gys) dict abort
  let l:x0 = self.inputs[0]
  let l:x1 = self.inputs[1]
  let l:gy = a:gys[0]

  let l:x0_shape_raw = l:x0.shape
  let l:x1_shape_raw = l:x1.shape

  " temporarily restores the shape of x when y is calculated.
  let l:x0.shape = self.x0_shape_fix
  let l:x1.shape = self.x1_shape_fix

  let l:gx0 = Matmul(l:gy, l:x1.T())
  let l:gx1 = Matmul(l:x0.T(), l:gy)

  " return to the original shape
  let l:x0.shape = l:x0_shape_raw
  let l:x1.shape = l:x1_shape_raw

  return [l:gx0, l:gx1]
endfunction


export def Reshape(x: any, shape: list<number>): dict<any>
  if x.shape == shape
    return x
  endif

  var fn = CreateFunction('Reshape')
  var fn['shape'] = shape
  return fn.apply(x)
enddef

function ReshapeForward(xs) dict abort
  let l:x = a:xs[0]
  if ShapeToSize(self.shape) != len(l:x.data)
    call Error('Cannot reshape array of size ' . len(l:x.data). ' into ' . string(self.shape))
    return
  endif
  return [CreateTensor(l:x.data, self.shape)]
endfunction

function ReshapeBackward(gys) dict abort
  return [Reshape(a:gys[0], self.inputs[0].shape)]
endfunction


def Flatten(x: any): dict<any>
  return Reshape(x, [len(x.data)])
enddef


export def Pi(): float
  return PI
enddef

export def Grad(
    output: dict<any>,
    inputs: any,
    create_graph: bool = false,
    retain_outgrad: bool = false): dict<any>
  var xs = IsTensor(inputs) ? [inputs] : inputs

  var old_grads = []
  for x in xs
    add(old_grads, x.grad)
    x.cleargrad()
  endfor

  output.backward(create_graph, retain_outgrad)

  var grads = []
  for i in range(len(xs))
    add(grads, xs[i].grad)
    var xs[i].grad = old_grads[i]
  endfor

  return len(grads) > 1 ? grads : grads[0]
enddef


function NoGrad_end() dict abort
  let ENABLE_BACKPROP = self.state
endfunction

export def NoGrad(): dict<any>
  var ng = {
    'state': ENABLE_BACKPROP,
    'end', function('NoGrad_end'),
  }
  var ENABLE_BACKPROP = 0
  return ng
enddef


export def Elementwise(
    inputs: list<dict<any>>,
    fn: func(...list<float>): float,
    out: dict<any> = {}): dict<any>
  if len(inputs) == 1
    var x = inputs[0]
    var tensor = empty(out) ? ZerosLike(x) : out

    var td = tensor.data
    var xd = x.data
    for i in range(len(xd))
      var td[i] = fn(xd[i])
    endfor
    return tensor
  endif

  var x0 = inputs[0]
  var x1 = inputs[1]

  var ng = NoGrad()

  var x0_dim = len(x0.shape)
  var x1_dim = len(x1.shape)

  if x0_dim > x1_dim
    var x1 = BroadcastTo(x1, x0.shape)
  elseif x0_dim < x1_dim
    var x0 = BroadcastTo(x0, x1.shape)
  else
    if len(x0.data) > len(x1.data)
      var x1 = BroadcastTo(x1, x0.shape)
    else
      var x0 = BroadcastTo(x0, x1.shape)
    endif
  endif
  ng.end()

  var tensor = empty(out) ? ZerosLike(x0) : out

  var td = tensor.data
  var x0d = x0.data
  var x1d = x1.data

  for i in range(len(tensor.data))
    var td[i] = fn(x0d[i], x1d[i])
  endfor
  return tensor
enddef

# Utilities

# Fisher-Yates shuffle algorithm
export def Shuffle(data: list<any>): list<any>
  for i in range(len(data) - 1, 1, -1)
    var j = float2nr((i + 0.99999) * Random())
    var tmp = data[i]
    var data[i] = data[j]
    var data[j] = tmp
  endfor
  return data
enddef

def IsClose(
    a: float,
    b: float,
    rtol: float = 0.00001,
    atol: float = 0.00000001): bool
  return abs(a - b) <= (atol + rtol * abs(b))
enddef

def AllClose(
    a: float,
    b: float,
    rtol: float = 0.00001,
    atol: float = 0.00000001): bool
  var results = Elementwise([a, b], {x, y -> IsClose(x, y, rtol, atol)})
  return min(results.data) == 1
enddef

def DumpTensorAsDotlang(tensor: dict<any>): string
  return tensor.id .. '[label="' .. tensor.name .. '", color=lightblue, style=filled]'
enddef

def DumpFuncAsDotlang(fn: dict<any>): string
  var fndef = fn.id .. '[label="' .. fn.name .. '", color=gray, style=filled, shape=box]'

  var links = []
  for x in fn.inputs
    add(links, x.id .. ' -> ' .. fn.id)
  endfor

  for y in fn.outputs
    add(links, fn.id .. ' -> ' .. y.id)
  endfor

  return [fndef, links]
enddef

export def DumpGraph(last_node: dict<any>, filepath: string): void
  var defs = [DumpTensorAsDotlang(last_node)]
  var links = []
  var funcs = [last_node.parent_fn]

  while len(funcs) > 0
    var func = remove(funcs, -1)
    var fn_dot = DumpFuncAsDotlang(func)
    add(defs, fn_dot[0])
    var links += fn_dot[1]

    for x in func.inputs
      add(defs, DumpTensorAsDotlang(x))

      if !empty(x.parent_fn)
        add(funcs, x.parent_fn)
      endif
    endfor
  endwhile

  var links = uniq(sort(links))

  var texts = ['digraph g {'] + defs + links + ['}']

  var paths = split(filepath, '/\|\')
  var path = paths[-1]
  if len(paths) > 1
    var dir = join(paths[:-2], '/')
    if !isdirectory(dir)
      mkdir(dir, 'p')
    endif
    var path = dir .. '/' .. path
  endif

  writefile(texts, path .. '.dot')

  if executable('dot')
    echo system(
      'dot ' .. path .. '.dot' ..
      ' -T ' .. split(path , '\.')[-1] ..
      ' -o ' .. path
    )
  endif
enddef

export def NumericalGrad(fn: dict<any>, x: any): dict<any>
  var eps = Tensor(0.000001)
  var dx = Tensor(eps.data[0] * 2)

  var x0 = Sub(x, eps)
  var x1 = Add(x, eps)

  var y0 = fn(x0)
  var y1 = fn(x1)

  var dy = Sub(y1, y0)
  return SumTo(Div(dy, dx), x.shape)
enddef

export def GradCheck(fn: dict<any>, inputs: list<any>): bool
  var y = f(inputs)

  for x in inputs
    x.cleargrad()
  endfor
  y.backward()

  var grads = []
  for x in inputs
    add(grads, x.grad)
  endfor

  var ng = NoGrad()

  var input_num = len(inputs)
  for i in range(input_num)
    var before_args = i > 0 ? inputs[:i - 1] : []
    var after_args = i < input_num - 1 ? inputs[i + 1:] : []

    var num_grad = NumericalGrad(
      {x -> f(before_args + [x] + after_args)},
      inputs[i])

    assert_true(AllClose(grads[i], num_grad))
  endfor

  ng.end()
enddef
