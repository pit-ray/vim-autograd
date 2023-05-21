vim9script

import '../autoload/autograd.vim' as ag

var Tensor = ag.Tensor


def TestClearGrad()
  var x = Tensor.new(2.0)
  var y = ag.Mul(60, x)

  ag.Backward(y)
  var gx: Tensor = x.grad
  assert_equal([60.0], gx.data)

  x.ClearGrad()
  assert_true(x.grad == null)
enddef


def TestClone()
  var a = Tensor.new(2.0)
  var b = ag.Clone(a)
  assert_notequal(a.id, b.id)
enddef


def TestTensor()
  var a = Tensor.new([[1, 2, 3], [4, 5, 6]])
  assert_equal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], a.data)
  assert_equal([2, 3], a.shape)

  var b = Tensor.new(5)
  assert_equal([5.0], b.data)
  assert_equal([1], b.shape)
enddef


def TestAsTensor()
  var a = Tensor.new([2, 4, 5])
  var ar = ag.AsTensor(a)
  assert_equal(a.id, ar.id)
  assert_true(a is ar)

  var b = [5, 6, 7]
  var br = ag.AsTensor(b)
  assert_equal([5.0, 6.0, 7.0], br.data)
  assert_equal([3], br.shape)

  var c = 5
  var cr = ag.AsTensor(c)
  assert_equal([5.0], cr.data)
  assert_equal([1], cr.shape)
enddef



def TestZeros()
  var d1 = float2nr(ag.Uniform(1.0, 10.0).data[0])
  var d2 = float2nr(ag.Uniform(1.0, 10.0).data[0])
  var d3 = float2nr(ag.Uniform(1.0, 10.0).data[0])

  var x = ag.Zeros([d1, d2, d3])
  assert_equal([d1, d2, d3], x.shape)
  for i in range(len(x.data))
    assert_equal(0.0, x.data[i])
  endfor
enddef


def TestZerosLike()
  var a = Tensor.new([
      [1, 5, 6],
      [2, 6, 4],
      [2, 6, 7]
    ])

  var b = ag.ZerosLike(a)
  assert_notequal(a.id, b.id)
  assert_equal(a.shape, b.shape)
  for i in range(len(b.data))
    assert_equal(0.0, b.data[i])
  endfor
enddef


def TestOnes()
  var d1 = float2nr(ag.Uniform(1.0, 10.0).data[0])
  var d2 = float2nr(ag.Uniform(1.0, 10.0).data[0])
  var d3 = float2nr(ag.Uniform(1.0, 10.0).data[0])

  var x = ag.Ones([d1, d2, d3])
  assert_equal([d1, d2, d3], x.shape)
  for i in range(len(x.data))
    assert_equal(1.0, x.data[i])
  endfor
enddef


def TestOnesLike()
  var a = Tensor.new([
      [2, 2, 6],
      [0, 3, 4],
      [7, 0, 5]
    ])

  var b = ag.OnesLike(a)
  assert_notequal(a.id, b.id)
  assert_equal(a.shape, b.shape)
  for i in range(len(b.data))
    assert_equal(1.0, b.data[i])
  endfor
enddef


def TestGeneration()
  var x = Tensor.new(2.0)

  var y = ag.Add(ag.Pow(ag.Pow(x, 2), 2), ag.Pow(ag.Pow(x, 2), 2))
  ag.Backward(y)
  var gx: Tensor = x.grad

  assert_equal([32.0], y.data)
  assert_equal([64.0], gx.data)
enddef


export def RunTestSuite()
  TestClearGrad()
  TestClone()
  TestTensor()
  TestAsTensor()
  TestZeros()
  TestZerosLike()
  TestOnes()
  TestOnesLike()
  TestGeneration()
enddef
