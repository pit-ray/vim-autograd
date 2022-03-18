# vim-autograd
[![test](https://github.com/pit-ray/vim-autograd/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/pit-ray/vim-autograd/actions/workflows/test.yml)  
Define-by-Run style automatic differentiation library written in pure Vim Script.  

It uses the same algorithm as Chainer and PyTorch to perform differentiation by generating a computational graph at runtime, making it easy to obtain derivative values even for complex expressions.

It is based on [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3).

**However it is still under development.**

## ToDo
- [x] support basic operations.
- [x] support higher-order differentiation.
- [ ] support for basic mathematical functions (e.g. sin(), cos()) supported by Vim.
- [ ] support matrix.
- [ ] add documentation.

## Usage

```vim
function! s:f(x) abort
  " y = x^5 - 2x^3
  let y = autograd#sub(a:x.p(5), a:x.p(3).m(2))
  return y
endfunction

function! s:example1() abort
  let x = autograd#tensor(2.0)

  let y = s:f(x)
  call y.backward()

  " output: 56
  echo x.grad.data
endfunction

call s:example1()
```

```
56
```

## Installation
```vim
Plug 'pit-ray/vim-autograd'
```

## License
This library is provided by **MIT License**.

## Author
- pit-ray
